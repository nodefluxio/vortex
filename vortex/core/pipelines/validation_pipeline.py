import os
from pathlib import Path

from easydict import EasyDict
from typing import Union
import warnings


from vortex.utils.parser import check_config
from vortex.utils.common import check_and_create_output_dir
from vortex.core.factory import create_model,create_dataset
from vortex.utils.reporting.report import generate_reports
from vortex_runtime import model_runtime_map
from vortex.core.pipelines.base_pipeline import BasePipeline
from vortex.core import engine as engine

__all__ = ['PytorchValidationPipeline','IRValidationPipeline']

class BaseValidationPipeline(BasePipeline):
    """Vortex Base Validation Pipeline

    Args:
        BasePipeline : Base class for Vortex Pipeline
    """

    def __init__(self,
                 config : EasyDict,
                 backends : Union[list,str]=[],
                 generate_report : bool = True,
                 hypopt : bool =False
                 ) :
        """Class initialization

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            backends (Union[list,str], optional): devices or runtime to be used for model's computation. Defaults to [].
            generate_report (bool, optional): if enabled will generate validation report in markdown format. Defaults to True.
            hypopt (bool, optional): flag for hypopt, disable several pipeline process. Defaults to False.

        Raises:
            RuntimeError: raise error if experiment config is not valid for validation
        """
        # Check config
        check_result = check_config(config, 'validate')
        if not check_result.valid:
            raise RuntimeError("invalid config : %s" % str(check_result))
        self.hypopt = hypopt
        self.generate_report = generate_report

        # Output directory check and set
        self.experiment_name = config.experiment_name
        self.experiment_directory , _ = check_and_create_output_dir(config)
        self.reports_dir = None
        self.assets_dir = None
        if self.generate_report and not hypopt:
            self.reports_dir = Path(self.experiment_directory)/'reports'
            if not self.reports_dir.exists():
                self.reports_dir.mkdir(exist_ok=True, parents=True)
            self.assets_dir = self.reports_dir / 'assets'
            if not self.assets_dir.exists():
                self.assets_dir.mkdir(exist_ok=True, parents=True)

        # Compute devices check
        if isinstance(backends,str):
            backends = [backends]
        if len(backends) != 0:
            self.backends = backends
        else:
            if 'device' in config:
                device = config.device
            elif 'device' in config.trainer:
                device = config.trainer.device
            else:
                raise RuntimeError("'device' field not found in config. Please specify properly in main level.")
            self.backends = [device]

        # Must be initizalized in sub-class
        self.model = None
        self.filename_suffix = None

        # Dataset initialization
        # TODO selection to validate also on training data
        self.dataset = create_dataset(config.dataset, config.model.preprocess_args,stage='validate')
        if 'name' in config.dataset.eval:
            dataset_name = config.dataset.eval.name
        elif 'dataset' in config.dataset.eval:
            dataset_name = config.dataset.eval.dataset
        else:
            raise RuntimeError("Dataset name in 'config.dataset.eval.name' is not set "
                "in config.dataset ({}).".format(config.dataset.eval))
        self.dataset_info = ('eval', dataset_name)

        # Validator arguments
        if 'validator' in config:
            validator_cfg = config.validator
        elif 'validation' in config.trainer:
            validator_cfg = config.trainer.validation
        else:
            raise RuntimeError("Validator config in 'config.validator' is not set.")
        self.validation_args = validator_cfg.args
        self.val_experiment_name = self.experiment_name

    def run(self,
            batch_size : int = 1) -> EasyDict:
        """Function to execute the validation pipeline

        Args:
            batch_size (int, optional): size of validation input batch. Defaults to 1.

        Returns:
            EasyDict: dictionary containing validation metrics result
        
        Example:
            ```python
            
            # Initialize validation pipeline
            validation_executor = PytorchValidationPipeline(config=config,
                                                            weights = weights_file,
                                                            backends = backends,
                                                            generate_report = True)
            ## OR
            validation_executor = IRValidationPipeline(config=config,
                                                       model = model_file,
                                                       backends = backends,
                                                       generate_report = True)
            
            # Run validation process
            results = validation_executor.run(batch_size = 2)

            ## OR (for IRValidationPipeline only, PytorchValidationPipeline can accept flexible batch size)
            ## 'batch_size' information is embedded in model.input_specs['input']['shape'][0]

            batch_size = validation_executor.model.input_specs['input']['shape'][0]
            results = validation_executor.run(batch_size = batch_size)
            ```
        """


        assert self.model, "'self.model' must be initialized in the sub-class!!"
        assert self.filename_suffix, "'self.filename_suffix' must be initialized in the sub-class!!"

        # Initial validation process
        eval_results = {}
        metric_assets = {}
        resource_filenames = {}
        resource_usages = {}

        for backend in self.backends :

            # Computing device assignment
            if isinstance(self.model,EasyDict):
                self.model.network = self.model.network.to(backend)
            
            # Validator initialization
            
            if isinstance(self.model,EasyDict) :
                val_experiment_name = self.val_experiment_name + '_{}'.format(backend)
            else:
                val_experiment_name = self.model.name.rsplit('.', 1)[0] + '_{}'.format(backend)

            if self.assets_dir:
                self.validation_args.update(dict(output_directory=self.assets_dir,
                                                experiment_name=val_experiment_name,
                                                batch_size=batch_size,
                                                ))
            else:
                self.validation_args.update(dict(experiment_name=val_experiment_name,
                                                batch_size=batch_size,
                                                ))
            validator = engine.create_validator(self.model, self.dataset, self.validation_args, device=backend)

            # Validation process
            eval_result = validator()
            eval_results.update({backend : eval_result})

                # Disable several validation features for hyperparameter optimization
            if not self.hypopt:
                metric_asset = validator.save_metrics(output_directory=self.assets_dir)

                resource_filename = validator.plot_resource_metrics(
                    output_directory=str(self.assets_dir)
                )
                resource_usage = validator.resource_usage()
                metric_assets.update({backend : metric_asset})
                resource_filenames.update({backend : resource_filename})
                resource_usages.update({backend : resource_usage})
        validation_args = validator.validation_args()
    
        if self.generate_report :
            generate_reports(
                eval_results=eval_results,
                output_directory=(self.reports_dir),
                experiment_name=self.experiment_name,
                dataset_info = self.dataset_info,
                metric_assets=metric_assets,
                resources=resource_filenames,
                resource_usage=resource_usages,
                validation_args=validation_args,
                # filename_suffix='_validation_{}'.format('_'.join(self.backends))
                filename_suffix=self.filename_suffix
            )
        ## NOTE : workaround for return values so other module still works as expected
        ## TODO : unify
        if len(eval_results.keys())==1 :
            eval_results = eval_results[backend]
        return EasyDict(eval_results)

class PytorchValidationPipeline(BaseValidationPipeline):
    """Vortex Validation Pipeline API for Vortex model
    """

    def __init__(self,config : EasyDict,
                 weights : Union[str,Path,None] = None,
                 backends : Union[list,str]=[],
                 generate_report : bool = True,
                 hypopt : bool =False):
        """Class initialization

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            weights (Union[str,Path,None], optional): path to selected Vortex model's weight. If set to None, it will \
                                                      assume that final model weights exist in **experiment directory**. \
                                                      Defaults to None.
            backends (Union[list,str], optional): device(s) to be used for validation process. If not provided, \
                                                  it will use the device described in **experiment file**. Defaults to [].
            generate_report (bool, optional): if enabled will generate validation report in markdown format. Defaults to True.
            hypopt (bool, optional): flag for hypopt, disable several pipeline process. Defaults to False.
        
        Example:
            ```python
            from vortex.utils.parser import load_config
            from vortex.core.pipelines import PytorchValidationPipeline
            
            # Parse config
            config_path = 'experiments/config/example.yml'
            weights_file = 'experiments/outputs/example/example.pth'
            backends = ['cpu','cuda']
            config = load_config(config_path)
            validation_executor = PytorchValidationPipeline(config=config,
                                                            weights = weights_file,
                                                            backends = backends,
                                                            generate_report = True)
            ```
        """
        super().__init__(config = config, backends = backends, generate_report = generate_report, hypopt = hypopt)
        
        # Model initialization

        if weights is None:
            filename = self.experiment_directory / ('%s.pth' % self.experiment_name)
        else:
            filename = weights
        warnings.warn('loading state dict from : %s' % str(filename))
        self.model = create_model(config.model,state_dict=filename,stage='validate')
        self.filename_suffix = '_validation_{}'.format('_'.join(self.backends))

class IRValidationPipeline(BaseValidationPipeline):
    """Vortex Validation Pipeline API for Vortex IR model
    """

    def __init__(self,config : EasyDict,
                 model : Union[str,Path,None],
                 backends : Union[list,str]=['cpu'],
                 generate_report : bool = True,
                 hypopt : bool =False):
        """Class initialization

        Args:
            config (EasyDict): ictionary parsed from Vortex experiment file
            model (Union[str,Path,None]): path to Vortex IR model, file with extension '.onnx' or '.pt'
            backends (Union[list,str], optional): runtime(s) to be used for validation process. Defaults to ['cpu'].
            generate_report (bool, optional): if enabled will generate validation report in markdown format. Defaults to True.
            hypopt (bool, optional): flag for hypopt, disable several pipeline process. Defaults to False.

        Raises:
            RuntimeError: raise error if the provided model file's extension is not '*.onnx' or '*.pt'
        
        Example:
            ```python
            from vortex.utils.parser import load_config
            from vortex.core.pipelines import IRValidationPipeline
            
            # Parse config
            config_path = 'experiments/config/example.yml'
            model_file = 'experiments/outputs/example/example.pt'
            backends = ['cpu','cuda']
            config = load_config(config_path)
            validation_executor = IRValidationPipeline(config=config,
                                                       model = model_file,
                                                       backends = backends,
                                                       generate_report = True)
            ```
        """

        super().__init__(config = config, backends = backends, generate_report = generate_report, hypopt = hypopt)
        
         # Model IR runtime check and selection
        runtime = backends
        model = Path(model)
        model_type = model.name.rsplit('.', 1)[1]

        avail_runtime = model_runtime_map[model_type]
        if model_type not in model_runtime_map:
            raise RuntimeError("Unknown model type extension, use .onnx or .pt file")
        if isinstance(runtime, str):
            runtime = [runtime]
        for backend in runtime:
            if backend not in avail_runtime:
                warnings.warn("Unable to run {} model on '{}', make sure to specify '--runtime' "\
                    "argument properly".format(str(model), backend))
                runtime.remove(backend)

        self.backends = runtime
        if not self.backends:
            self.backends = ['cpu'] # Fallback if configured device is empty and device in experiment file is unavailable
            warnings.warn('IR validation is running on CPU due to unavailability of selected device')
        self.model = model

        if model_type == 'pt':
            model_type = 'torchscript'
        self.filename_suffix = '_{}_IR_validation_{}'.format(model_type,'_'.join(self.backends))