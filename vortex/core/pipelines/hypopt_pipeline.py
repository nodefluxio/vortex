import os
import sys
from pathlib import Path

from easydict import EasyDict
from typing import Union
import logging
import optuna
import operator
import numpy as np
import warnings

from vortex.core.pipelines.training_pipeline import TrainingPipeline
from vortex.core.pipelines.validation_pipeline import PytorchValidationPipeline
from vortex.utils.parser.override import override_param
from vortex.utils.common import check_and_create_output_dir
from vortex.core.pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

__all__=['HypOptPipeline']

class HypOptPipeline(BasePipeline):
    """Vortex Hyperparameters Optimization Pipeline API
    """

    def __init__(self,
                 config : EasyDict,
                 optconfig : EasyDict,
                 weights : Union[str,Path,None] = None):
        """Hyperparameters Optimization Pipeline API Initialization

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            optconfig (EasyDict): dictionary parsed from Vortex hypopt configuration file
            weights (Union[str,Path,None], optional): path to selected Vortex model's weight. If set to None, it will \
                                                      assume that final model weights exist in **experiment directory**. \
                                                      Only used for ValidationObjective. Defaults to None.
        
        Example:
            ```python
            from vortex.core.pipelines import HypOptPipeline
            from vortex.utils.parser.loader import Loader
            import yaml

            # Parse config
            config_path = 'experiments/config/example.yml'
            optconfig_path = 'experiments/hypopt/learning_rate_search.yml'

            with open(config_path) as f:
                config_data = yaml.load(f, Loader=Loader)
            with open(optconfig_path) as f:
                optconfig_data = yaml.load(f, Loader=Loader)

            graph_exporter = HypOptPipeline(config=config,
                                            optconfig=optconfig)
            ```
        """
        
        self.optconfig = optconfig
        self.config = config

        experiment_directory , _ = check_and_create_output_dir(config)
        self.optuna_report_output_dir=os.path.join(experiment_directory,'hypopt',optconfig.study_name)

        if not os.path.isdir(self.optuna_report_output_dir):
            os.makedirs(self.optuna_report_output_dir)

        objective_args = {}
        if hasattr(optconfig.objective, 'args') :
            objective_args = optconfig.objective.args
        self.objective = create_objective_fn(
            optconfig.objective.module,
            config=config, 
            param_opt_config=optconfig,
            weights=weights,
            **objective_args,
        )
        self.trial_name = '{}_{}'.format(
                config.experiment_name, optconfig.study_name
            )
        self.n_trials = None if not hasattr(optconfig.study, 'n_trials') else optconfig.study.n_trials

    def _create_optuna_study(self,
                            config : Union[EasyDict,str],
                            name : str) -> optuna.study.Study:
        """Function to create Optuna study

        Args:
            config (Union[EasyDict,str]): hypopt configuration from `study` field
            name (str): configured name for the study

        Returns:
            optuna.study.Study: Optuna Study object
        """

        default_fn = lambda *args, **kwargs : None
        pruner, pruner_args = default_fn, {}
        sampler, sampler_args = default_fn, {}
        if hasattr(config, 'pruner'):
            if hasattr(config.pruner, 'method') :
                pruner: str = config.pruner.method
                if pruner in optuna.pruners.__dict__.keys():
                    pruner = optuna.pruners.__getattribute__(pruner)
                else:
                    sys.exit(
                        'pruner %s not available, tried to retrieve from optuna.pruners. aborting.' % pruner)
            if hasattr(config.pruner, 'args') :
                pruner_args = config.pruner.args
        if hasattr(config, 'sampler'):
            if hasattr(config.sampler, 'method') :
                sampler: str = config.sampler.method
            if hasattr(config.sampler, 'args') :
                sampler_args: dict = config.sampler.args
            if isinstance(sampler, str):
                if sampler in optuna.samplers.__dict__.keys():
                    sampler = optuna.samplers.__getattribute__(sampler)
                else:
                    try:
                        sampler = optuna.integration.__getattr__(sampler)
                    except AttributeError:
                        sys.exit(
                            'sampler %s not available, tried to retrieve from optuna.samplers and optuna.integration. aborting.' % sampler)
        direction = config.direction
        study_name = name
        pruner = pruner(**pruner_args)
        sampler = sampler(**sampler_args)
        study_args = config.args if hasattr(config,'args') else {}
        study_args_keys=list(study_args.keys())
        for key in study_args_keys:
            if key not in ['storage','load_if_exists']:
                warnings.warn('Only ["storage","load_if_exists"] are supported in this field, parameter "%s" will be omitted'%key)
                study_args.pop(key)
        return optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            **study_args
        )

    def _visualize_study(self,
                         study : optuna.study.Study,
                         output_dir : str, 
                         study_name: str):
        """Function to visualize Optuna trials in a study

        Args:
            study (optuna.study.Study): Optuna Study object
            output_dir (str): directory to dump visualization result
            study_name (str): name of Optuna study
        """

        logger.warn("saving visualization")

        contour = optuna.visualization.plot_contour(study)
        intermediate_value = optuna.visualization.plot_intermediate_values(study)
        optimization_history = optuna.visualization.plot_optimization_history(
            study)
        parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
        slice_graph = optuna.visualization.plot_slice(study)

        contour.write_image(os.path.join(output_dir,'%s_hypopt_contour.png' % study_name))
        intermediate_value.write_image(os.path.join(output_dir,
            '%s_hypopt_intermediate_value.png' % study_name))
        optimization_history.write_image(os.path.join(output_dir,
            '%s_hypopt_optimization_history.png' % study_name))
        parallel_coordinate.write_image(os.path.join(output_dir,'%s_parallel_coordinate.png' % study_name))
        slice_graph.write_image(os.path.join(output_dir,'%s_slice.png' % study_name))

    def run(self) -> EasyDict:
        """Function to execute the hypopt pipeline

        Returns:
            EasyDict: dictionary containing result of the hypopt process

        Example:
            ```python
            graph_exporter = HypOptPipeline(config=config,
                                            optconfig=optconfig)
            results = graph_exporter.run()
            ```
        """
        study = self._create_optuna_study(self.optconfig.study,self.trial_name)
        study.optimize(self.objective, n_trials=self.n_trials)

        logger.info('Number of finished trials : %s' %len(study.trials))

        logger.info('Best trial:')
        trial = study.best_trial

        logger.info('\tvalue : %s' %trial.value)
        logger.info('\tparams : ')
        for key, value in trial.params.items():
            logger.info('\t\t{}: {}'.format(key, value))

        with open(os.path.join(self.optuna_report_output_dir,'best_params.txt'),'w') as f_out:
            f_out.write('Number of finished trials : %s\n' %len(study.trials))
            f_out.write('Best trial:\n')
            f_out.write('\tvalue : %s\n' %trial.value)
            f_out.write('\tparams : \n')
            for key, value in trial.params.items():
                f_out.write('\t\t{}: {}\n'.format(key, value))

        if optuna.visualization.is_available():
            try:
                self._visualize_study(study,self.optuna_report_output_dir, self.trial_name)
            except Exception as e:
                warnings.warn('Hypopt Visualization is not succesful, found error : {}'.format(e))
        best_trial_results = EasyDict({'best_trial' : {'metric_value' : trial.value , 'params' : trial.params}})
        return best_trial_results

## TODO : update logging, dump config, clean up, move to directory
class BaseObjective :
    """Base class for Optuna objective
    """
    def __init__(self, 
                 config : EasyDict, 
                 param_opt_config : EasyDict, 
                 reduction : str='latest', 
                 reduction_args : dict={}, 
                 direction : str ='minimize'):
        """Base initialization

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            param_opt_config (EasyDict): dictionary parsed from Vortex hypopt configuration file
            reduction (str, optional): the reduction function used to averaged the returned value. Defaults to 'latest'.
                                       Supported reduction :
                                            - 'latest' : select the last value ( index [-1] ),
                                            - 'mean' : see numpy.mean
                                            - 'sum' : see numpy.sum
                                            - 'min' : see numpy.min
                                            - 'max' : see numpy.max
                                            - 'median' : see numpy.median
                                            - 'average' : see numpy.average
                                            - 'percentile' : see numpy.percentile
                                            - 'quantile' : see numpy.quantile
            reduction_args (dict, optional): the corresponding arguments for selected `reduction`. Defaults to {}.
            direction (str, optional): either 'maximize' or 'minimize' the objective value. Defaults to 'minimize'.

        Raises:
            KeyError: raise error if overrided parameters not exist in experiment config
        """

        self.config = EasyDict(config)
        self.param_opt_config = EasyDict(param_opt_config)
        self.best_metric = None
        self.original_param = {}
        for key, value in self.param_opt_config.override.items() :
            try :
                source = 'self.config.%s' %key
                original_value = eval(source)
            except AttributeError:
                raise KeyError('parameter %s (which is to be overriden) does not exist in original config' %key)
            logger.info('recording original param : <%s,%s>' %(key,original_value))
            self.original_param[key] = original_value
        self.best_val_metrics = []

        self.metric_comparator = operator.lt if direction=='maximize' else operator.gt

        supported_reductions = ['mean', 'sum', 'min', 'max', 'median', 'average', 'percentile', 'quantile','latest']
        assert reduction in supported_reductions, 'current implementation only support %s for val_metrcis reduction' %', '.join(supported_reductions)
        if reduction not in ['latest']:
            self.reduction = np.__dict__[reduction]
        else:
            self.reduction = self._get_last_value
        self.reduction_args = reduction_args

    def _get_last_value(self,
                        metrics : Union[list,np.ndarray,float],
                        *args,
                        **kwargs) -> float:
        """Function to get the latest value from an array

        Args:
            metrics (Union[list,np.ndarray,float]): returned objective values from pipelines

        Returns:
            float: reduced metrics
        """
        if isinstance(metrics,list) or isinstance(metrics,np.ndarray):
            metric=metrics[-1]
        else:
            metric=metrics
        return metric

    def evaluate(self, 
                 config : EasyDict, 
                 *args, 
                 **kwargs) :
        """Function to run objective evaluation

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file

        Raises:
            NotImplementedError: must be implemented in sub class
        """
        raise NotImplementedError
    
    def __call__(self, 
                 trial : optuna.trial.Trial) -> float:
        """Pipeline to execute an objective evaluation

        Args:
            trial (optuna.trial.Trial): Optuna trial object

        Raises:
            RuntimeError: raise error if configured parameter have more than 1 key value pairs
            RuntimeError: raise error of parameter suggestion is not supported by Optuna

        Returns:
            float: trial metric result
        """
        for parameter in self.param_opt_config.parameters :
            if not len(parameter)==1 :
                raise RuntimeError("expects parameter %s only to have single key value pairs, got %s" %(parameter, len(parameter)))
            param_name = list(parameter.keys())[0]
            suggest_type = parameter[param_name].suggestion
            suggest_args = parameter[param_name].args
            if not suggest_type in optuna.trial.Trial.__dict__.keys() :
                raise RuntimeError("param suggestion %s not supported by optuna, available : %s" %(suggest_type, ','.join([key for key in optuna.trial.Trial.__dict__.keys() if 'suggest_' in key])))
            suggested_value = optuna.trial.Trial.__dict__[suggest_type](trial,param_name,**suggest_args)
            logger.info('config.%s = %s' %(param_name, suggested_value))
            exec('self.config.%s = suggested_value' %(param_name))
        
        self.config = override_param(self.config, self.param_opt_config.override)
        
        for key, value in self.param_opt_config.additional_config.items() :
            logger.info('adding param : %s' %key)
            logger.info('config.%s = %s' %(key, value))
            exec('self.config.%s = value' %(key))
        
        metric = self.evaluate(config=self.config)
        logger.info('metrics : {}'.format(metric))
        metric = self.reduction(metric, **self.reduction_args)
        logger.info('reduced metric : {}'.format(metric))
        logger.warn('objective metric : %s' %metric)

        # TODO eagerly save best params for allowing remote stopping via comet_ml
        if not self.best_metric:
            self.best_metric = metric
        elif self.metric_comparator(self.best_metric, metric) :
            self.best_metric = metric
        return metric

class TrainObjective(BaseObjective): 
    """Objective for training pipeline hypopt

    Args:
        BaseObjective : Base class for Optuna objective
    """

    def __init__(self, 
                 metric_type : str = 'loss', 
                 metric_name : Union[str,None] = None,
                 *args, 
                 **kwargs) :
        """Class initialization

        Args:
            metric_type (str, optional): type of metric to be optimized : 'val' or 'loss'. Defaults to 'loss'.
            metric_name (Union[str,None], optional): only used if metric_type is set to 'val'. 
                                                     This argument denotes the name of the metric which want 
                                                     to be optimized. The available setting for this argument 
                                                     also related to the model's task. Defaults to None.

                                                     - Detection task :

                                                        - 'mean_ap' : using the mean-Average Precision metrics

                                                    - Classification task :

                                                        - 'accuracy' : using the accuracy metrics
        """
        kwargs.pop('weights')
        super().__init__(*args, **kwargs)
        self.metric_type = metric_type
        assert self.metric_type in ['loss', 'val']
        if self.metric_type == 'val':
            assert metric_name is not None, "'metric_name' must be defined when 'metric_type' set to 'val'"
        self.metric_name = metric_name
    
    def evaluate(self, 
                 config : EasyDict) ->  np.ndarray:
        """Function to run objective evaluation

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file

        Returns:
            np.ndarray: array of metrics collected from training pipelines
        """
        vortex_trainer = TrainingPipeline(config=self.config,hypopt=True)
        output = vortex_trainer.run(save_model=False)
        epoch_losses = output.epoch_losses
        val_metrics = output.val_metrics
        learning_rates = output.learning_rates
        if self.metric_type == 'loss' :
            metric = epoch_losses
        elif self.metric_type == 'val' :
            assert self.metric_name in val_metrics[0], "'metric_name' = '%s' not found, available metrics = %s"%(self.metric_name,list(val_metrics[0].keys()))
            metric = [val_metric[self.metric_name] for val_metric in val_metrics]
        return np.array(metric)

class ValidationObjective(BaseObjective) :
    """Objective for validation pipeline hypopt

    Args:
        BaseObjective : Base class for Optuna objective
    """
    def __init__(self, 
                 metric_name : str,
                 *args, 
                 **kwargs) :
        """Objective funtion for validation pipelines hypopt

        Args:
            metric_name (str): denotes the name of the metric which want 
                               to be optimized. The available setting for this argument 
                               also related to the model's task. Defaults to None.

                               - Detection task :

                                   - 'mean_ap' : using the mean-Average Precision metrics

                               - Classification task :

                                   - 'accuracy' : using the accuracy metrics
        """
        self.weights = kwargs['weights']
        kwargs.pop('weights')
        super().__init__(*args, **kwargs)
        assert metric_name is not None, "'metric_name' must be defined when using ValidationObjective"
        self.metric_name = metric_name

    # TODO added IRValidationPipeline for IR validation hypopt
    def evaluate(self, 
                 config : EasyDict) -> float:
        """Function to run objective evaluation

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file

        Returns:
            float: metric collected from validation pipelines
        """

        vortex_validator = PytorchValidationPipeline(config=self.config,weights=self.weights,generate_report=False,hypopt=True)
        val_metrics = vortex_validator.run(batch_size=config.dataset.dataloader.args.batch_size) #batch size is inferred from experiment file
        assert self.metric_name in val_metrics, "'metric_name' = '%s' not found, available metrics = %s"%(self.metric_name,list(val_metrics.keys()))
        metric = val_metrics[self.metric_name]
        return metric

supported_objective_fn = {
    TrainObjective.__name__ : TrainObjective,
    ValidationObjective.__name__ : ValidationObjective,
}

def create_objective_fn(module : str, 
                        *args, 
                        **kwargs) -> object:
    """Function to create objective function

    Args:
        module (str): objective module's name

    Returns:
        object : objective function object
    """
    assert module in supported_objective_fn
    module_type = supported_objective_fn[module]
    return module_type(*args, **kwargs)