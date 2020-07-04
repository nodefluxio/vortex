from easydict import EasyDict
from typing import Union
from pathlib import Path

from vortex.utils.common import check_and_create_output_dir
from vortex.core.factory import create_model,create_dataset,create_exporter
from vortex.predictor import create_predictor
from vortex.core.pipelines.base_pipeline import BasePipeline

__all__ = ['GraphExportPipeline']

class GraphExportPipeline(BasePipeline):
    """Vortex Graph Export Pipeline API
    """

    def __init__(self,
                 config: EasyDict, 
                 weights : Union[str,Path,None] = None):
        """Class initialization

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            weights (Union[str,Path,None], optional): path to selected Vortex model's weight. If set to None, it will \
                                                      assume that final model weights exist in **experiment directory**. \
                                                      Defaults to None.
        
        Example:
            ```python
            from vortex.utils.parser import load_config
            from vortex.core.pipelines import GraphExportPipeline
            
            # Parse config
            config = load_config('experiments/config/example.yml')
            graph_exporter = GraphExportPipeline(config=config,
                                                 weights='experiments/outputs/example/example.pth')
            ```
        """
        
         # Configure output directory
        self.experiment_directory, _ = check_and_create_output_dir(config)
        self.experiment_name = config.experiment_name

        # Initialize Pytorch model
        if weights is None:
            state_dict = self.experiment_directory / '{}.pth'.format(self.experiment_name)
        else:
            state_dict = weights
        model_components, ckpt = create_model(config.model, state_dict=state_dict, 
                                              stage='validate', return_checkpoint=True)
        model_components.network = model_components.network.eval()
        self.predictor = create_predictor(model_components).eval()
        self.image_size = config.model.preprocess_args.input_size

        cls_names = None
        if 'class_names' in ckpt:
            cls_names = ckpt['class_names']
        else:
            # Initialize dataset to get class_names
            warnings.warn("'class_names' is not available in your model checkpoint, please update "
                "them using 'scripts/update_model.py' script. \nCreating dataset to get 'class_names'")
            dataset = create_dataset(config.dataset, stage='train', 
                preprocess_config=config.model.preprocess_args)
            if hasattr(dataset.dataset, 'class_names'):
                cls_names = dataset.dataset.class_names
            else:
                warnings.warn("'class_names' is not available in dataset, setting "
                    "'class_names' to None.")
        self.class_names = cls_names

        # Initialize export config
        self.export_configs = [config.exporter] \
            if not isinstance(config.exporter, list) \
                else config.exporter

    def run(self,
            example_input : Union[str,Path,None] = None) -> EasyDict :
        """Function to execute the graph export pipeline

        Args:
            example_input (Union[str,Path,None], optional): path to example input image to help graph tracing. Defaults to None.

        Returns:
            EasyDict: dictionary containing status of the export process
        
        Example:
            ```python
            example_input = 'image1.jpg'
            graph_exporter = GraphExportPipeline(config=config,
                                                 weights='experiments/outputs/example/example.pth')

            result = graph_exporter.run(example_input=example_input)
            ```
        """
        outputs = []
        ok = True
        for export_config in self.export_configs :
            exporter = create_exporter(
                config=export_config,
                experiment_name=self.experiment_name,
                image_size=self.image_size,
                output_directory=(self.experiment_directory),
            )
            ok = exporter(
                predictor=self.predictor,
                class_names=self.class_names,
                example_image_path=example_input
            ) and ok
            outputs.append(str(exporter.filename))
        print('model is exported to:', ' and '.join(outputs))
        # TODO specify which export is failed
        result = EasyDict({'export_status' : ok})
        return result