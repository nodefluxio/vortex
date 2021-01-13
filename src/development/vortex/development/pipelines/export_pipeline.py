from easydict import EasyDict
from typing import Union
from pathlib import Path

import os
import torch
import warnings

from vortex.development.utils.common import check_and_create_output_dir
from vortex.development.core.factory import create_model,create_dataset,create_exporter
from vortex.development.predictor import create_predictor
from vortex.development.core.pipelines.base_pipeline import BasePipeline

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
            weights (Union[str,Path], optional): path to selected Vortex model's weight. If set to None, it will \
                                                 assume that final model weights exist in **experiment directory**. \
                                                 Defaults to None.

        Example:
            ```python
            from vortex.development.utils.parser import load_config
            from vortex.development.core.pipelines import GraphExportPipeline
            
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
            weights = self.experiment_directory / '{}.pth'.format(self.experiment_name)
            if not os.path.isfile(weights):
                raise RuntimeError("Default weight in {} is not exist, please provide weight "
                    "path using '--weights' argument.".format(str(weights)))
        ckpt = torch.load(weights)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        model_components = create_model(config.model, state_dict=state_dict, stage='validate')
        model_components.network = model_components.network.eval()
        self.predictor = create_predictor(model_components).eval()
        self.image_size = config.model.preprocess_args.input_size

        cls_names = None
        if 'class_names' in ckpt:
            cls_names = ckpt['class_names']
        else:
            dataset_name = None
            if 'name' in config.dataset.train:
                dataset_name = config.dataset.train.name
            elif 'dataset' in config.dataset.train:
                dataset_name = config.dataset.train.dataset

            if dataset_name:
                from vortex.development.utils.data.dataset.dataset import all_datasets
                dataset_available = False
                for datasets in all_datasets.values():
                    if dataset_name in datasets:
                        dataset_available = True
                        break

                if dataset_available:
                    # Initialize dataset to get class_names
                    warnings.warn("'class_names' is not available in your model checkpoint, please "
                        "update your model using 'scripts/update_model.py' script. \nCreating dataset "
                        "to get 'class_names'")
                    dataset = create_dataset(config.dataset, stage='train', 
                        preprocess_config=config.model.preprocess_args)
                    if hasattr(dataset.dataset, 'class_names'):
                        cls_names = dataset.dataset.class_names
                    else:
                        warnings.warn("'class_names' is not available in dataset, setting "
                            "'class_names' to None.")
            else:
                warnings.warn("Dataset {} is not available, setting 'class_names' to None.".format(
                    config.dataset))
        if cls_names is None:
            num_classes = 2     ## default is binary class
            if 'n_classes' in config.model.network_args:
                num_classes = config.model.network_args.n_classes
            self.class_names = ["class_{}".format(i) for i in range(num_classes)]
        self.class_names = cls_names

        # Initialize export config
        self.export_configs = [config.exporter] \
            if not isinstance(config.exporter, list) \
                else config.exporter

    def run(self,
            example_input : Union[str,Path,None] = None) -> EasyDict :
        """Function to execute the graph export pipeline

        Args:
            example_input (Union[str,Path], optional): path to example input image to help graph tracing. 
                Defaults to None.

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
        print('model is exported to:', ', '.join(outputs))
        # TODO specify which export is failed
        result = EasyDict({'export_status' : ok})
        return result
