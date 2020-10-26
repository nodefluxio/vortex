from pathlib import Path
from easydict import EasyDict
from typing import Union,List, Type
from collections import OrderedDict

import os
import warnings
import numpy as np
import cv2
import torch

from vortex.development.core.factory import create_model,create_dataset , create_runtime_model
from vortex.runtime import model_runtime_map
from vortex.development.utils.visual import visualize_result
from vortex.development.predictor import create_predictor, get_prediction_results
from vortex.development.utils.common import check_and_create_output_dir
from vortex.development.core.pipelines.base_pipeline import BasePipeline

__all__ = ['PytorchPredictionPipeline','IRPredictionPipeline']

class BasePredictionPipeline(BasePipeline):
    """Vortex Base Prediction Pipeline

    Args:
        BasePipeline : Base class for Vortex Pipeline
    """

    def __init__(self):
        """Class initialization
        """
        pass

    def _run_inference(self,
                       batch_imgs : List[np.ndarray],
                       **kwargs):
        """Default function to handle batch of inputs

        Args:
            batch_imgs (List[np.ndarray]):list of numpy array representation of batched input image(s)

        Raises:
            NotImplementedError: This method must be implemented in subclass
        """
        raise NotImplementedError("This method must be implemented in the subclass")

    def run(self,
            images : Union[List[str],np.ndarray],
            output_coordinate_format: str = 'relative',
            visualize : bool = False,
            dump_visual : bool = False,
            output_dir : Union[str,Path] = '.',
            **kwargs) -> EasyDict:
        """Function to execute the prediction pipeline

        Args:
            images (Union[List[str],np.ndarray]): list of images path or array of image
            output_coordinate_format (str, optional) - output coordinate format, especially usefull for models that returns
                coordinates in the input, e.g. bounding box, landmark, etc. Available: 
                `'relative'`: the coordinate is relative to input size (have range of [0, 1]), so to visualize the output needs to be multplied by input size; 
                `'absolute'`: the coordinate is absolute to input size (range of [widht, height]). 
                Default `'relative'`.
            visualize (bool, optional): option to return prediction visualization. Defaults to False.
            dump_visual (bool, optional): option to dump prediction visualization. Defaults to False.
            output_dir (Union[str,Path], optional): directory path to dump visualization. Defaults to '.' .
            kwargs (optional): forwarded to model's forward pass, so this kwargs is placement for additional input parameters, 
                make sure to have this if your model needs an additional inputs, e.g. `score_threshold`, etc.

        Raises:
            TypeError: raise error if provided 'images' is not list of image path or array of images

        Returns:
            EasyDict: dictionary of prediction result

        Example:
            ```python

            # Initialize prediction pipeline
            vortex_predictor=PytorchPredictionPipeline(config = config,
                                                       weights = weights_file,
                                                       device = device)

            ## OR
            vortex_predictor=IRPredictionPipeline(model = model_file,
                                                  runtime = runtime)

            # You can get model's required parameter by extracting model's 'input_specs' attributes

            input_shape  = vortex_predictor.model.input_specs['input']['shape']

            ## `input_specs['input']['shape']` will provide (batch_size,height,width,channel) dimension
            ## NOTES : PytorchPredictionPipeline can accept flexible batch size,
            ## however the `input_specs['input']['shape']` of the batch_size dimension 
            ## will always set to 1, ignore this

            # Extract additional run() input parameters specific for each model

            additional_run_params = [key for key in vortex_predictor.model.input_specs.keys() if key!='input']
            print(additional_run_params)

            ## Assume that the model is detection model
            ## ['score_threshold', 'iou_threshold'] << this parameter must be provided in run() arguments

            # Prepare batched input from image files path
            batch_input = ['image1.jpg','image2.jpg']

            ## OR
            import cv2
            input_size = input_shape[1] # Assume square input
            image1 = cv2.resize(cv2.imread('image1.jpg'), (input_size,input_size))
            image2 = cv2.resize(cv2.imread('image2.jpg'), (input_size,input_size))
            batch_input = np.array([image1,image2])

            results = vortex_predictor.run(images=batch_input,
                                           score_threshold=0.9,
                                           iou_threshold=0.2)

            # Additional process : obtain class_names from model
            class_names = vortex_predictor.model.class_names
            print(class_names)

            ```
        """
        ## make class_names optional
        ## assert self.class_names , "'self.class_names' must be implemented in the sub class"
        assert self.output_file_prefix , "'self.output_file_prefix' must be implemented in the sub class"

        assert isinstance(images,list) or isinstance(images,np.ndarray), "'images' arguments must be "\
            "provided with list or numpy ndarray, found {}".format(type(images))

        if isinstance(images[0],np.ndarray) and dump_visual:
            warnings.warn("Provided 'images' arguments type is np ndarray and 'dump_visual' is set to "
                "True, will not dump any image file due to lack of filename information")

        # Check image availability if image path is provided
        if isinstance(images[0],str) or isinstance(images[0],Path):
            image_paths = [Path(image) for image in images]
            for image_path in image_paths :
                assert image_path.exists(), "image {} doesn't exist".format(str(image_path))
            batch_mat = [cv2.imread(image) for image in images]
        elif isinstance(images[0],np.ndarray):
            batch_mat = images
            for image in batch_mat:
                assert len(image.shape) == 3, "Provided 'images' list member in numpy ndarray must be "\
                    "of dim 3, [h , w , c]"
        else:
            raise TypeError("'images' arguments must be provided with list of image path or list of "
                "numpy ndarray, found {}".format(type(images[0])))

        assert output_coordinate_format in ['relative', 'absolute'], "available 'output_coordinate_format': ['relative', 'absolute']"

        # Resize input images
        batch_vis = [mat.copy() for mat in batch_mat]
        batch_imgs = batch_mat
        results = self._run_inference(batch_imgs,**kwargs)

        # Transform coordinate-based result from relative coordinates to absolute value
        if output_coordinate_format == 'relative':
            results = self._check_and_transform(batch_vis = batch_vis, batch_results = results)

        # Visualize prediction
        if visualize:
            result_vis = self._visualize(batch_vis=batch_vis,
                                         batch_results=results)
            # Dump prediction
            if dump_visual and isinstance(images,list):
                filenames = []
                filename_fmt = "{filename}.{suffix}"
                for i, (vis, image_path) in enumerate(zip(batch_vis,images)) :
                    filename = Path(image_path)
                    suffix = filename.suffix.replace('.','')

                    filename = Path(output_dir) / filename_fmt.format_map(
                        dict(filename='_'.join([self.output_file_prefix,filename.stem]),suffix=suffix)
                    )
                    if not filename.parent.exists():
                        filename.parent.mkdir(parents=True)
                    cv2.imwrite(str(filename), vis)
                    filenames.append(str(filename))
                print('prediction saved to {}'.format(str(', '.join(filenames))))

            return EasyDict({'prediction' : results , 'visualization' : result_vis})
        else:
            return EasyDict({'prediction' : results , 'visualization' : None})


    def _visualize(self,
                   batch_vis : List,
                   batch_results : List) -> List:
        """Function to visualize prediction result

        Args:
            batch_vis (List): list of image(s) to be visualized
            batch_results (List): list of prediction result(s) correspond to batch_vis

        Returns:
            List: list of visualized image(s)
        """

        result_vis = []
        for vis, results in zip(batch_vis,  batch_results) :
            result_vis.append(visualize_result(
                vis=vis, results=[results],
                class_names=self.model.class_names
            ))
        return result_vis

    def _check_and_transform(self,
                             batch_vis : List,
                             batch_results : List) -> List:
        """Function to transform relative coords to absolute coords

        Args:
            batch_vis (List): list of image(s) to be visualized
            batch_results (List): list of prediction result(s) correspond to batch_vis

        Returns:
            List: list of transformed prediction result(s) correspond to batch_vis
        """

        for vis, result in zip(batch_vis, batch_results) :
            im_h, im_w, im_c = vis.shape
            if 'bounding_box' in result:
                bounding_box = result['bounding_box']
                if bounding_box is None:
                    continue
                bounding_box[...,0::2] *= im_w
                bounding_box[...,1::2] *= im_h
                result['bounding_box'] = bounding_box
            if 'landmarks' in result:
                landmarks = result['landmarks']
                if landmarks is None:
                    continue
                landmarks[...,0::2] *= im_w
                landmarks[...,1::2] *= im_h
                result['landmarks'] = landmarks
        return batch_results

class PytorchPredictionPipeline(BasePredictionPipeline):
    """Vortex Prediction Pipeline API for Vortex model
    """

    def __init__(self,
                 config : EasyDict,
                 weights : Union[str,Path,None] = None,
                 device : Union[str,None] = None,
                 ):
        """Class initialization

        Args:
            config (EasyDict): dictionary parsed from Vortex experiment file
            weights (Union[str,Path,None], optional): path to selected Vortex model's weight. If set to None, it will \
                                                      assume that final model weights exist in **experiment directory**. \
                                                      Defaults to None.
            device (Union[str,None], optional): selected device for model's computation. If None, it will use the device \
                                                described in **experiment file**. Defaults to None.

        Raises:
            FileNotFoundError: raise error if selected 'weights' file is not found

        Example:
            ```python
            from vortex.development.core.pipelines import PytorchPredictionPipeline
            from vortex.development.utils.parser import load_config

            # Parse config
            config_path = 'experiments/config/example.yml'
            config = load_config(config_path)
            weights_file = 'experiments/outputs/example/example.pth'
            device = 'cuda'

            vortex_predictor = PytorchPredictionPipeline(config = config,
                                                       weights = weights_file,
                                                       device = device)
            ```
        """

        self.config = config
        self.output_file_prefix = 'prediction'

        # Configure experiment directory
        experiment_directory, _ = check_and_create_output_dir(config)

        # Set compute device
        if device is None:
            if 'device' in config:
                device = config.device
            elif 'device' in config.trainer:
                device = config.trainer.device
            else:
                raise RuntimeError("'device' argument is not configured and not found in 'config.device'. "
                    "Please specify either one.")
        device = torch.device(device)

        # Initialize model
        if weights is None:
            if hasattr(config, 'checkpoint') and config.checkpoint is not None:
                weights = config.checkpoint
            else:
                weights = Path(experiment_directory) / ('{}.pth'.format(config.experiment_name))
                if not os.path.isfile(weights):
                    raise RuntimeError("Default weight in {} is not exist, please provide weight "
                        "path using '--weights' argument.".format(str(weights)))
        ckpt = torch.load(weights)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        model_components = create_model(config.model, state_dict=state_dict, stage='validate')
        model_components.network = model_components.network.to(device)
        self.model = create_predictor(model_components)
        self.model.to(device)

        ## input_specs -> {input_name: {shape, pos, type}}
        input_specs = OrderedDict()
        img_size = config.model.preprocess_args.input_size
        additional_inputs = tuple()
        if hasattr(model_components.postprocess, 'additional_inputs') :
            additional_inputs = model_components.postprocess.additional_inputs
            assert isinstance(additional_inputs, tuple) and len(additional_inputs) > 0
            assert all(isinstance(additional_input, tuple) for additional_input in additional_inputs)

        if isinstance(img_size, int):
            input_specs['input'] = {'shape': (1, img_size, img_size, 3), 'pos': 0, 'type': 'uint8'}
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            input_specs['input'] = {'shape': (1, img_size[0], img_size[1], 3), 'pos': 0, 'type': 'uint8'}
        else:
            raise RuntimeError("Unknown config of model.preprocess_args.input_size of type {} with value {}"
                .format(type(img_size), img_size))
        for n, (name, shape) in enumerate(additional_inputs):
            input_specs[name] = {
                'shape': tuple(shape) if shape is not None else shape,
                'pos': n+1,
                'type': 'float'
            }
        self.model.input_specs = input_specs

        cls_names = None
        if 'class_names' in ckpt:
            cls_names = ckpt['class_names']
        else:
            dataset_name = None
            if 'dataset' in config and 'name' in config.dataset.train:
                dataset_name = config.dataset.train.name
            elif 'dataset' in config and 'dataset' in config.dataset.train:
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
                    config.dataset if 'dataset' in config else ''))
        self.model.class_names = cls_names

    def _run_inference(self,
                       batch_imgs : List[np.ndarray],
                       **kwargs) -> List:
        """Function to run model's inference

        Args:
            batch_imgs (List[np.ndarray]): list of numpy array representation of batched input image(s)

        Returns:
            List: list of batched prediction result(s)
        """

        # TODO enable padding to keep aspect ratio
        # Resize input ( assuming square input, and stretch, no padding provided yet )
        input_size = self.model.input_specs['input']['shape']
        batch_imgs = [cv2.resize(img, input_size[1:3]) for img in batch_imgs]
        batch_imgs = np.stack(batch_imgs)

        # Do model inference
        device = list(self.model.parameters())[0].device
        inputs = {'input' : torch.from_numpy(batch_imgs).to(device)}

        ## Get model additional input specific for each task
        ## TODO: use input_specs
        if hasattr(self.model.postprocess, 'additional_inputs') :
            additional_inputs = self.model.postprocess.additional_inputs
            assert isinstance(additional_inputs, tuple)
            for additional_input in additional_inputs :
                key, in_size = additional_input
                if key in kwargs:
                    value = kwargs[key]
                    if in_size is not None:
                        value = [value] if in_size[-1] == 1 and not isinstance(value, (tuple, list)) else value
                    inputs[key] = torch.from_numpy(np.asarray(value)).to(device)

        with torch.no_grad() :
            results = self.model(**inputs)

        if isinstance(results, torch.Tensor) :
            results = results.cpu().numpy()
        if isinstance(results, (np.ndarray, (list, tuple))) \
            and not isinstance(results[0], (dict)):
            ## first map to cpu/numpy
            results = list(map(lambda x: x.cpu().numpy() if isinstance(x,torch.Tensor) else x, results))

        # Formatting inference result
        output_format = self.model.output_format
        results = get_prediction_results(
            results=results,
            output_format=output_format
        )
        return results

class IRPredictionPipeline(BasePredictionPipeline):
    """Vortex Prediction Pipeline API for Vortex IR model
    """
    def __init__(self,
                 model : Union[str,Path],
                 runtime : str = 'cpu'):
        """Class initialization

        Args:
            model (Union[str,Path]): path to Vortex IR model, file with extension '.onnx' or '.pt'
            runtime (str, optional): backend runtime to be selected for model's computation. Defaults to 'cpu'.
        
        Example:
            ```python
            from vortex.development.core.pipelines import IRPredictionPipeline
            from vortex.development.utils.parser import load_config

            # Parse config
            model_file = 'experiments/outputs/example/example.pt' # Model file with extension '.onnx' or '.pt'
            runtime = 'cpu'

            vortex_predictor=IRPredictionPipeline(model = model_file,
                                                  runtime = runtime)
            ```
        """
        # Create IR model specified by its runtime
        model_type = Path(model).name.rsplit('.', 1)[1]
        runtime_map = model_runtime_map[model_type]
        for name, rt in runtime_map.items() :
            print('Runtime {} <{}>: {}'.format(
                name, rt.__name__, 'available' \
                    if rt.is_available() else 'unavailable'
            ))
        self.model = create_runtime_model(model, runtime)
        if model_type == 'pt':
            model_type = 'torchscript'
        self.output_file_prefix = '{}_ir_prediction'.format(model_type)

    def _run_inference(self,
                       batch_imgs : List[np.ndarray],
                       **kwargs) -> List:
        """Function to run model's inference

        Args:
            batch_imgs (List[np.ndarray]): list of numpy array representation of batched input image(s)

        Returns:
            List: list of batched prediction result(s)
        """

        # Check input batch size to match with IR model input specs
        input_shape = self.model.input_specs['input']['shape']
        n = input_shape[0]
        assert len(batch_imgs) <= n, "expects 'images' <= n batch ({}) got {}".format(n, len(batch_imgs))

        # TODO add resize with pad in runtime
        # Resize input
        batch_imgs = type(self.model).resize_batch([mat.copy() for mat in batch_imgs], input_shape)
        results = self.model(batch_imgs, **kwargs)

        return results
