# vortex.core.factory

---

---



## Functions

---

---

### create_model



Function to create model and it's signature components. E.g. loss function, collate function, etc



```python
def create_model(
      model_config : easydict.EasyDict,
      state_dict : typing.Union[str, dict, pathlib.Path] = None,
      stage : str = 'train',
)
```



**Arguments**:

- `model_config` _EasyDict_ - Experiment file configuration at `model` section, as EasyDict object
- `state_dict` _Union[str, dict, Path], optional_ - [description]. `model` Pytorch state dictionary or commonly known as weight, can be provided as the path to the file, or the returned dictionary object from `torch.load`. If this param is provided, it will override checkpoint specified in the experiment file. Defaults to None.
- `stage` _str, optional_ - If set to 'train', this will enforce that the model must have `loss` and `collate_fn` attributes, hence it will make sure model can be used for training stage. If set to 'validate' it will ignore those requirements but cannot be used in training pipeline, but may still valid for other pipelines. Defaults to 'train'.


**Returns**:

- `EasyDict` - The dictionary containing the model's components


**Raises**:

- `TypeError` - Raises if the provided `stage` not in 'train' or 'validate'


**Examples**:



The dictionary returned will contain several keys :

- `network` : Pytorch model's object which inherit `torch.nn.Module` class.
- `preprocess` : model's preprocessing module
- `postprocess` : model's postprocessing module
- `loss` : if provided, module for model's loss function
- `collate_fn` : if provided, module to be embedded to dataloader's `collate_fn` function to modify dataset label's format into desirable format that can be accepted by `loss` components

```python
from vortex.core.factory import create_model
from easydict import EasyDict

model_config = EasyDict({
    'name': 'softmax',
    'network_args': {
        'backbone': 'efficientnet_b0',
        'n_classes': 10,
        'pretrained_backbone': True,
    },
    'preprocess_args': {
        'input_size': 32,
        'input_normalization': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
        'scaler': 255,
        }
    },
    'loss_args': {
        'reduction': 'mean'
    }
})

model_components = create_model(
    model_config = model_config
)
print(model_components.keys())
```



---

---

### create_runtime_model



Functions to create runtime model



```python
def create_runtime_model(
      model_path : typing.Union[str, pathlib.Path],
      runtime : str,
      output_name = ['output'],
      *args,
      **kwargs,
)
```



**Arguments**:

- `model_path` _Union[str, Path]_ - Path to Intermediate Representation (IR) model file
- `runtime` _str_ - Backend runtime to be used, e.g. : 'cpu' or 'cuda' (Depends on available runtime options)
- `output_name` _list, optional_ - Runtime output(s) variable name. Defaults to ["output"].


**Returns**:

- `Type[BaseRuntime]` - Runtime model objects based on IR file model's type and selected `runtime`


**Raises**:

- `RuntimeError` - Raises if selected `runtime` is not available


**Examples**:



```python
from vortex.core.factory import create_runtime_model
import numpy as np
import cv2

model_path = 'tests/output_test/test_classification_pipelines/test_classification_pipelines.onnx'

runtime_model = create_runtime_model(
    model_path = model_path,
    runtime = 'cpu'
)

print(type(runtime_model))

## Get model's input specifications and additional inferencing parameters

print(runtime_model.input_specs)

# Inferencing example

input_shape = runtime_model.input_specs['input']['shape']
batch_imgs = np.array([cv2.resize(cv2.imread('tests/images/cat.jpg'),(input_shape[2],input_shape[1]))])

## Make sure the shape of input data is equal to input specifications
assert batch_imgs.shape == tuple(input_shape)

## Additional parameters can be inspected from input_specs,
## E.g. `score_threshold` or `iou_threshold` for object detection
additional_input_parameters = {}

## Inference is done by utilizing __call__ method
prediction_results = runtime_model(batch_imgs,
                                    **additional_input_parameters)

print(prediction_results)
```



---

---

### create_dataloader



Function to create iterable data loader object



```python
def create_dataloader(
      dataloader_config : easydict.EasyDict,
      dataset_config : easydict.EasyDict,
      preprocess_config : easydict.EasyDict,
      stage : str = 'train',
      collate_fn : typing.Union[typing.Callable, str, NoneType] = None,
)
```



**Arguments**:

- `dataloader_config` _EasyDict_ - Experiment file configuration at `dataloader` section, as EasyDict object
- `dataset_config` _EasyDict_ - Experiment file configuration at `dataset` section, as EasyDict object
- `preprocess_config` _EasyDict_ - Experiment file configuration at `model.preprocess_args` section, as EasyDict object
- `stage` _str, optional_ - Specify the experiment stage, either 'train' or 'validate'. Defaults to 'train'.
- `collate_fn` _Union[Callable,str,None], optional_ - Collate function to reformat batch data serving. Defaults to None.


**Returns**:

- `Type[Iterable]` - Iterable dataloader object which served batch of data in every iteration


**Raises**:

- `TypeError` - Raises if provided `collate_fn` type is neither 'str' (registered in Vortex), Callable (custom function), or None
- `RuntimeError` - Raises if specified 'dataloader' module is not registered


**Examples**:



```python
from vortex.core.factory import create_dataloader
from easydict import EasyDict

dataloader_config = EasyDict({
    'module': 'PytorchDataLoader',
    'args': {
    'num_workers': 1,
    'batch_size': 4,
    'shuffle': True,
    },
})

dataset_config = EasyDict({
    'train': {
        'dataset': 'ImageFolder',
        'args': {
            'root': 'tests/test_dataset/classification/train'
        },
        'augmentations': [{
            'module': 'albumentations',
            'args': {
                'transforms': [
                {
                    'transform' : 'RandomBrightnessContrast', 
                    'args' : {
                        'p' : 0.5, 'brightness_by_max': False,
                        'brightness_limit': 0.1, 'contrast_limit': 0.1,
                    }
                },
                {'transform': 'HueSaturationValue', 'args': {}},
                {'transform' : 'HorizontalFlip', 'args' : {'p' : 0.5}},
                ]
            }
        }]
    },
})

preprocess_config = EasyDict({
    'input_size' : 224,
    'input_normalization' : {
        'mean' : [0.5,0.5,0.5],
        'std' : [0.5, 0.5, 0.5],
        'scaler' : 255
    },
})

dataloader = create_dataloader(dataloader_config=dataloader_config,
                                dataset_config=dataset_config,
                                preprocess_config = preprocess_config,
                                collate_fn=None)
for data in dataloader:
    images,labels = data
```



---



