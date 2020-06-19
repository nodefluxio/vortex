# vortex.core.pipelines

---

---

## Classes

---

---

### GraphExportPipeline



Vortex Graph Export Pipeline API
    

#### `__init__`



```python
def __init__(
      self,
      config : easydict.EasyDict,
      weights : typing.Union[str, pathlib.Path, NoneType] = None,
)
```



**Arguments**:

- `config` _EasyDict_ - dictionary parsed from Vortex experiment file
- `weights` _Union[str,Path,None], optional_ - path to selected Vortex model's weight. If set to None, it will                                                       assume that final model weights exist in **experiment directory**.                                                       Defaults to None.


**Examples**:



```python
from vortex.utils.parser import load_config
from vortex.core.pipelines import GraphExportPipeline

# Parse config
config = load_config('experiments/config/example.yml')
graph_exporter=GraphExportPipeline(config=config,weights='experiments/outputs/example/example.pth')
```



---

#### `run`



```python
def run(
      self,
      example_input : typing.Union[str, pathlib.Path, NoneType] = None,
)
```



**Arguments**:

- `example_input` _Union[str,Path,None], optional_ - path to example input image to help graph tracing. Defaults to None.


**Returns**:

- `EasyDict` - dictionary containing status of the export process


**Examples**:



```python
example_input = 'image1.jpg'
graph_exporter = GraphExportPipeline(config=config,
                                     weights='experiments/outputs/example/example.pth')

result = graph_exporter.run(example_input=example_input)
```



---

---

### HypOptPipeline



Vortex Hyperparameters Optimization Pipeline API
    

#### `__init__`



```python
def __init__(
      self,
      config : easydict.EasyDict,
      optconfig : easydict.EasyDict,
      weights : typing.Union[str, pathlib.Path, NoneType] = None,
)
```



**Arguments**:

- `config` _EasyDict_ - dictionary parsed from Vortex experiment file
- `optconfig` _EasyDict_ - dictionary parsed from Vortex hypopt configuration file
- `weights` _Union[str,Path,None], optional_ - path to selected Vortex model's weight. If set to None, it will                                                       assume that final model weights exist in **experiment directory**.                                                       Only used for ValidationObjective. Defaults to None.


**Examples**:



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

graph_exporter=HypOptPipeline(config=config,optconfig=optconfig)
```



---

#### `run`



```python
def run(
      self,
)
```



**Returns**:

- `EasyDict` - dictionary containing result of the hypopt process


**Examples**:



```python
graph_exporter=HypOptPipeline(config=config,optconfig=optconfig)
results=graph_exporter.run()
```



---

---

### PytorchPredictionPipeline



Vortex Prediction Pipeline API for Vortex model
    

#### `__init__`



```python
def __init__(
      self,
      config : easydict.EasyDict,
      weights : typing.Union[str, pathlib.Path, NoneType] = None,
      device : typing.Union[str, NoneType] = None,
)
```



**Arguments**:

- `config` _EasyDict_ - dictionary parsed from Vortex experiment file
- `weights` _Union[str,Path,None], optional_ - path to selected Vortex model's weight. If set to None, it will                                                       assume that final model weights exist in **experiment directory**.                                                       Defaults to None.
- `device` _Union[str,None], optional_ - selected device for model's computation. If None, it will use the device                                                 described in **experiment file**. Defaults to None.


**Raises**:

- `FileNotFoundError` - raise error if selected 'weights' file is not found


**Examples**:



```python
from vortex.core.pipelines import PytorchPredictionPipeline
from vortex.utils.parser import load_config

# Parse config
config_path = 'experiments/config/example.yml'
config = load_config(config_path)
weights_file = 'experiments/outputs/example/example.pth'
device = 'cuda'

vortex_predictor=PytorchPredictionPipeline(config = config,
                                           weights = weights_file,
                                           device = device)
```



---

#### `run`



```python
def run(
      self,
      images : typing.Union[typing.List[str], numpy.ndarray],
      visualize : bool = False,
      dump_visual : bool = False,
      output_dir : typing.Union[str, pathlib.Path] = '.',
      **kwargs,
)
```



**Arguments**:

- `images` _Union[List[str],np.ndarray]_ - list of images path or array of image
- `visualize` _bool, optional_ - option to return prediction visualization. Defaults to False.
- `dump_visual` _bool, optional_ - option to dump prediction visualization. Defaults to False.
- `output_dir` _Union[str,Path], optional_ - directory path to dump visualization. Defaults to '.' .
- `kwargs` _optional_ - this kwargs is placement for additional input parameters specific to                                 models'task


**Returns**:

- `EasyDict` - dictionary of prediction result


**Raises**:

- `TypeError` - raise error if provided 'images' is not list of image path or array of images


**Examples**:



```python

# Initialize prediction pipeline
vortex_predictor=PytorchPredictionPipeline(config = config,
                                           weights = weights_file,
                                           device = device)

### TODO: adding 'input_specs'

## OR
vortex_predictor=IRPredictionPipeline(model = model_file,
                                      runtime = runtime)

### You can get model's required parameter by extracting
### model's 'input_specs' attributes

input_shape  = vortex_predictor.model.input_specs.shape
additional_run_params = [key for key in vortex_predictor.model.input_specs.keys() if key!='input']
print(additional_run_params)

#### Assume that the model is detection model
#### ['score_threshold', 'iou_threshold'] << this parameter must be provided in run() arguments

# Prepare batched input
batch_input = ['image1.jpg','image2.jpg']

## OR
import cv2
batch_input = np.array([cv2.imread('image1.jpg'),cv2.imread('image2.jpg')])

results = vortex_predictor.run(images=batch_input,
                               score_threshold=0.9,
                               iou_threshold=0.2)
```



---

---

### IRPredictionPipeline



Vortex Prediction Pipeline API for Vortex IR model
    

#### `__init__`



```python
def __init__(
      self,
      model : typing.Union[str, pathlib.Path],
      runtime : str = 'cpu',
)
```



**Arguments**:

- `model` _Union[str,Path]_ - path to Vortex IR model, file with extension '.onnx' or '.pt'
- `runtime` _str, optional_ - backend runtime to be selected for model's computation. Defaults to 'cpu'.


**Examples**:



```python
from vortex.core.pipelines import IRPredictionPipeline
from vortex.utils.parser import load_config

# Parse config
model_file = 'experiments/outputs/example/example.pt' # Model file with extension '.onnx' or '.pt'
runtime = 'cpu'

vortex_predictor=IRPredictionPipeline(model = model_file,
                                      runtime = runtime)
```



---

#### `run`



```python
def run(
      self,
      images : typing.Union[typing.List[str], numpy.ndarray],
      visualize : bool = False,
      dump_visual : bool = False,
      output_dir : typing.Union[str, pathlib.Path] = '.',
      **kwargs,
)
```



**Arguments**:

- `images` _Union[List[str],np.ndarray]_ - list of images path or array of image
- `visualize` _bool, optional_ - option to return prediction visualization. Defaults to False.
- `dump_visual` _bool, optional_ - option to dump prediction visualization. Defaults to False.
- `output_dir` _Union[str,Path], optional_ - directory path to dump visualization. Defaults to '.' .
- `kwargs` _optional_ - this kwargs is placement for additional input parameters specific to                                 models'task


**Returns**:

- `EasyDict` - dictionary of prediction result


**Raises**:

- `TypeError` - raise error if provided 'images' is not list of image path or array of images


**Examples**:



```python

# Initialize prediction pipeline
vortex_predictor=PytorchPredictionPipeline(config = config,
                                           weights = weights_file,
                                           device = device)

### TODO: adding 'input_specs'

## OR
vortex_predictor=IRPredictionPipeline(model = model_file,
                                      runtime = runtime)

### You can get model's required parameter by extracting
### model's 'input_specs' attributes

input_shape  = vortex_predictor.model.input_specs.shape
additional_run_params = [key for key in vortex_predictor.model.input_specs.keys() if key!='input']
print(additional_run_params)

#### Assume that the model is detection model
#### ['score_threshold', 'iou_threshold'] << this parameter must be provided in run() arguments

# Prepare batched input
batch_input = ['image1.jpg','image2.jpg']

## OR
import cv2
batch_input = np.array([cv2.imread('image1.jpg'),cv2.imread('image2.jpg')])

results = vortex_predictor.run(images=batch_input,
                               score_threshold=0.9,
                               iou_threshold=0.2)
```



---

#### `runtime_predict`



```python
def runtime_predict(
      predictor,
      image : numpy.ndarray,
      **kwargs,
)
```



**Arguments**:

- `predictor `- Vortex runtime object
- `image` _np.ndarray_ - array of batched input image(s) with dimension of 4 (n,h,w,c)
- `kwargs` _optional_ - this kwargs is placement for additional input parameters specific to                                 models'task


**Returns**:

- `List` - list of prediction results


**Examples**:



```python
from vortex.core.factory import create_runtime_model
from vortex.core.pipelines import IRPredictionPipeline
import cv2

model_file = 'experiments/outputs/example/example_bs2.pt' # Model file with extension '.onnx' or '.pt'
runtime = 'cpu'
model = create_runtime_model(model_file, runtime)

batch_imgs = np.array([cv2.imread('image1.jpg'),cv2.imread('image2.jpg')])

results = IRPredictionPipeline.runtime_predict(model, 
                                               batch_imgs,
                                               score_threshold=0.9,
                                               iou_threshold=0.2
                                               )
```



---

---

### TrainingPipeline



Vortex Training Pipeline API
    

#### `__init__`



```python
def __init__(
      self,
      config : easydict.EasyDict,
      config_path : typing.Union[str, pathlib.Path, NoneType] = None,
      hypopt : bool = False,
)
```



**Arguments**:

- `config` _EasyDict_ - dictionary parsed from Vortex experiment file
- `config_path` _Union[str,Path,None], optional_ - path to experiment file. Need to be provided for                                                           backup **experiment file**. Defaults to None.
- `hypopt` _bool, optional_ - flag for hypopt, disable several pipeline process. Defaults to False.


**Raises**:

- `Exception` - raise undocumented error if exist


**Examples**:



```python
from vortex.utils.parser import load_config
from vortex.core.pipelines import TrainingPipeline

# Parse config
config_path = 'experiments/config/example.yml'
config = load_config(config_path)
train_executor = TrainingPipeline(config=config,config_path=config_path,hypopt=False)
```



---

#### `run`



```python
def run(
      self,
      save_model : bool = True,
)
```



**Arguments**:

- `save_model` _bool, optional_ - dump model's checkpoint. Defaults to True.


**Returns**:

- `EasyDict` - dictionary containing loss, val results and learning rates history


**Examples**:



```python
train_executor = TrainingPipeline(config=config,config_path=config_path,hypopt=False)
outputs = train_executor.run()
```



---

---

### PytorchValidationPipeline



Vortex Validation Pipeline API for Vortex model
    

#### `__init__`



```python
def __init__(
      self,
      config : easydict.EasyDict,
      weights : typing.Union[str, pathlib.Path, NoneType] = None,
      backends : typing.Union[list, str] = [],
      generate_report : bool = True,
      hypopt : bool = False,
)
```



**Arguments**:

- `config` _EasyDict_ - dictionary parsed from Vortex experiment file
- `weights` _Union[str,Path,None], optional_ - path to selected Vortex model's weight. If set to None, it will                                                       assume that final model weights exist in **experiment directory**.                                                       Defaults to None.
- `backends` _Union[list,str], optional_ - device(s) to be used for validation process. If not provided,                                                   it will use the device described in **experiment file**. Defaults to [].
- `generate_report` _bool, optional_ - if enabled will generate validation report in markdown format. Defaults to True.
- `hypopt` _bool, optional_ - flag for hypopt, disable several pipeline process. Defaults to False.


**Examples**:



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



---

#### `run`



```python
def run(
      self,
      batch_size : int = 1,
)
```



**Arguments**:

- `batch_size` _int, optional_ - size of validation input batch. Defaults to 1.


**Returns**:

- `EasyDict` - dictionary containing validation metrics result


**Examples**:



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

## OR (Currently for IRValidationPipeline only)
## 'batch_size' information is also embedded in model.input_specs['input']['shape'][0]

batch_size = validation_executor.model.input_specs['input']['shape'][0]
results = validation_executor.run(batch_size = batch_size)
```



---

---

### IRValidationPipeline



Vortex Validation Pipeline API for Vortex IR model
    

#### `__init__`



```python
def __init__(
      self,
      config : easydict.EasyDict,
      model : typing.Union[str, pathlib.Path, NoneType],
      backends : typing.Union[list, str] = ['cpu'],
      generate_report : bool = True,
      hypopt : bool = False,
)
```



**Arguments**:

- `config` _EasyDict_ - ictionary parsed from Vortex experiment file
- `model` _Union[str,Path,None]_ - path to Vortex IR model, file with extension '.onnx' or '.pt'
- `backends` _Union[list,str], optional_ - runtime(s) to be used for validation process. Defaults to ['cpu'].
- `generate_report` _bool, optional_ - if enabled will generate validation report in markdown format. Defaults to True.
- `hypopt` _bool, optional_ - flag for hypopt, disable several pipeline process. Defaults to False.


**Raises**:

- `RuntimeError` - raise error if the provided model file's extension is not '*.onnx' or '*.pt'


**Examples**:



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



---

#### `run`



```python
def run(
      self,
      batch_size : int = 1,
)
```



**Arguments**:

- `batch_size` _int, optional_ - size of validation input batch. Defaults to 1.


**Returns**:

- `EasyDict` - dictionary containing validation metrics result


**Examples**:



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

## OR (Currently for IRValidationPipeline only)
## 'batch_size' information is also embedded in model.input_specs['input']['shape'][0]

batch_size = validation_executor.model.input_specs['input']['shape'][0]
results = validation_executor.run(batch_size = batch_size)
```



---





