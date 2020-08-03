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
- `task` : model's described task. E.g. : 'classification' or 'detection'
- `loss` : if provided, module for model's loss function
- `collate_fn` : if provided, module to be embedded to dataloader's `collate_fn` function to modify dataset label's format into desirable format that can be accepted by `loss` components



---



