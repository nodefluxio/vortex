Integrating Custom Classification Model to Vortex
=================================================

This tutorial shows how to develop your own model and integrate it to `vortex`.
The tutorial consists of 5 steps:

1. Define model architecture
2. Create post-process module
3. Define loss function
4. Register model's *builder function*
5. Integration with `vortex` CLI

```python
import torch
import torch.nn as nn
import torchvision as vision

import vortex
import vortex.networks as networks
import vortex.networks.modules as vortex_modules
```

---

1. Model Architecture
---------------------

We first define our model, in this case we will define AlexNet model
that can be integrated to `vortex`. In order to do that, our model needs
only to have `task` and `output_format` member variables.

For the sake of simplicity, we will reuse `AlexNet` from `torchvision` by
instantiate `torchvision`'s `AlexNet` and add the required member variable
`task` and `output_format`. `output_format` will be used to slice tensor
output from post-process module. 

`output_format` is a nested `dict`, with signature of `Dict[str,dict]`,
with inner `dict` with signature of `Dict[str,Union[list,int]]`.  The inner
dictionary is a mapping from output name to arguments that will be used for tensor slicing.
The tensor slicing operation will be performed using `np.take`, `Onnx.Gather`, or
`torch.index_select`; the arguments' naming use numpy's take, that is `indices` and `axis`;
check numpy, onnx, or torch docs for more details.

For classification models, we need `class_label` and `class_confidence`.
Note that this `output_format` will be used for single sample.

The code will looks like:
```
       alexnet = vision.models.alexnet(...)
       alexnet.task = "classification"
       alexnet.output_format = dict(
           class_label=dict(
               indices=[0], axis=0
           ),
           class_confidence=dict(
               indices=[1], axis=0
           )
       )
```

---

2. Create PostProcess module
----------------------------

Next, let's define our post-process module. This module will be responsible
for slicing tensor output from model's output tensor, as well as some additional postprocess
if necessary. For example, non-max suppression for detection model, or apply
softmax for clasification model if necessary.

Note that this module is used for evaluation only, skipped when training.
Additional arguments may be supplied as attributes via initializer.

```python
class AlexNetPostProcess(nn.Module):
    def __init__(self):
        super(AlexNetPostProcess, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.softmax(input)
        conf_label, cls_label = input.max(dim=1, keepdim=False)
        return torch.stack((cls_label.float(), conf_label), dim=1)
```

---

3. Defining Loss Function
-------------------------

Since we do not include softmax to our model (performed in post-process instead), 
we will use cross entropy loss.

Note that the function signature for loss function is `forward(input,target) -> Tensor`.
Additional arguments may be supplied as attributes via initializer.

```python
class ClassificationLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ClassificationLoss,self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(*args,**kwargs)
    
    def forward(self, input, target):
        target = target.squeeze()
        if target.size() == torch.Size([]):
            target = target.unsqueeze(0)
        return self.loss_fn(input, target)
```

---

4. Registering our model to vortex
----------------------------------

After defining necessary components, we need to define *builder* function
that will create instances of necessary components, then we register it to vortex
by decorating the function and providing the model name.

This function serves as entry point and customization point for our loss, model,
and post-process, and optionally preprocess and dataset collater. For example, you
need to reuse parameter from networks to postprocess, you can do it here.

The required components are `preprocess`, `network`, and `postprocess`.
For training, we need additional components, `loss` and optional `collate_fn`.
For preprocess, it is recommended to use modules from `vortex_modules.preprocess` to 
make sure it is exportable.

note that the function signature is 
```
create_model_components(model_name, preprocess_args, network_args, loss_args, postprocess_args, stage) -> dict
```
where model_name is a `str` holds the model name (`'alexnet'` in this case), `preprocess_args`,
`network_args`, `loss_args`, `postprocess_args` are mapping (`dict`) containing parameters from
configuration file, while `stage` is a string containing experiment stage (either `'train'` or `'validate'`)
given by vortex driver. Note that we simply *unpack argument mapping*  using `**` operator.

There are two ways of registering *builder* function, using decorator `register_model`
```
@vortex.networks.models.register_model(model_name='my_model_name')
def create_model_components(...):
```
or by directly calling `register_model_`
```
vortex.networks.models.register_model_('my_model_name',create_model_components)
```

```python
@networks.models.register_model(model_name='alexnet')
def create_model_components(
    model_name, preprocess_args,
    network_args, loss_args,
    postprocess_args, stage) -> dict:

    ## let's get model from torchvision
    ## then add 'task' and 'output_format' to our AlexNet class,
    ## adding attributes is not necessary 
    ## if your model already have 'task' and 'output_format'
    alexnet = vision.models.alexnet(**network_args)
    ## necessary attributes
    alexnet.task = "classification"
    alexnet.output_format = dict(
        class_label=dict(
            indices=[0], axis=0
        ),
        class_confidence=dict(
            indices=[1], axis=0
        )
    )

    postprocess = AlexNetPostProcess(**postprocess_args)
    loss_fn = ClassificationLoss(**loss_args)

    # using vortex' preprocess module
    preprocess = vortex_modules.preprocess.get_preprocess(
        'normalizer', **preprocess_args
    )

    ## wrap-up core components
    components = dict(
        preprocess=preprocess,
        network=alexnet,
        postprocess=postprocess,
    )
    if stage == 'train':
        ## if we are training then append loss function
        components.update(dict(
            loss=loss_fn
        ))
    elif stage == 'validate':
        ## do nothing for now
        pass
    
    return components
```

---

5. Integration with vortex CLI
----------------------------

Finally, we make this module as entry-point for our next experiments
to get more vortex' features in a *configurable* way:  

- hyperparameter optimization  
- training  
- validation  
- TorchScript & ONNX export (with batch inference support)   
- TensorRT inference (using ONNX model)  

```python
if __name__=='__main__':
    ## let's use vortex_cli to demonstrate vortex features
    ## this will be our entrypoint to supported experiments
    vortex.vortex_cli.main()
```

Note that since our custom model is added outside the vortex distribution,
it is unavailable from `vortex` command-line, to properly run experiments
with our custom model registered we need to invoke python, for example
```Shell
# run hyperparam optimization experiment
python3 alexnet.py hypopt --config cifar10.yml --optconfig hyperparam.yml
```
```Shell
# run training experiment
python3 alexnet.py train --config cifar10.yml
```
