```python
%matplotlib inline
```

Implements Custom Backbone Module in Vortex
===========================================
This tutorial shows how to create custom backbone module and integrate to `vortex`.
The tutorial consists of 5 steps:

1. Vortex Backbone Module
2. Registering *builder* function
3. Pretrained weights and module restructure *hack*
4. Integration with `vortex` CLI
```python
import torch
import torch.nn as nn
import torchvision as vision

import warnings

import vortex
import vortex.networks as networks
import vortex.networks.modules as vortex_modules
```

1. Vortex Backbone Module
-------------------------

On computer vision tasks, it is often necessary to perform experiment
on various backbone networks. Hence, it is useful to have adapter
to multiple backbone, if desired.

`vortex` has such *adapter* capability with its `BackbonePoolConnector`
that provides interfaces to any registered backbone. The registered 
backbone itself should be type of `Backbone` class which actually
an adapter for `nn.Sequential` to also returns its *intermediate results*.


2. Registering *builder* function
---------------------------------

Registering backbone is done by adding *builder* function to `vortex`.
To register the function, we decorate our function with 
```
@vortex.networks.modules.backbones.register_backbone(model_name)
def get_backbone(...)
```
or we can call `register_backbone_` directly.

The function *should* be named `get_backbone`. When calling decorator,
we can register multiple model on single function. And the function signature:
```
get_backbone(model_name, pretrained, feature_type, **kwargs) -> Backbone
```
where `model_name` is `str` representing the requested model, 
`pretrained` may be `str` or path-like depending on the caller, 
and `feature_type` is `str` for customization point if necessary.

In this case we will add `SqueezeNet` as backbone to `vortex`. For the sake
of simplicity, we'll load existing network & pretrained from `torchvision`.

```python
register_backbone = vortex_modules.backbones.register_backbone

@register_backbone(['squeezenetv1.1', 'squeezenetv1.0'])
def get_backbone(model_name, pretrained=False, feature_type='tri_stage_fpn', **kwargs):
    """
    create squeezenet (v1.0 & v1.1) backbone
    """
```

3. Pretrained weights and module restructure *hack*
---------------------------------------------------

Often times, we do not want to develop from scratch but use existing
module instead. But, this raises an issue since not all existing module
are the same structure and dynamic nature of pytorch scripts make us
difficult to reuse module and pretrained weights.

To be able reuse pretrained weights, we could first load the original
structure of the network and then load the `state_dict` and then
restructure filter parts of the network to `nn.Sequential` 
(as required by `Backbone`). We do this *hack* on our *builder* function.

Since we know that available pretrained model is trained on ImageNet, and
we want to use pretrained weights, let's set number of classes to 1000
before loading the model.

```python
    num_classes = kwargs.get('num_classes',1000)
    if pretrained and num_classes != 1000:
        warnings.warn('temporarily change number of classes to get imagenet pretrained weights')
        kwargs.update({'num_classes': 1000})
    
    if model_name == 'squeezenetv1.1':
        model = vision.models.squeezenet1_1(pretrained=pretrained, **kwargs)
    elif model_name == 'squeezenetv1.0':
        model = vision.models.squeezenet1_0(pretrained=pretrained, **kwargs)
    else:
        raise RuntimeError("model f{model_name} not supported")
```

Upon instantiate model, the next step is to prepare feature extractor.
We can use `feature_type` as customization point if specific feature
extractor are required, for example, you want to return feature from
activation input instead of activation output, etc. Current 
implementation needs `'tri_stage_fpn'` # and `'classifier'` 
to be defined. For `'tri_stage_fpn'` we need to split
the `feature` to five stages. For `'classifier'`, we need to
provide both `feature` and `classifier`.

```python
    if feature_type == 'tri_stage_fpn':
        if model_name == 'squeezenetv1.1':
            features = [
                model.features[0],
                model.features[1:3],
                model.features[3:7],
                model.features[7:12],
                model.features[12:],
            ]
        elif model_name == 'squeezenetv1.0':
            features = [
                model.features[0],
                model.features[1:3],
                model.features[3:6],
                model.features[6:8],
                model.features[8:],
            ]
        features = vortex_modules.Backbone(nn.Sequential(*features))
```

For `'classifier'` feature, we may need to restore original number of
classes if we were loading from pretrained model.

```python
    elif feature_type == 'classifier':
        # looking at squeezenet implementation, we need to reset this layer
        if pretrained and num_classes != 1000:
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        classifier_feature = model.classifier
        features = model.features
        features = vortex_modules.ClassifierFeature(
            features,classifier=classifier_feature
        )
```

Raise error here if given `feature_type` is not supported.

```python
    else:
        raise RuntimeError("feature type f{feature_tyep} not supported by f{model_name}")

    return features

```

4. Integration with vortex CLI
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
python squeezenet.py train --config cifar10.yml
```
