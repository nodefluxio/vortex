# Getting Started

The main idea of Vortex is to use a configuration file which have all the necessary components, 
which can be used interchangeably, for specific development phase or we call it *pipeline*, e.g. 
training, validation, export. 

The interchangeability is achieved by using standardized modular components. 
You can have a configuration file minimaly for only a single pipeline, but the best practice is 
to have all configuration components for all pipelines so that you can use a single configuration 
file for them.

Before moving forward in this getting started guide, please make sure to have a vortex installation
in your machine by following [this installation guide](index.md#installation).
This installation will result in a command line `vortex` in your environment.

To get started, let's use vortex to train a classifier for [CIFAR10][1] dataset using 
[example configuration here][2]. Once you have downloaded the [configuration file][2] you could then
use the vortex cli tool to train it.

```
$ vortex train --config resnet18_classification_cifar10.yml
```

Those configuration file, will train a `resnet18` classification model for cifar10 dataset with
20 epoch and using gpu (`cuda:0`) on your machine. Vortex will automatically download the CIFAR10
dataset if you don't have one, as it is a built-in dataset in vortex.

After the training is done, the normal workflow usually is to validate the model to know how good 
the model is. In vortex, this could be done using the `validate` command:

```
$ vortex validate --config resnet18_classification_cifar10.yml
```

But this process is optional as validation is also automatically done in training pipeline as well.

At this stage, you would then ready to use our model in deployment. 
To deploy our model seamlessly, you need to export your model into what we called an 
*intermediate representation* (IR) which is a representation of the model that is more general for other
runtime, that is what you use to run the model, and also that we can use a better or faster inference
that is more optimize for the machine. 
In vortex we support two kinds of intermediate representation, torchscript and ONNX. 
Torchscript model is the builtin representation from PyTorch, so to run this representation we still 
need PyTorch installed to be able to run the model.
ONNX model is an open standard representation that is generally used and could run in a variety of
runtime libraries.

To export our model, use `export` command:

```
$ vortex export --config resnet18_classification_cifar10.yml
```

This would then resulted in exported ONNX model in the path pointed by `output_directory` in config.
You can specify what IR you want your models to export to by specifying in the `exporter` field in
the config file.

To understand more about each pipeline in the your model development, see [Pipeline User Guide][3]


[1]: https://www.cs.toronto.edu/~kriz/cifar.html
[2]: https://github.com/nodefluxio/vortex/blob/master/experiments/configs/resnet18_classification_cifar10.yml
[3]: user-guides/pipelines.md
