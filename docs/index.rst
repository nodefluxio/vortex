Welcome to Vortex's documentation!
==================================

Vortex aims to unify various task for deep learning based computer vision,
such as classification and object detection, into single interface.
Usually, classification and object detection has different handling mechanism
when dealing with predicted tensor. For instance, classification task often
predicts (possibly batched) class label and class confidence in single tensor
while detection task requires different handling mechanism to deal with variable
detected object hence can't use single tensor for batched output. Furthermore,
the arrangement of values may not be the same for one model to another, for example,
one may organize the predicted class label at the first index of prediction tensor
while other may prefer class confidence first.

To deal the such problems, we simply annotate the model with neccessary information
about the prediction tensor. Specifically, we unify the way we take the prediction
tensor using generic operation using numpy.

The project consists of two major parts: ``development`` and ``runtime``. The ``development``
part define our :py:mod:`~vortex.development.networks.models.ModelBase` that enforce
additional information regarding prediction tensor to be defined. Also, an ``onnx`` exporter
for such model is provided at :py:mod:`~vortex.development.exporter.onnx.ONNXExporter`.
The ``runtime`` part provides class to perform inference on model with such metadata.
Our :py:mod:`~vortex.development.networks.models.ModelBase` is derived from PyTorchLightning_,
so we can easily define scalable model with maximum flexibility, to learn more about PyTorchLightning
including how to train the model, please refer to https://pytorchlightning.ai/.

.. _PyTorchLightning: https://github.com/PyTorchLightning/pytorch-lightning


Contents:
---------

.. toctree::
   :maxdepth: 2

   installing
   quickstart
   examples/index
   api/index
   metadata



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
