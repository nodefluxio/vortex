Vortex Metadata
===============

Here we describe metadata format used by vortex.

1. Output Format
----------------
Output format **defines the slices** for each output for **single batch**.
Our interface relies on numpy's take_ operation to extract the output
from possibly-batched prediction tensor. Numpy take is utilized to
take elements from an array along an axis. Arguments to numpy take
that we use for extracting the output can be summarized in the
following table:

+-----------+---------------------------------------+
| Argument  | Description                           |
+===========+=======================================+
| `indices` | The indices of values to extract      |
+-----------+---------------------------------------+
| `axis`    | The axis over which values to extract |
+-----------+---------------------------------------+

We also assume that the output tensor is batched, hence if the model
is defined for single batch, one may need to unsqueeze or expand the dimension
at the first axis.

The number of output that can be defined from the output tensor is not limited.
The output format is expected to be nested dictionary, with outer dictionary
defines what the output is and the inner dictionary defines the arguments to
numpy take.

For example, assume we have classification model with the prediction tensor consists
of class labels and class confidences, as follow:

.. code-block:: python

    array([[3 , 0.71518937],
        [1, 0.54488318],
        [2, 0.64589411],
        [2, 0.891773  ]])


where the array is 2-dimensional array, with shape of :code:`(4,2)`, with first axis is the batch index,
and the second axis represents class labels and class confidences for the corresponding batch index.
For instance, the prediction of the first batch is :code:`[3 , 0.71518937]`, where the predicted class
is :code:`3` and the confidence is :code:`0.71518937`.

Let's :code:`'class_label'` and :code:`'class_confidence'` represent the predicted class labels and class_confidence,
the output format should be:

.. code-block:: python

    output_format = dict(
        class_label=dict(
            indices=[0],
            axis=0,
        ),
        class_confidence=dict(
            indices=[1],
            axis=0,
        )
    )

Note that the output format is defined to extract output of single batch.


.. _take: https://numpy.org/doc/stable/reference/generated/numpy.take.html

2. Class Names
--------------
Class Names is simply a mapping from integer to string representing a class name.
This can be described using dictionary, for example:

.. code-block:: python

    class_names = dict(
        0='cat',
        1='dog',
    )

3. Embedding Model Metadata
---------------------------
To embed the described metadata above, one may use utility function
:py:mod:`~vortex.runtime.onnx.graph_ops.embed_model_property.embed_model_property`
which takes onnx model and model property defined as nested dictionary, for example:

.. code-block:: python

    output_format = dict(
        class_label=dict(
            indices=[0],
            axis=0,
        ),
        class_confidence=dict(
            indices=[1],
            axis=0,
        )
    )
    class_names = dict(
        0='cat',
        1='dog',
    )
    model_props = dict(
        class_names=class_names,
        output_format=output_format,
    )

    model : onnx.ModelProto = embed_model_property(model,model_props)


Note that the model above is already an onnx model, and `embed_model_property` also
expects onnx model. This may be useful when one wants to use model defined on other framework.
since most deep learning also support exporting to onnx.