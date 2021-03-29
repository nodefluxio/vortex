Installation
============


Vortex consists of 2 packages:

* Vortex development package (``vortex.development``), which contains all the 
  necessary components to develop a model.

* Vortex runtime package (``vortex.runtime``), which contains only minimal 
  dependencies to run a model for inference with the option for:
  * all
  * torchscript
  * onnxruntime
  * onnxruntime-tensorrt

Currently vortex is only work using ``python3.6``, but support for other 
versions are to come.

There are two ways to use vortex, using **docker** or **install 
from source**. But we generallly recommend to use the provided docker images.


Docker (recommended)
--------------------

We have provided the pre-built docker images in our official docker hub repo:

https://hub.docker.com/r/nodefluxio/vortex

For the vortex development the tag name is ``<version>``, for vortex runtime 
the tag is ``runtime-<runtime_type>-<version>``.

If you want to build the images yourself, run these command in the root 
directory of the vortex source:

- vortex development

  .. code-block:: bash

    docker build --target=development -t vortex:dev .

- vortex runtime

  .. code-block:: bash

    docker build --target=runtime --build-arg RUNTIME_TYPE=<runtime_type> -t vortex:runtime .

  you need to change the ``<runtime_type>`` option. The available options are: 
  ``all``, ``torchscript``, and ``onnxruntime``. So if you want to only build
  ``onnxruntime`` image:

  .. code-block:: bash

    docker build --target=runtime --build-arg RUNTIME_TYPE=onnxruntime -t vortex:runtime .

  if you want to build the **onnxruntime TensorRT**, you can follow the steps 
  in https://github.com/nodefluxio/vortex/blob/master/dockerfiles/README.md

From Source
-----------

Vortex currently only tested in **python 3.6**, so if you want to build it from 
source make sure to build it for python3.6. You also need to make sure that you 
have ``pip`` installed for the python3.6.

Before installing vortex, you need to first install the opencv dependencies:

  .. code-block:: bash

    sudo apt-get update
    sudo apt-get install libsm6 libxrender-dev

Then, clone the `vortex repository <https://github.com/nodefluxio/vortex>`_.

- vortex development

  It should be noted that ``vortex.development`` package also depends on 
  ``vortex.runtime`` package, so you need to install both packages:

  .. code-block:: bash

    pip install ./src/runtime[all] ./src/development

  Or if you want to install with additional optuna visualization support:

  .. code-block:: bash

    pip install 'src/development[optuna_vis]'

- vortex runtime

  To install, you can simply

  .. code-block:: bash

    pip install ./src/runtime[all]

  you can also change the option ``all`` to the desired runtime type. The available
  options are ``all``, ``torchscript``, and ``onnxruntime``.

It should be noted that for TensorRT runtime type, you need to build the onnxruntime 
from the source. If you want to build vortex from source with TensorRT runtime 
support, you can follow the steps in `dockerfile here 
<https://github.com/nodefluxio/vortex/blob/master/dockerfiles/onnxruntime-tensorrt.dockerfile>`_.
