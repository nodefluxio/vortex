# adapted from onnxruntime

# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with TensorRT integration

# nVidia TensorRT Base Image
FROM nvcr.io/nvidia/tensorrt:20.07.1-py3

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=v1.5.3

RUN apt-get update &&\
    apt-get install -y sudo git bash unattended-upgrades

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:/opt/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/miniconda/lib:$LD_LIBRARY_PATH

# Prepare onnxruntime repository & build onnxruntime with TensorRT
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
    cp onnxruntime/docs/Privacy.md /code/Privacy.md &&\
    cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
    cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt &&\
    cd onnxruntime &&\
    /bin/sh ./build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) &&\
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime cmake-3.14.3-Linux-x86_64

# opencv deps
RUN apt install -y libsm6 libxext6 libxrender-dev ffmpeg x264 libx264-dev libsm6

WORKDIR /app/

# install vortex-runtime

COPY src/runtime src/runtime
COPY examples examples
# note dont provide extras since onnxruntime tensorrt need to be installed from source
# need to manually install onnx instead of using extras
RUN cd src/runtime && pip install onnx==1.6.0 && pip install -e .