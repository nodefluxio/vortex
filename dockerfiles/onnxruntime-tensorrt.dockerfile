# adapted from onnxruntime

# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with TensorRT integration

# nVidia TensorRT Base Image
FROM nvcr.io/nvidia/tensorrt:20.07.1-py3 AS runtime

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=v1.5.3
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:/opt/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/miniconda/lib:$LD_LIBRARY_PATH

RUN apt update && apt install -y --no-install-recommends \
        wget \
        zip \
        ca-certificates \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        python3-dev

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh && \
    /opt/miniconda/bin/conda clean -ya && \
    pip install numpy && \
    rm -rf /opt/miniconda/pkgs && \
    wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz && \
    tar zxf cmake-3.14.3-Linux-x86_64.tar.gz && \
    rm -rf cmake-3.14.3-Linux-x86_64.tar.gz

# Prepare onnxruntime repository & build onnxruntime with TensorRT
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
    cp onnxruntime/docs/Privacy.md /code/Privacy.md &&\
    cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
    cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt &&\
    cd onnxruntime &&\
    /bin/sh ./build.sh --parallel --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) &&\
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime cmake-3.14.3-Linux-x86_64

# opencv deps
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev ffmpeg x264 libx264-dev libsm6 pciutils

WORKDIR /app/

# install vortex-runtime
COPY src/runtime src/runtime
COPY examples examples
# note dont provide extras since onnxruntime tensorrt need to be installed from source
# need to manually install onnx instead of using extras
RUN cd src/runtime && pip install -U setuptools onnx==1.6.0 && pip install -e .

# TODO: use CMD instead of RUN for proper testing
RUN python -c "import vortex.runtime as vrt; assert(vrt.model_runtime_map['onnx']['tensorrt'].is_available())"

FROM runtime AS development
# install vortex-development
COPY src/development src/development
RUN pip install --upgrade pip
RUN cd src/development && pip install --ignore-installed --timeout=10000 -e .

# TODO: use CMD instead of RUN for proper testing
RUN python -c "import vortex.development"
