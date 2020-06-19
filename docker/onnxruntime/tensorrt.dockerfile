# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with TensorRT integration

# nVidia TensorRT Base Image
FROM nvcr.io/nvidia/tensorrt:19.09-py3

# MAINTAINER Vinitra Swamy "viswamy@microsoft.com"

# ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
# ARG ONNXRUNTIME_BRANCH=v1.1.0

# RUN apt-get update &&\
#     apt-get install -y sudo git bash

WORKDIR /opt/

COPY requirements.apt.tensorrt.rt.txt requirements.txt
RUN apt update && xargs apt install -y < requirements.txt

COPY requirements.pip.tensorrt.rt.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /opt/onnx

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:/opt/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/miniconda/lib:$LD_LIBRARY_PATH

# Prepare onnxruntime repository & build onnxruntime with TensorRT
# RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
#     /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
#     cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
#     cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt &&\
#     cd onnxruntime &&\
#     /bin/sh ./build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) &&\
#     pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
#     cd .. &&\
#     rm -rf onnxruntime cmake-3.14.3-Linux-x86_64

RUN wget https://github.com/microsoft/onnxruntime/archive/v1.1.0.tar.gz && tar -zxvf v1.1.0.tar.gz && rm v1.1.0.tar.gz && \ 
    wget https://github.com/google/nsync/archive/1.20.1.tar.gz && tar -zxvf 1.20.1.tar.gz && rm 1.20.1.tar.gz && \
    wget https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz && tar -zxvf v3.6.1.tar.gz && rm v3.6.1.tar.gz && \
    wget https://github.com/HowardHinnant/date/archive/v2.4.1.tar.gz && tar -zxvf v2.4.1.tar.gz && rm v2.4.1.tar.gz && \
    wget https://github.com/google/re2/archive/2019-03-01.tar.gz && tar -zxvf 2019-03-01.tar.gz && rm 2019-03-01.tar.gz && \
    wget https://github.com/onnx/onnx-tensorrt/archive/release/6.0.tar.gz && tar -zxvf 6.0.tar.gz && rm 6.0.tar.gz && \
    wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz && tar -zxvf release-1.8.0.tar.gz && rm release-1.8.0.tar.gz && \
    wget https://github.com/onnx/onnx/archive/v1.6.0.tar.gz && tar -zxvf v1.6.0.tar.gz && rm v1.6.0.tar.gz && \
    git clone https://github.com/eigenteam/eigen-git-mirror.git && cd eigen-git-mirror && git checkout 899973d && cd .. && \
    wget https://github.com/nlohmann/json/archive/v3.7.1.tar.gz && tar -zxvf v3.7.1.tar.gz && rm v3.7.1.tar.gz && \
    wget https://github.com/NVlabs/cub/archive/v1.8.0.tar.gz && tar -zxvf v1.8.0.tar.gz && rm v1.8.0.tar.gz && \
    git clone https://github.com/google/gemmlowp.git && cd gemmlowp && git checkout 42c5318 && cd .. && \
    mv onnxruntime-1.1.0 onnxruntime && \
    mv nsync-1.20.1/* onnxruntime/cmake/external/nsync/ && rm -rf nsync-1.20.1 && \
    mv protobuf-3.6.1/* onnxruntime/cmake/external/protobuf/ && rm -rf protobuf-3.6.1/ && \
    mv date-2.4.1/* onnxruntime/cmake/external/date/ && rm -rf date-2.4.1/ && \
    mv re2-2019-03-01/* onnxruntime/cmake/external/re2/ && rm -rf re2-2019-03-01/ && \
    mv onnx-tensorrt-release-6.0/* onnxruntime/cmake/external/onnx-tensorrt/ && rm -rf onnx-tensorrt-release-6.0/ && \
    mv googletest-release-1.8.1/* onnxruntime/cmake/external/googletest/ && rm -rf googletest-release-1.8.1/ && \
    mv onnx-1.6.0/* onnxruntime/cmake/external/onnx/ && rm -rf onnx-1.6.0/ && \
    mv eigen-git-mirror/* onnxruntime/cmake/external/eigen/ && rm -rf eigen-git-mirror/ && \
    mv json-3.7.1/* onnxruntime/cmake/external/json/ && rm -rf json-3.7.1/ && \
    mv cub-1.8.0/* onnxruntime/cmake/external/cub/ && rm -rf cub-1.8.0/ && \
    mv gemmlowp/* onnxruntime/cmake/external/gemmlowp/ && rm -rf gemmlowp/ && \
    bash onnxruntime/dockerfiles/scripts/install_common_deps.sh && \
    cd onnxruntime && bash build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --skip_submodule_sync --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) && \
    pip install /opt/onnx/onnxruntime/build/Linux/Release/dist/*.whl
