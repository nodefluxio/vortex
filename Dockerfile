ARG BASE_IMAGE=nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
ARG PYTHON_VERSION=3.6
ARG RUNTIME_TYPE=all

FROM ${BASE_IMAGE} AS basebuild
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    ## clean up
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

FROM basebuild AS conda
ARG PYTHON_VERSION
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -c pytorch -y cudatoolkit=10.2 && \
    /opt/conda/bin/conda clean -ya
RUN python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version"

FROM basebuild AS runtime
WORKDIR /app/vortex
ARG RUNTIME_TYPE
COPY --from=conda /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH
COPY ./src/runtime /app/vortex/src/runtime
RUN pip install ./src/runtime[$RUNTIME_TYPE]
RUN python -c "import vortex.runtime"
WORKDIR /app

FROM runtime AS development
WORKDIR /app/vortex
COPY ./src/development /app/vortex/src/development
RUN apt-get update && \
    xargs apt-get install --no-install-recommends -y < src/development/requirements.sys && \
    ## clean up
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*
RUN pip install ./src/development[optuna_vis] --ignore-installed
RUN python -c "import vortex.development"
WORKDIR /app
