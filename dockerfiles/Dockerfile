ARG PYTHON_VERSION=3.6
ARG RUNTIME_TYPE=all

FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04 AS basebuild
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget libglib2.0-0 \
        libsm6 libxext6 libxrender-dev && \
    ## install python
    apt-get install -y python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev\
        python${PYTHON_VERSION}-distutils && \
    ## set default python
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    ## clean up
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*
RUN \
    ## install pip
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    ## disable cache
    pip config set global.cache-dir false && \
    pip install -U setuptools
RUN \
    ## checks
    pip --version && \
    pip list && \
    python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version"

FROM basebuild AS runtime
WORKDIR /app/vortex
ARG RUNTIME_TYPE
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
RUN pip3 install ./src/development[optuna_vis]
RUN python -c "import vortex.development"
WORKDIR /app
