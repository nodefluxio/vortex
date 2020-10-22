FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git

RUN git clone https://github.com/nodefluxio/vortex

## install system requirements
RUN apt update && xargs apt install -y < vortex/src/development/requirements.sys

## install python requirements
RUN pip3 install -U pip setuptools

RUN pip3 install 'vortex/src/development[optuna_vis]' 'vortex/src/runtime[all]'

WORKDIR /app/src

CMD cp vortex/src .

WORKDIR /app
