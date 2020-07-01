FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git

WORKDIR /app

RUN git clone https://github.com/nodefluxio/vortex.git

WORKDIR /app/vortex
RUN git checkout v0.1.0

## install system requirements
RUN apt update && xargs apt install -y < requirements.sys

## install python requirements
RUN pip3 install -U pip && \
    pip3 install -r requirements.txt

RUN pip3 install .[optuna-vis]

WORKDIR /app
