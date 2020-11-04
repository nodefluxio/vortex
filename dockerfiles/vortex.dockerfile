FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app/vortex

COPY ./src /app/vortex/src

## install system requirements
RUN apt-get update && xargs apt-get install -y < src/development/requirements.sys

## install python requirements
RUN pip3 install -U pip setuptools

RUN pip3 install 'src/runtime[all]' 'src/development[optuna_vis]'

WORKDIR /app
