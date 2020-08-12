FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git

WORKDIR /app/vortex

COPY ./packages /app/vortex/packages

## install system requirements
RUN apt update && xargs apt install -y < packages/development/requirements.sys

## install python requirements
RUN pip3 install -U pip setuptools

RUN pip3 install 'packages/runtime[all]' 'packages/development[optuna_vis]'

WORKDIR /app
