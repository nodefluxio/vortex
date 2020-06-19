# Start from Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

WORKDIR /opt/

## install system requirements
COPY requirements.sys requirements.sys
RUN apt update && xargs apt install -y < requirements.sys && \
    apt install -y python3 python3-pip

## install python requirements
COPY requirements.pip.cuda.rt.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY modules modules
COPY tests tests

CMD ["python3", "-m", "unittest", "tests/test_darknet53.py"]