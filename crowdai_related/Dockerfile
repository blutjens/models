FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
LABEL maintainer "Bjorn Lutjens <lutjens@mit.edu>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip \
    python-setuptools

RUN apt-get install -y python-pil python-numpy \
    python-matplotlib wget

# To convert MS coco dataset in PASCAL (for Deeplab)
RUN pip install baker
RUN pip install -I path.py==7.7.1
RUN apt-get install -y python3.6-dev
RUN pip install cytoolz
