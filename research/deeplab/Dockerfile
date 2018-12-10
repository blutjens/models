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

RUN pip install --upgrade pip
# For GPU
RUN pip install --upgrade tensorflow-gpu==1.9.0

# To convert MS coco dataset in PASCAL (for Deeplab)
RUN pip install baker
RUN pip install -I path.py==7.7.1
RUN apt-get install -y python3-dev
RUN pip install cytoolz lxml

#RUN pip install jupyter

# From tensorflow/models/research/
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# DATASET_DIR="datasets"
# cd "${WORK_DIR}/${DATASET_DIR}"
# sh download_and_convert_voc2012.sh

# source local_test.sh