#!/bin/bash

NV_GPU=0 nvidia-docker run -it --name deeplab -p 8888:8888 -p 6006:6006 \
    -v /home/$USER/:/home/$USER blutjens/deeplab:v0
