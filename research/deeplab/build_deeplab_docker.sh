#!/bin/bash

docker build --build-arg user=$USER -t blutjens/deeplab:v0 . 
