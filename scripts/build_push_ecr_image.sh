#! /bin/bash

pip install -q sagemaker
pip install -q sagemaker-studio-image-build

ORIG_DIR=$PWD
ROOT_DIR_PATH=$1

# change directory to where Dockerfile is place
cd $ROOT_DIR_PATH

# replace tag for future build versions
sm-docker build . --file ./docker/Dockerfile --repository sm-mm-mqm-byoc:1.0 

# change back to original path
cd $ORIG_DIR