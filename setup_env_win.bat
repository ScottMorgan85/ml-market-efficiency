#!/bin/bash

# Script name: setup_env_win.bat

# Create a new conda environment
conda create --name ml_gpu_env python=3.8 -y

# Activate the environment
conda activate ml_gpu_env

# Install pip in the new environment (to use it for packages not available in conda)
conda install pip -y

# Install the requirements
pip install -r requirements.txt

# Install xbbg and blpapi (Bloomberg specific packages)
pip install xbbg
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/

# Install compatible CUDA and CuDNN versions
conda install cudatoolkit=11.2 cudnn=8.1 -y

# Verify xgboost is using GPU
python -c "import xgboost; print(xgboost.__version__); print(xgboost.runtime_capabilities())"

# Verify TensorFlow's GPU usage
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
