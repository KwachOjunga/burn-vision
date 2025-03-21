#!/bin/bash

# should probably create a way to check for the
# python env ;anyway

# model=$1  // var to pick model name
set -e
set -x
export PROJECT_DIR=".."
export MODEL_DIR="$PROJECT_DIR/models"
# mkdir $MODEL_DIR

python3 model.py

mv ~/.cache/torch/hub/checkpoints/* $MODEL_DIR
#for i in ~/.cache/torch/hub/checkpoints/; do 
#    onnx2burn ~/.cache/torch/hub/checkpoints/$i $MODEL_DIR
#done


