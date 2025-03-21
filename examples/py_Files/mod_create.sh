#!/bin/bash

set -x
#set -e

JSON_DIR=../../src/json
TXT_DIR=../../src/txt

echo "Creating json and text dirs"
mkdir $JSON_DIR $TXT_DIR

python3 ./convert_to_pt.py

mv ../../models/onnx_dir/*.json $JSON_DIR
mv ../../models/onnx_dir/*.txt $TXT_DIR
mv ../../models/onnx_dir/*.rs ../../src/
