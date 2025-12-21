#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/triton_inference_server-python-neuron
docker buildx build -t triton_inference_server-python:neuron .