#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/apiserver-neuronx-vllm
docker buildx build -t apiserver-neuronx-vllm:latest .