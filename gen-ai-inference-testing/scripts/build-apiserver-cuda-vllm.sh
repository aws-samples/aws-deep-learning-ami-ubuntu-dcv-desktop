#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/apiserver-cuda-vllm
docker buildx build -t apiserver-cuda-vllm:latest .