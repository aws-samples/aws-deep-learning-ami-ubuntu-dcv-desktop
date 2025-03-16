#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/openai-server-cuda-vllm
docker buildx build -t openai-server-cuda-vllm:latest .