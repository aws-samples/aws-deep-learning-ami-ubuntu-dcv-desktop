#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/openai-server-neuronx-vllm
docker buildx build -t openai-server-neuronx-vllm:latest .