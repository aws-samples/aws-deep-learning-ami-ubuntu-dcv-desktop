#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/openai_server-vllm-neuron

docker buildx build -t openai_server-vllm:neuron .
