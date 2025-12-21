#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/openai_server-vllm-neuron

# Check if USE_NEURON_VLLM environment variable is set, default to false
USE_NEURON_VLLM=${USE_NEURON_VLLM:-false}
docker buildx build  --build-arg USE_NEURON_VLLM="${USE_NEURON_VLLM}" \
    -t openai_server-vllm:neuron .
