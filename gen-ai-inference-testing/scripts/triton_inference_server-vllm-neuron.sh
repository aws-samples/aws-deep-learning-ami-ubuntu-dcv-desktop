#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/triton_inference_server-vllm-neuron

# Check if USE_NEURON_VLLM environment variable is set, default to false
USE_NEURON_VLLM=${USE_NEURON_VLLM:-false}
docker buildx build  --build-arg USE_NEURON_VLLM="${USE_NEURON_VLLM}" \
    -t triton_inference_server-vllm:neuron .