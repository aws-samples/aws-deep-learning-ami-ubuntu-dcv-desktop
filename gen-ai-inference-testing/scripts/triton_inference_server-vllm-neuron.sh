#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/triton_inference_server-vllm-neuron

VLLM_NEURON=${VLLM_NEURON:-"git+https://github.com/aws-neuron/upstreaming-to-vllm.git@2.26.1"}
docker buildx build  --build-arg VLLM_NEURON="${VLLM_NEURON}" \
    -t triton_inference_server-vllm:neuron .