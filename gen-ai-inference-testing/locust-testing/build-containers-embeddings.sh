#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

if [ -x "$(command -v nvidia-smi)" ]
then
    script=$DIR/scripts/build-tritonserver-cuda-vllm.sh 
elif [ -x "$(command -v neuron-top)" ]
then
    script=$DIR/scripts/build-tritonserver-neuronx.sh
fi
echo "Building $(basename $script): This may take several minutes...."
bash $script 1>/tmp/build.log 2>&1