#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

export IMAGE=docker.io/library/tritonserver-cuda-trtllm:latest
export COMMAND=/scripts/triton-tensorrt-llm.sh
export HF_HOME=/snapshots/huggingface

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..


if [ "$1" == "up" ]
then
mkdir -p $HOME/scripts/triton
cp $scripts_dir/build-engine.sh $HOME/scripts/triton/
cp $scripts_dir/triton-tensorrt-llm.sh $HOME/scripts/triton/
chmod a+x $HOME/scripts/triton/*.sh
mkdir -p $HOME/cache

docker compose -f $DIR/compose/compose-triton-cuda.yaml up -d
else
docker compose -f $DIR/compose/compose-triton-cuda.yaml down
fi
