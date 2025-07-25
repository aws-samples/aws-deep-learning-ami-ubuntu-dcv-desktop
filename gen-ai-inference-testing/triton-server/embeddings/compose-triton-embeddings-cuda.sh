#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

export IMAGE=docker.io/library/tritonserver-cuda-vllm:latest
export COMMAND="/scripts/triton-embeddings-cuda.sh"
export HF_HOME=/snapshots/huggingface

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

if [ "$1" == "up" ]
then
mkdir -p $HOME/scripts/triton
cp $scripts_dir/triton-embeddings-cuda.sh $HOME/scripts/triton/
cp $scripts_dir/triton_embeddings_backend.py $HOME/scripts/triton/
chmod a+x $HOME/scripts/triton/*.sh
mkdir -p $HOME/cache

docker compose -f $DIR/compose/compose-triton-cuda.yaml up -d 
else
docker compose -f $DIR/compose/compose-triton-cuda.yaml down 
fi
