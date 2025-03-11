#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

export IMAGE=docker.io/library/apiserver-cuda-vllm:latest
export COMMAND="/scripts/apiserver-vllm-cuda.sh"
export HF_HOME=/snapshots/huggingface

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

if [ "$1" == "up" ]
then
mkdir -p $HOME/scripts/apiserver
cp $scripts_dir/apiserver-vllm-cuda.sh $HOME/scripts/apiserver/
chmod a+x $HOME/scripts/apiserver/*.sh
mkdir -p $HOME/cache

docker compose -f $DIR/compose/compose-apiserver-cuda.yaml up -d
else
docker compose -f $DIR/compose/compose-apiserver-cuda.yaml down 
fi
