#!/bin/bash


[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1
export IMAGE=docker.io/library/openai-server-neuronx-vllm:latest
export COMMAND="/scripts/openai-server-vllm-neuronx.sh"
export HF_HOME=/snapshots/huggingface

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

if [ "$1" == "up" ]
then
mkdir -p $HOME/scripts/openai-server
cp $scripts_dir/openai-server-vllm-neuronx.sh $HOME/scripts/openai-server/
chmod a+x $HOME/scripts/openai-server/*.sh
mkdir -p $HOME/cache

docker compose -f $DIR/compose/compose-openai-server-neuronx-${NUM_DEVICE}.yaml up -d 
else
docker compose -f $DIR/compose/compose-openai-server-neuronx-${NUM_DEVICE}.yaml down 
fi
