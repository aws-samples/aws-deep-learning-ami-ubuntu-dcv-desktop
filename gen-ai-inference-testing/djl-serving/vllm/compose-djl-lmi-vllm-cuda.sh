#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

export IMAGE="deepjavalibrary/djl-serving:0.32.0-lmi"
export COMMAND="/scripts/djl-lmi-vllm-cuda.sh"
export HF_HOME=/snapshots/huggingface

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..


if [ "$1" == "up" ]
then
mkdir -p $HOME/scripts/djl-lmi
cp $scripts_dir/djl-lmi-vllm-cuda.sh $HOME/scripts/djl-lmi/
chmod a+x $HOME/scripts/djl-lmi/*.sh

docker compose -f $DIR/compose/compose-djl-lmi-cuda.yaml up -d 
else
docker compose -f $DIR/compose/compose-djl-lmi-cuda.yaml down 
fi
