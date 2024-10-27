#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

export IMAGE="deepjavalibrary/djl-serving:0.29.0-tensorrt-llm"
export COMMAND="/scripts/djl-lmi-tensorrt-llm.sh"
export HF_HOME=/snapshots/huggingface

if [ "$1" == "up" ]
then
mkdir -p $HOME/scripts/djl-lmi
cp $scripts_dir/djl-lmi-tensorrt-llm.sh $HOME/scripts/djl-lmi/
chmod a+x $HOME/scripts/djl-lmi/*.sh

docker compose -f $DIR/compose/compose-djl-lmi-cuda.yaml up -d 
else
docker compose -f $DIR/compose/compose-djl-lmi-cuda.yaml down 
fi
