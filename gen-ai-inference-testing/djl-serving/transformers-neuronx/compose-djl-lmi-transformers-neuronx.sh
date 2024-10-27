#!/bin/bash

[ $# -ne 1 ] && echo "usage: $0 <up/down>" && exit 1

export IMAGE="deepjavalibrary/djl-serving:0.29.0-pytorch-inf2"
export COMMAND="/scripts/djl-lmi-transformers-neuronx.sh"
export HF_HOME=/snapshots/huggingface

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..


if [ "$1" == "up" ]
then
mkdir -p $HOME/scripts/djl-lmi
cp $scripts_dir/djl-lmi-transformers-neuronx.sh $HOME/scripts/djl-lmi/
chmod a+x $HOME/scripts/djl-lmi/*.sh
mkdir -p $HOME/cache

docker compose -f $DIR/compose/compose-djl-lmi-neuronx.yaml up -d 
else
docker compose -f $DIR/compose/compose-djl-lmi-neuronx.yaml down 
fi
