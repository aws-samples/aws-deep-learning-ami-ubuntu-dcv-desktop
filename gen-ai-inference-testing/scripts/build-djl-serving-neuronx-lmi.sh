#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/djl-serving-neuronx-lmi
docker buildx build -t djl-serving-neuronx-lmi:latest .