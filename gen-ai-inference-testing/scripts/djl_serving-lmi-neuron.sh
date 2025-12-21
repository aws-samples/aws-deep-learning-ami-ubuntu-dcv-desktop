#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/djl_serving-lmi-neuron
docker buildx build -t djl_serving-lmi:neuron .