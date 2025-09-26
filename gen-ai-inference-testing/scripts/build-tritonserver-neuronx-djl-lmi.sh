#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

cd $DIR/containers/tritonserver-neuronx-djl-lmi
docker buildx build -t tritonserver-neuronx-djl-lmi:latest .