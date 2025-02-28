#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

if [ -x "$(command -v nvidia-smi)" ]
then
    for script in `ls $DIR/scripts/*cuda*.sh`
        do
            echo "Building $(basename $script): This may take several minutes...."
            source $script 1>build.log 2>&1 
            echo "Building $(basename $script): Completed"
        done
elif [ -x "$(command -v neuron-top)" ]
then
    for script in `ls $DIR/scripts/*neuron*.sh`
        do
            echo "Building $(basename $script): This may take several minutes...."
            source $script 1>build.log 2>&1 
            echo "Building $(basename $script): Completed"
        done
fi