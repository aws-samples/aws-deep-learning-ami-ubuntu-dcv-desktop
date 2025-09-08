#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$scripts_dir/..

if [ -x "$(command -v nvidia-smi)" ]
then
    for script in `ls $DIR/scripts/*cuda*.sh`
        do
            echo "Building $(basename $script). See logs: /tmp/buid.log"
            source $script 1>/tmp/build.log 2>&1 
            status=$?
            if [ $status -ne 0 ]
            then 
                echo "Building $(basename $script): failed"; 
            else
                echo "Building $(basename $script): success";
            fi
        done
elif [ -x "$(command -v neuron-top)" ]
then
    for script in `ls $DIR/scripts/*neuron*.sh`
        do
            echo "Building $(basename $script). See logs: /tmp/build.log"
            source $script 1>/tmp/build.log 2>&1 
            status=$?
            if [ $status -ne 0 ]
            then 
                echo "Building $(basename $script): failed"; 
            else
                echo "Building $(basename $script): success";
            fi
        done
fi