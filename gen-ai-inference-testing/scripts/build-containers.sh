#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -x "$(command -v nvidia-smi)" ]
then
    for script in `ls $DIR/*cuda*.sh`
        do
            echo "Building $(basename $script). See logs: /tmp/buid.log"
            bash $script 1>/tmp/build.log 2>&1 
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
    for script in `ls $DIR/*neuron*.sh`
        do
            echo "Building $(basename $script). See logs: /tmp/build.log"
            bash $script 1>/tmp/build.log 2>&1 
            status=$?
            if [ $status -ne 0 ]
            then 
                echo "Building $(basename $script): failed"; 
            else
                echo "Building $(basename $script): success";
            fi
        done
fi