#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -x "$(command -v nvidia-smi)" ]
then
    pattern="*cuda*.sh"
elif [ -x "$(command -v neuron-top)" ]
then
    pattern="*neuron*.sh"
else
    echo "No supported accelerator detected (need nvidia-smi or neuron-top)."
    exit 1
fi

failed=0
for script in $(ls $DIR/$pattern)
    do
        name=$(basename $script .sh)
        # One log per container, otherwise each build overwrites the previous one's log
        log=/tmp/build-$name.log
        echo "Building $name. See logs: $log"
        bash $script 1>$log 2>&1
        status=$?
        if [ $status -ne 0 ]
        then
            echo "Building $name: failed. Last 20 lines of $log:"
            tail -20 $log
            failed=1
        else
            echo "Building $name: success"
        fi
    done

exit $failed
