#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

export NEURON_CC_FLAGS="--model-type=transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export OMP_NUM_THREADS=32
export MODEL_SERVER_CORES=8
export FI_EFA_FORK_SAFE=1
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"