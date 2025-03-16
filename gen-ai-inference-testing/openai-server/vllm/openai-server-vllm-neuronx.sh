#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

: ${TENSOR_PARALLEL_SIZE:=8}
: ${MAX_MODEL_LEN:=8192}
: ${OMP_NUM_THRADS:=16}

CACHE_DIR=/cache
export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export FI_EFA_FORK_SAFE=1
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"