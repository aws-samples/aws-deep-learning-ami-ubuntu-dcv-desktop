#!/bin/bash

[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1


CACHE_DIR=/cache

cat > /tmp/config.pbtxt <<EOF
backend: "python"
max_batch_size: 0
model_transaction_policy {
    decoupled: true
}

input [ 
    {
        name: "text_input"
        data_type: TYPE_STRING
        dims: [1]
    },
    {
        name: "sampling_parameters"
        data_type: TYPE_STRING
        dims: [1]
        optional: true
    }
] 
output [
    {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [-1]
    }
]

instance_group [
    {
    count: 1
    kind: KIND_MODEL
    }
]

EOF

cat > /tmp/model.json <<EOF
{
  "model_id": "$MODEL_ID",
  "tensor_parallel_degree": 8,
  "amp": "f16",
  "n_positions": 8192,
  "model_loading_timeout": 1800,
  "model_loader": "tnx",
  "rolling_batch": "auto",
  "rolling_batch_strategy": "continuous_batching",
  "max_rolling_batch_size": 8,
  "output_formatter": "json",
  "trust_remote_code": true
}

EOF

export MODEL_REPO=/opt/ml/model/model_repo
mkdir -p $MODEL_REPO
VERSION=1
MODEL_NAME=model
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /scripts/tnx_lmi_backend.py $MODEL_REPO/$MODEL_NAME/$VERSION/model.py
cp /tmp/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /tmp/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt
export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export FI_EFA_FORK_SAFE=1
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"