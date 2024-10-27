#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ ! -d /cache ] && echo "/cache dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

MODEL_PATH=/snapshots/$MODEL_ID
if [ ! -d $MODEL_PATH ]
then
pip3 install -U "huggingface_hub[cli]"
huggingface-cli download --repo-type model \
    --local-dir $MODEL_PATH \
    --token $HF_TOKEN  $MODEL_ID
fi

CACHE_DIR=/cache

cat > /tmp/config.pbtxt <<EOF
backend: "vllm"
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
      name: "stream"
      data_type: TYPE_BOOL
      dims: [1]
      optional: true
  },
  {
      name: "sampling_parameters"
      data_type: TYPE_STRING
      dims: [1]
      optional: true
  },
  {
      name: "exclude_input_in_output"
      data_type: TYPE_BOOL
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
  "model": "$MODEL_PATH",
  "disable_log_requests": true,
  "tensor_parallel_size": 8,
  "max_num_seqs": 4,
  "dtype": "float16",
  "max_model_len": 8192,
  "block_size": 8192,
  "use_v2_block_manager": true
}

EOF

export MODEL_REPO=/opt/ml/model/model_repo
mkdir -p $MODEL_REPO
VERSION=1
MODEL_NAME=model
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /tmp/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /tmp/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt

export NEURON_CC_FLAGS="--model-type=transformer"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export OMP_NUM_THREADS=32
export MODEL_SERVER_CORES=8
export FI_EFA_FORK_SAFE=1
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"