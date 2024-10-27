#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1


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
  "model": "$MODEL_ID",
  "disable_log_requests": true,
  "tensor_parallel_size": 8,
  "max_num_seqs": 8,
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
export MODEL_SERVER_CORES=8
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"