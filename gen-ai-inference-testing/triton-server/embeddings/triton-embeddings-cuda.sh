#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

TENSOR_PARALLEL_SIZE=1 # Force Tensor Parallel Size to 1 for encoder
: ${MAX_MODEL_LEN:=512}
: ${MAX_NUM_SEQS:=4}
: ${OMP_NUM_THREADS:=16}

cat > /tmp/config.pbtxt <<EOF
  backend: "python"
  max_batch_size: $MAX_NUM_SEQS
  model_transaction_policy {
    decoupled: false
  }
  dynamic_batching {
    max_queue_delay_microseconds: 1000
  }

  input [ 
    {
      name: "text_input"
      data_type: TYPE_STRING
      dims: [1]
    }
  ] 
  output [
    {
      name: "embeddings"
      data_type: TYPE_FP32
      dims: [-1, -1]
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
    "model_id_or_path": "$MODEL_ID"
  }

EOF

export MODEL_REPO=/opt/ml/model/model_repo
mkdir -p $MODEL_REPO
VERSION=1
MODEL_NAME=model
mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /scripts/triton_embeddings_backend.py $MODEL_REPO/$MODEL_NAME/$VERSION/model.py
cp /tmp/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /tmp/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt
export OMP_NUM_THREADS
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"