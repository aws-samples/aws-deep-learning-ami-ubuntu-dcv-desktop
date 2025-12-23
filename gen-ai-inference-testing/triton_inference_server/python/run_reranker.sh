#!/bin/bash

# 1. Common Validation
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ -z "$MODEL_ID" ] && echo "MODEL_ID environment variable must exist" && exit 1
[ -z "$DEVICE" ] && echo "DEVICE environment variable must exist" && exit 1

# 2. Defaults & Shared Variables
TENSOR_PARALLEL_SIZE=1 # Force Tensor Parallel Size to 1 for encoder
: ${MAX_MODEL_LEN:=512}
: ${MAX_NUM_SEQS:=8}
: ${OMP_NUM_THREADS:=16}

# 3. Hardware-specific setup
if [ "$DEVICE" == "neuron" ]; then
    [ ! -d /cache ] && echo "/cache dir must exist for neuron" && exit 1
    
    export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
    export NEURON_COMPILE_CACHE_URL="/cache"
    export FI_EFA_FORK_SAFE=1
    export NEURON_COMPILED_ARTIFACTS="$CACHE_DIR/$MODEL_ID/neuron-compiled-artifacts-tp-$TENSOR_PARALLEL_SIZE"
    mkdir -p "$NEURON_COMPILED_ARTIFACTS"
    echo "Running in Neuron mode..."
else
    echo "Running in CUDA mode..."
fi

# 4. Generate Triton Config (config.pbtxt)
cat > /tmp/config.pbtxt <<EOF
backend: "python"
max_batch_size: 0
model_transaction_policy {
  decoupled: false
}

input [ 
  {
    name: "query"
    data_type: TYPE_STRING
    dims: [1]
  },
  {
    name: "texts"
    data_type: TYPE_STRING
    dims: [-1]
  }
] 
output [
  {
    name: "scores"
    data_type: TYPE_FP32
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

# 5. Generate Model Metadata (model.json)
cat > /tmp/model.json <<EOF
  {
    "model_id_or_path": "$MODEL_ID",
    "bucket_batch_size": [1,  $(( (MAX_NUM_SEQS / 2) < 1 ? 1 : (MAX_NUM_SEQS / 2) )), $MAX_NUM_SEQS ],
    "bucket_seq_len": [$(($MAX_MODEL_LEN/2)), $MAX_MODEL_LEN]
  }
EOF

# 6. Model Repository Setup
export MODEL_REPO=/opt/ml/model/model_repo
VERSION=1
MODEL_NAME=model

mkdir -p "$MODEL_REPO/$MODEL_NAME/$VERSION"

cp /scripts/reranker.py "$MODEL_REPO/$MODEL_NAME/$VERSION/model.py"
cp /scripts/encoder_base.py "$MODEL_REPO/$MODEL_NAME/$VERSION/encoder_base.py"
cp /tmp/model.json "$MODEL_REPO/$MODEL_NAME/$VERSION/model.json"
cp /tmp/config.pbtxt "$MODEL_REPO/$MODEL_NAME/config.pbtxt"

# 7. Final Execution
export OMP_NUM_THREADS
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"