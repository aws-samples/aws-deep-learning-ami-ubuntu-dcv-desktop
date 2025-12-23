#!/bin/bash

# Validation
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ -z "$MODEL_ID" ] && echo "MODEL_ID must be set" && exit 1
[ -z "$DEVICE" ] && echo "DEVICE must be set" && exit 1

# Defaults
VLLM_NEURON_USE_V1=${VLLM_NEURON_USE_V1:-false}
BLOCK_SIZE=${BLOCK_SIZE:-16}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

# Triton config
cat > /tmp/config.pbtxt <<EOF
backend: "vllm"
max_batch_size: 0
model_transaction_policy { decoupled: true }
input [ 
  { name: "text_input", data_type: TYPE_STRING, dims: [1] },
  { name: "stream", data_type: TYPE_BOOL, dims: [1], optional: true },
  { name: "sampling_parameters", data_type: TYPE_STRING, dims: [1], optional: true },
  { name: "exclude_input_in_output", data_type: TYPE_BOOL, dims: [1], optional: true }
] 
output [ { name: "text_output", data_type: TYPE_STRING, dims: [-1] } ]
instance_group [ { count: 1, kind: KIND_MODEL } ]
EOF

# Neuron setup
if [ "$DEVICE" = "neuron" ]; then
  [ ! -d /cache ] && echo "/cache dir must exist" && exit 1
  CACHE_DIR=/cache
  instance_family=$(/opt/aws/neuron/bin/neuron-ls | grep instance-type | awk -F': ' '{split($2, a, "."); print a[1]}')
  export NEURON_CORES_PER_DEVICE=$(/opt/aws/neuron/bin/neuron-ls --json-output | grep nc_count | head -1 | awk -F': ' '{print $2}' | tr -d ',')
  export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries --target=${instance_family}"
  export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
  export FI_EFA_FORK_SAFE=1
  export NEURON_COMPILED_ARTIFACTS="$CACHE_DIR/$MODEL_ID/neuron-compiled-artifacts-tp-$TENSOR_PARALLEL_SIZE"
  mkdir -p "$NEURON_COMPILED_ARTIFACTS"
fi

# Base model config
cat > /tmp/model.json <<EOF
{
  "model": "$MODEL_ID",
  "tokenizer": "$MODEL_ID",
  "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
  "max_num_seqs": $MAX_NUM_SEQS,
  "dtype": "auto",
  "max_model_len": $MAX_MODEL_LEN,
  "max_num_batched_tokens": $MAX_MODEL_LEN,
EOF

# Add VLLM V0 options
if [ "$DEVICE" = "neuron" ] && [ "$VLLM_NEURON_USE_V1" = "false" ]; then
  cat >> /tmp/model.json <<EOF
  "preemption_mode": "swap",
  "swap_space": 4,
  "override_neuron_config": {
    "continuous_batching": {
      "max_num_seqs": $MAX_NUM_SEQS,
      "optimized_paged_attention": true
    }
  }
EOF
else
  cat >> /tmp/model.json <<EOF
  "block_size": $BLOCK_SIZE
EOF
fi

# Close JSON
echo "}" >> /tmp/model.json

# Deploy
export MODEL_REPO=/opt/ml/model/model_repo
mkdir -p $MODEL_REPO/model/1
cp /tmp/model.json $MODEL_REPO/model/1/model.json
cp /tmp/config.pbtxt $MODEL_REPO/model/config.pbtxt

/opt/program/serve && /bin/bash -c "trap : TERM INT; sleep infinity & wait"