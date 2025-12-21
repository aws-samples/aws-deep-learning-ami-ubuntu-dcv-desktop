#!/bin/bash

# 1. Common Validation
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ -z "$MODEL_ID" ] && echo "MODEL_ID environment variable must exist" && exit 1
[ -z "$DEVICE" ] && echo "DEVICE environment variable must exist" && exit 1

# 2. Defaults
: ${TENSOR_PARALLEL_SIZE:=8}
: ${MAX_MODEL_LEN:=8192}
: ${MAX_NUM_SEQS:=8}
: ${OMP_NUM_THREADS:=16}

# 3. Triton config.pbtxt (Shared)
cat > /tmp/config.pbtxt <<EOF
backend: "vllm"
max_batch_size: 0
model_transaction_policy {
  decoupled: true
}
input [ 
  { name: "text_input", data_type: TYPE_STRING, dims: [1] },
  { name: "stream", data_type: TYPE_BOOL, dims: [1], optional: true },
  { name: "sampling_parameters", data_type: TYPE_STRING, dims: [1], optional: true },
  { name: "exclude_input_in_output", data_type: TYPE_BOOL, dims: [1], optional: true }
] 
output [
  { name: "text_output", data_type: TYPE_STRING, dims: [-1] }
]
instance_group [
  { count: 1, kind: KIND_MODEL }
]
EOF

# 4. Device-Specific Logic
if [ "$DEVICE" == "neuron" ]; then
    [ ! -d /cache ] && echo "/cache dir must exist for neuron" && exit 1
    
    export VLLM_NEURON_FRAMEWORK=${VLLM_NEURON_FRAMEWORK:-"neuronx-distributed-inference"}
    CACHE_DIR=/cache

    # Neuron-specific model.json
    cat > /tmp/model.json <<EOF
{
  "model": "$MODEL_ID",
  "tokenizer": "$MODEL_ID",
  "disable_log_stats": true,
  "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
  "max_num_seqs": $MAX_NUM_SEQS,
  "dtype": "auto",
  "max_model_len": $MAX_MODEL_LEN,
  "gpu_memory_utilization": 0.95,
  "enforce_eager": false,
  "enable_prefix_caching": false,
  "preemption_mode": "swap",
  "max_num_batched_tokens": $MAX_MODEL_LEN,
  "override_neuron_config": {
      "continuous_batching": {
        "max_num_seqs": $MAX_NUM_SEQS,
        "optimized_paged_attention": true
      }
    }
}
EOF

    # Neuron Hardware Detection & Flags
    instance_family=$(/opt/aws/neuron/bin/neuron-ls | grep instance-type | awk -F': ' '{split($2, a, "."); print a[1]}')
    export NEURON_CORES_PER_DEVICE=$(/opt/aws/neuron/bin/neuron-ls --json-output | grep nc_count | head -1 | awk -F': ' '{print $2}' | tr -d ',')
    echo "Neuron instance family: $instance_family, Number of neuron cores per device: $NEURON_CORES_PER_DEVICE"
    
    export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries --target=${instance_family}"
    export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
    export FI_EFA_FORK_SAFE=1
    export NEURON_COMPILED_ARTIFACTS=$MODEL_ID/neuron-compiled-artifacts

else
    # CUDA-specific model.json
    cat > /tmp/model.json <<EOF
{
  "model": "$MODEL_ID",
  "tokenizer": "$MODEL_ID",
  "disable_log_stats": true,
  "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
  "max_num_seqs": $MAX_NUM_SEQS,
  "dtype": "auto",
  "max_model_len": $MAX_MODEL_LEN,
  "gpu_memory_utilization": 0.95,
  "enforce_eager": false,
  "enable_prefix_caching": false,
  "enable_chunked_prefill": true,
  "max_num_batched_tokens": $MAX_MODEL_LEN
}
EOF
fi

# 5. Final Setup and Execution
export MODEL_REPO=/opt/ml/model/model_repo
VERSION=1
MODEL_NAME=model

mkdir -p $MODEL_REPO/$MODEL_NAME/$VERSION
cp /tmp/model.json $MODEL_REPO/$MODEL_NAME/$VERSION/model.json
cp /tmp/config.pbtxt $MODEL_REPO/$MODEL_NAME/config.pbtxt

/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"