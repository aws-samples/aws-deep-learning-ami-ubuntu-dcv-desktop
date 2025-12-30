#!/bin/bash

# Validation
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ -z "$MODEL_ID" ] && echo "MODEL_ID must be set" && exit 1
[ -z "$DEVICE" ] && echo "DEVICE must be set" && exit 1

# Defaults
VLLM_NEURON_USE_V1=${VLLM_NEURON_USE_V1:-true}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}

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

cat > /tmp/config.yaml <<EOF
served-model-name: $MODEL_ID
tokenizer: $MODEL_ID
model-impl: auto
tensor-parallel-size: $TENSOR_PARALLEL_SIZE
max-num-seqs: $MAX_NUM_SEQS
dtype: auto
max-model-len: $MAX_MODEL_LEN
max-num-batched-tokens: $(($MAX_NUM_SEQS * $MAX_MODEL_LEN))
gpu-memory-utilization: $GPU_MEMORY_UTILIZATION
task: "classify"
EOF

# Final Execution
export VLLM_CONFIG=/tmp/config.yaml

/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"