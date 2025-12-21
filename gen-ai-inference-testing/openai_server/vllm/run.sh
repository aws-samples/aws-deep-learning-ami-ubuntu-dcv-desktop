#!/bin/bash

# 1. Common validation
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ -z "$MODEL_ID" ] && echo "MODEL_ID environment variable must exist" && exit 1

# 2. Hardware-specific logic
if [ "$DEVICE" = "neuron" ]; then
    echo "Configuring for AWS Neuron..."
    [ ! -d /cache ] && echo "/cache dir must exist" && exit 1
    
    # Identify instance and core count
    CACHE_DIR=/cache
    instance_family=$(/opt/aws/neuron/bin/neuron-ls | grep instance-type | awk -F': ' '{split($2, a, "."); print a[1]}')
    export NEURON_CORES_PER_DEVICE=$(/opt/aws/neuron/bin/neuron-ls --json-output | grep nc_count | head -1 | awk -F': ' '{print $2}' | tr -d ',')
    
    echo "Neuron instance family: $instance_family, Cores per device: $NEURON_CORES_PER_DEVICE"
    
    # Neuron-specific exports
    export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries --target=${instance_family}"
    export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
    export FI_EFA_FORK_SAFE=1
    export NEURON_COMPILED_ARTIFACTS=$MODEL_ID/neuron-compiled-artifacts

    # Neuron-specific config settings
    LOG_STATS_SETTING="disable-log-stats: true"
    EXTRA_NEURON_CONFIG=$(cat <<EOF
preemption-mode: swap
swap-space: 4
EOF
)
else
    echo "Configuring for CUDA..."
    LOG_STATS_SETTING="enable-log-requests: false"
    EXTRA_NEURON_CONFIG=""
fi

# 3. Generate the unified config.yaml
# We use a variable for the log setting and append Neuron-specific lines if they exist
cat > /tmp/config.yaml <<EOF
tokenizer: $MODEL_ID
model-impl: auto
$LOG_STATS_SETTING
tensor-parallel-size: $TENSOR_PARALLEL_SIZE
max-num-seqs: $MAX_NUM_SEQS
dtype: auto
max-model-len: $MAX_MODEL_LEN
gpu-memory-utilization: 0.95
enforce-eager: false
enable-prefix-caching: false
enable-chunked-prefill: $([ "$DEVICE" = "cuda" ] && echo "true" || echo "false")
max-num-batched-tokens: $MAX_MODEL_LEN
$EXTRA_NEURON_CONFIG
EOF

# 4. Final Execution
export VLLM_CONFIG=/tmp/config.yaml

/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"