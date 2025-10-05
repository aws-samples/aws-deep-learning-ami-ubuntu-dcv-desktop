#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

export VLLM_NEURON_FRAMEWORK=${VLLM_NEURON_FRAMEWORK:-"neuronx-distributed-inference"}

CACHE_DIR=/cache
instance_family=$(/opt/aws/neuron/bin/neuron-ls | grep instance-type | awk -F': ' '{split($2, a, "."); print a[1]}')
export NEURON_CORES_PER_DEVICE=$(/opt/aws/neuron/bin/neuron-ls --json-output | grep nc_count | head -1 | awk -F': ' '{print $2}' | tr -d ',')
echo "Neuron instance family: $instance_family, Number of neuron cores per device: $NEURON_CORES_PER_DEVICE"
export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries --target=${instance_family}"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
export FI_EFA_FORK_SAFE=1
export NEURON_COMPILED_ARTIFACTS=$MODEL_ID/neuron-compiled-artifacts

cat > /tmp/config.yaml <<EOF
tokenizer: $MODEL_ID
model-impl: auto
disable-log-stats: true
tensor-parallel-size: $TENSOR_PARALLEL_SIZE
max-num-seqs: $MAX_NUM_SEQS
dtype: auto
max-model-len: $MAX_MODEL_LEN
gpu-memory-utilization: 0.95
enforce-eager: false
enable-prefix-caching: false
preemption-mode: swap
swap-space: 4
max-num-batched-tokens: $MAX_MODEL_LEN
EOF

export VLLM_CONFIG=/tmp/config.yaml

/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"