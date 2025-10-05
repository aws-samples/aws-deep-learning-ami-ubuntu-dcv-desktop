#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

cat > /tmp/config.yaml <<EOF
tokenizer: $MODEL_ID
model-impl: auto
enable-log-requests: false
tensor-parallel-size: $TENSOR_PARALLEL_SIZE
max-num-seqs: $MAX_NUM_SEQS
dtype: auto
max-model-len: $MAX_MODEL_LEN
gpu-memory-utilization: 0.95
enforce-eager: false
enable-prefix-caching: false
enable-chunked-prefill: true
preemption-mode: swap
swap-space: 4
max-num-batched-tokens: $MAX_MODEL_LEN
EOF

export VLLM_CONFIG=/tmp/config.yaml

/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"