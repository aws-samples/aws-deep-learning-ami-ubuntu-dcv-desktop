#!/bin/bash

# 1. Validation: Snapshots dir and Model ID are required for both
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[ -z "$MODEL_ID" ] && echo "MODEL_ID environment variable must exist" && exit 1
[ -z "$DEVICE" ] && echo "DEVICE environment variable must exist" && exit 1

# 2. Defaults
: ${TENSOR_PARALLEL_SIZE:=8}
: ${MAX_MODEL_LEN:=8192}
: ${OMP_NUM_THREADS:=16}
: ${MAX_NUM_SEQS:=8}

# 3. Hardware-specific setup
if [ "$DEVICE" == "neuron" ]; then
    [ ! -d /cache ] && echo "/cache dir must exist for neuron" && exit 1
    
    CACHE_DIR=/cache
    mkdir -p "$CACHE_DIR"
    
    export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
    export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
fi

# 4. Generate serving.properties
cat > /opt/ml/model/serving.properties <<EOF
option.model_id=$MODEL_ID
option.tensor_parallel_degree=$TENSOR_PARALLEL_SIZE
option.dtype=fp16
option.max_model_len=$MAX_MODEL_LEN
option.max_num_batched_tokens=$MAX_MODEL_LEN
option.model_loading_timeout=1800
option.rolling_batch=vllm
option.max_rolling_batch_size=$MAX_NUM_SEQS
option.output_formatter=json
EOF

# 5. Start the service
/usr/local/bin/dockerd-entrypoint.sh serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"