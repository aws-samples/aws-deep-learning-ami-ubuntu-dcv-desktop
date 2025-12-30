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
: ${BLOCK_SIZE:=16}

# 3. Generate serving.properties
cat > /opt/ml/model/serving.properties <<EOF
engine=Python
option.entryPoint=djl_python.lmi_vllm.vllm_async_service
option.rolling_batch=disable
option.async_mode=True
option.model_id=$MODEL_ID
option.tensor_parallel_degree=$TENSOR_PARALLEL_SIZE
option.max_model_len=$MAX_MODEL_LEN
option.max_num_batched_tokens=$MAX_MODEL_LEN
option.model_loading_timeout=1800
option.max_rolling_batch_size=$MAX_NUM_SEQS
option.block_size=$BLOCK_SIZE
# Uncomment this line for testing models and tokenizers that need remote code access
# option.trust_remote_code=true 
EOF

# 4. Hardware-specific setup
if [ "$DEVICE" == "neuron" ]; then
    [ ! -d /cache ] && echo "/cache dir must exist for neuron" && exit 1
    
    CACHE_DIR=/cache
    mkdir -p "$CACHE_DIR"
    
    export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
    export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
    cat >> /opt/ml/model/serving.properties <<EOF
option.enable_prefix_caching=False
option.enable_chunked_prefill=False
EOF
fi



# 5. Start the service
/usr/local/bin/dockerd-entrypoint.sh serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"