#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

# Define a temporary working directory
TEMP_MODEL_DIR="/tmp/model_working_dir"
echo "Creating a safe local copy in $TEMP_MODEL_DIR to prevent original weight corruption..."

# Clean up any previous failed attempts and create the dir
rm -rf "$TEMP_MODEL_DIR"
mkdir -p "$TEMP_MODEL_DIR"

# Copy weights. -L follows symlinks (important for HuggingFace local caches)
# and -p preserves permissions.
cp -rpL "$MODEL_ID/." "$TEMP_MODEL_DIR/"

# Update the variable so the rest of the script uses the local copy
MODEL_ID="$TEMP_MODEL_DIR"

: ${TENSOR_PARALLEL_SIZE:=8}
: ${MAX_MODEL_LEN:=8192}
: ${MAX_NUM_SEQS:=8}
: ${OMP_NUM_THREADS:=16}

cat > /opt/ml/model/serving.properties <<EOF
option.model_id=$MODEL_ID
option.entryPoint=djl_python.tensorrt_llm
option.tensor_parallel_degree=$TENSOR_PARALLEL_SIZE
option.max_num_tokens=$MAX_MODEL_LEN
option.dtype=fp16
option.rolling_batch=trtllm
option.max_rolling_batch_size=$MAX_NUM_SEQS
option.output_formatter=json
option.trust_remote_code=true
EOF

/usr/local/bin/dockerd-entrypoint.sh \
serve  \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"


