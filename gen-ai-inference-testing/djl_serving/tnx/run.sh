#!/bin/bash

[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

CACHE_DIR=/cache
mkdir -p $CACHE_DIR

: ${TENSOR_PARALLEL_SIZE:=8}
: ${MAX_MODEL_LEN:=8192}
: ${MAX_NUM_SEQS:=8}
: ${OMP_NUM_THREADS:=16}


cat > /opt/ml/model/serving.properties <<EOF
option.model_id=$MODEL_ID
option.entryPoint=djl_python.transformers_neuronx
option.tensor_parallel_degree=$TENSOR_PARALLEL_SIZE
option.amp=f16
option.n_positions=$MAX_MODEL_LEN
option.model_loading_timeout=1800
option.rolling_batch=auto
option.rolling_batch_strategy=continuous_batching
option.max_rolling_batch_size=$MAX_NUM_SEQS
option.output_formatter=json
option.trust_remote_code=true

EOF

instance_family=$(/opt/aws/neuron/bin/neuron-ls | grep instance-type | awk -F': ' '{split($2, a, "."); print a[1]}')
export NEURON_CORES_PER_DEVICE=$(/opt/aws/neuron/bin/neuron-ls --json-output | grep nc_count | head -1 | awk -F': ' '{print $2}' | tr -d ',')
echo "Neuron instance family: $instance_family, Number of neuron cores per device: $NEURON_CORES_PER_DEVICE"
export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries --target=${instance_family}"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
/usr/local/bin/dockerd-entrypoint.sh serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"

