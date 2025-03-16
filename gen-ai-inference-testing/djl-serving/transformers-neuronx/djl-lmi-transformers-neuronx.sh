#!/bin/bash

[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1

[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

CACHE_DIR=/cache
mkdir -p $CACHE_DIR

cat > /opt/ml/model/serving.properties <<EOF
option.model_id=$MODEL_ID
option.entryPoint=djl_python.transformers_neuronx
option.tensor_parallel_degree=8
option.amp=f16
option.n_positions=8192
option.model_loading_timeout=1800
option.model_loader=tnx
option.rolling_batch=auto
option.rolling_batch_strategy=continuous_batching
option.max_rolling_batch_size=8
option.output_formatter=json
option.trust_remote_code=true

EOF

export NEURON_CC_FLAGS="--model-type=transformer --enable-fast-loading-neuron-binaries"
export NEURON_COMPILE_CACHE_URL="$CACHE_DIR"
/usr/local/bin/dockerd-entrypoint.sh serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"

