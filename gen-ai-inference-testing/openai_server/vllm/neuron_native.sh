#!/bin/bash
# vLLM on the native TorchNeuron stack (vllm-neuron 0.29 plugin, Neuron SDK 2.33).
# run.sh hands over to this script when the image's stack is "native", which
# scripts/openai_server-vllm-neuron.sh selects on trn2/trn3.
#
# Standard inputs (compose/neuron.yaml, same as every engine):
#   MODEL_ID, TENSOR_PARALLEL_SIZE, MAX_MODEL_LEN, MAX_NUM_SEQS, TRUST_REMOTE_CODE
#
# Native-only inputs. compose/neuron.yaml has no slot for them, so they are read
# from /scripts/neuron_native.env ($HOME/scripts/neuron_native.env on the host)
# when it exists. The notebooks write it. Every value has a default. The file is
# parsed, not executed: only the keys below (and VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
# OMP_NUM_THREADS, NGINX_TIMEOUT_S) with plain values are applied.
#   PREFIX_CACHING=false|true   vLLM automatic prefix caching (default false, as
#                               on the other vLLM paths). Each setting compiles
#                               separately. On this stack a prefix-cache hit was
#                               measured to take longer to the first token than
#                               a fresh prefill; test with your own traffic.
#   NUM_REPLICAS=<n>            cap on servers (default: every TP-sized core group)
#   BUCKET_PROFILE=ladder|recipe
#       ladder (default): decode batch buckets 1,2,4,.. up to MAX_NUM_SEQS and
#         prefill buckets 1024..8192 up to MAX_NUM_BATCHED_TOKENS, so short
#         prompts and small batches are not padded up to the largest shape.
#       recipe: one bucket each, [MAX_NUM_BATCHED_TOKENS]/[MAX_NUM_SEQS] -- the
#         shape of AWS's published model recipes.
#   MAX_NUM_BATCHED_TOKENS      prefill chunk size (default min(8192, MAX_MODEL_LEN))
#   OPT_LEVEL                   --optimization-level (default 2)

set -e

[ ! -d /cache ] && echo "/cache dir must exist" && exit 1
instance_family=$(/opt/aws/neuron/bin/neuron-ls | awk -F': ' '/instance-type/ {split($2, a, "."); print a[1]}')
case "$instance_family" in
    trn2*|trn3*) ;;
    *) echo "This image carries the native TorchNeuron stack, which Neuron supports on trn2/trn3"
       echo "only (detected: '$instance_family'). Rebuild with NEURON_VLLM_STACK=legacy (the"
       echo "default off trn2/trn3): bash scripts/openai_server-vllm-neuron.sh"
       exit 1 ;;
esac

OVERRIDES=/scripts/neuron_native.env
ALLOWED=" PREFIX_CACHING NUM_REPLICAS BUCKET_PROFILE MAX_NUM_BATCHED_TOKENS OPT_LEVEL VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS OMP_NUM_THREADS NGINX_TIMEOUT_S "
if [ -f "$OVERRIDES" ]; then
    echo "Applying $OVERRIDES (written $(date -r "$OVERRIDES" -u +%Y-%m-%dT%H:%M:%SZ)):"
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" == \#* ]] && continue
        if [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)=([A-Za-z0-9._-]*)$ ]] && [[ "$ALLOWED" == *" ${BASH_REMATCH[1]} "* ]]; then
            export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
            echo "    ${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
        else
            echo "    WARNING: ignored line (unknown key or unsupported value): $line"
        fi
    done < "$OVERRIDES"
fi

TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
PREFIX_CACHING=${PREFIX_CACHING:-false}
BUCKET_PROFILE=${BUCKET_PROFILE:-ladder}
OPT_LEVEL=${OPT_LEVEL:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$(( MAX_MODEL_LEN < 8192 ? MAX_MODEL_LEN : 8192 ))}
export NUM_REPLICAS=${NUM_REPLICAS:-}

# Native stack environment. Must be set before Python starts.
unset PJRT_DEVICE                                   # selects the legacy XLA path
export VLLM_NEURON_BACKEND=neuron_native            # the precompile cache needs it set explicitly
export NEURON_LOGICAL_NC_CONFIG=${NEURON_LOGICAL_NC_CONFIG:-2}
# MANDATORY in a container: without it every TP worker dies with "No EFA device
# found at /sys/bus/pci/devices/.../infiniband". EFA affinity is a CPU
# optimisation for single-node inference, not a correctness requirement.
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/cache}
# NKI kernel binaries must persist WITH the compile cache. Their default home,
# /var/tmp/nki-intermediate-cache, dies with the container while /cache survives;
# a later start that reuses cached graphs but compiles a new shape then fails in
# neuronx-cc with NCC_EVRF059 "Kernel file ... does not exist on the host".
export NKI_COMPILE_CACHE_URL=${NKI_COMPILE_CACHE_URL:-$VLLM_CACHE_ROOT/nki}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
# Logical cores per device (4 on trn2 at LNC2); serve splits on this.
export NEURON_CORES_PER_DEVICE=$(/opt/aws/neuron/bin/neuron-ls --json-output | grep nc_count | head -1 | awk -F': ' '{print $2}' | tr -d ',')
# Long-prompt requests under load can take minutes end to end.
export NGINX_TIMEOUT_S=${NGINX_TIMEOUT_S:-1800}

# Buckets. Neuron's torch.compile backend has no dynamic shapes: every bucket is
# a separately compiled graph, and a shape outside them is padded up to the next.
case "$BUCKET_PROFILE" in
    recipe)
        NUM_SEQS_BUCKETS="[$MAX_NUM_SEQS]"
        BATCHED_TOKENS_BUCKETS="[$MAX_NUM_BATCHED_TOKENS]"
        ;;
    ladder)
        b=1; seqs=""
        while [ $b -lt $MAX_NUM_SEQS ]; do seqs="$seqs$b,"; b=$((b * 2)); done
        NUM_SEQS_BUCKETS="[${seqs}${MAX_NUM_SEQS}]"
        toks=""
        for t in 1024 2048 4096 8192; do [ $t -lt $MAX_NUM_BATCHED_TOKENS ] && toks="$toks$t,"; done
        BATCHED_TOKENS_BUCKETS="[${toks}${MAX_NUM_BATCHED_TOKENS}]"
        ;;
    *)
        echo "BUCKET_PROFILE must be ladder or recipe (got '$BUCKET_PROFILE')"; exit 1 ;;
esac
# Segmented prefill (prefill over KV that is already cached), one segment size =
# the chunk size. Needed when a prompt can span several chunks, and ALWAYS with
# prefix caching: the plugin serves a prefix-cache hit only through segmented
# prefill and refuses to start without it ("APC requires segmented prefill"),
# and it leaves segmented prefill off when MAX_NUM_BATCHED_TOKENS >= MAX_MODEL_LEN.
SEGMENT_SIZES=" 512 1024 2048 4096 8192 "
if [ "$PREFIX_CACHING" = "true" ] || [ $MAX_NUM_BATCHED_TOKENS -lt $MAX_MODEL_LEN ]; then
    if [[ "$SEGMENT_SIZES" != *" $MAX_NUM_BATCHED_TOKENS "* ]]; then
        echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS must be one of${SEGMENT_SIZES}for segmented"
        echo "prefill (needed for PREFIX_CACHING=true or MAX_MODEL_LEN > MAX_NUM_BATCHED_TOKENS)."
        exit 1
    fi
    KV_SEGMENT_LINE="    kv_segment_size_buckets: [$MAX_NUM_BATCHED_TOKENS]"
else
    KV_SEGMENT_LINE=""
fi

case "$PREFIX_CACHING" in
    true)  PREFIX_LINE="enable-prefix-caching: true" ;;
    false) PREFIX_LINE="no-enable-prefix-caching: true" ;;
    *) echo "PREFIX_CACHING must be true or false (got '$PREFIX_CACHING')"; exit 1 ;;
esac

# block-size is left to the plugin (it resolves its own on Neuron).
cat > /tmp/config.yaml <<EOF
tokenizer: $MODEL_ID
dtype: bfloat16
optimization-level: $OPT_LEVEL
tensor-parallel-size: $TENSOR_PARALLEL_SIZE
max-num-seqs: $MAX_NUM_SEQS
max-model-len: $MAX_MODEL_LEN
max-num-batched-tokens: $MAX_NUM_BATCHED_TOKENS
trust-remote-code: $TRUST_REMOTE_CODE
$PREFIX_LINE
additional-config:
  neuron_config:
    num_seqs_buckets: $NUM_SEQS_BUCKETS
    num_batched_tokens_buckets: $BATCHED_TOKENS_BUCKETS
$KV_SEGMENT_LINE
EOF
echo "=== vLLM native TorchNeuron config (instance: $instance_family, buckets: $BUCKET_PROFILE, replicas: ${NUM_REPLICAS:-all}) ==="
cat /tmp/config.yaml

export VLLM_CONFIG=/tmp/config.yaml
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"
