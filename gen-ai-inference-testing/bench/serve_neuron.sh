#!/usr/bin/env bash
# Start the Neuron vLLM server through the repo's standard path and wait until every
# replica is up -- the scripted equivalent of the notebooks' launch cells:
#   launch.sh -> compose/neuron.yaml -> openai_server/vllm/run.sh
#             -> containers/openai_server-vllm-neuron (serve: replicas + nginx :8080)
#
# Usage (from gen-ai-inference-testing/):
#   MODEL_ID=Qwen/Qwen3-8B bash bench/serve_neuron.sh up
#   MODEL_DIR=$HOME/snapshots/Qwen3-32B MAX_MODEL_LEN=10240 MAX_NUM_SEQS=16 bash bench/serve_neuron.sh up
#   bash bench/serve_neuron.sh down
#   MODEL_ID=... bash bench/serve_neuron.sh precompile   # up, wait for every replica, down
#
# Every setting is an environment variable. Defaults are the notebooks': TP 8,
# MAX_MODEL_LEN 8192, MAX_NUM_SEQS 8, NEURON_VLLM_STACK auto. Native-stack knobs
# (PREFIX_CACHING, NUM_REPLICAS, BUCKET_PROFILE, MAX_NUM_BATCHED_TOKENS) are written to
# $HOME/scripts/neuron_native.env; the legacy stack ignores them.
#
# Model: MODEL_ID (Hugging Face id, downloaded into the HF cache under $HOME/snapshots)
# or MODEL_DIR (a checkpoint directory). launch.sh mounts only $HOME/snapshots, so
# MODEL_DIR must be inside it; if $HOME/snapshots does not exist it is created as a
# symlink to MODEL_DIR's parent.
#
# Cache: the server container runs as root (compose/neuron.yaml), so $HOME/cache must
# not resolve to EFS -- root-owned entries on a shared EFS block other users. Keep it
# local and copy it to shared storage as your own user, or set
# ALLOW_ROOT_CACHE_ON_EFS=1.
#
# Replica i listens on 8000+128*i inside the container; bench/client.sh reaches them.
set -euo pipefail

ACTION="${1:-up}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"                 # gen-ai-inference-testing/
CONTAINER="${CONTAINER:-compose-server_0-1}"

export INFERENCE_SERVER=openai_server INFERENCE_ENGINE=vllm DEVICE=neuron
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}" HF_TOKEN="${HF_TOKEN:-}"
export NEURON_VLLM_STACK="${NEURON_VLLM_STACK:-auto}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || echo us-east-2)}"

if [[ "$ACTION" == "down" ]]; then
  export MODEL_ID="${MODEL_ID:-unused}"
  cd "$REPO" && exec bash launch.sh down
fi

# --- model ---------------------------------------------------------------
if [[ -z "${MODEL_ID:-}" ]]; then
  [[ -n "${MODEL_DIR:-}" && -f "$MODEL_DIR/config.json" ]] || {
    echo "set MODEL_ID, or MODEL_DIR to a checkpoint directory (got '${MODEL_DIR:-}')" >&2; exit 1; }
  if [[ ! -e "$HOME/snapshots" ]]; then
    ln -s "$(dirname "$(realpath "$MODEL_DIR")")" "$HOME/snapshots"
    echo "linked $HOME/snapshots -> $(readlink "$HOME/snapshots")"
  fi
  snap="$(realpath "$HOME/snapshots")"; mdl="$(realpath "$MODEL_DIR")"
  [[ "$mdl" == "$snap"/* ]] || {
    echo "MODEL_DIR ($mdl) is not under \$HOME/snapshots ($snap), which is all launch.sh mounts." >&2
    echo "Move or copy the checkpoint under it, or set MODEL_ID to a Hugging Face id." >&2; exit 1; }
  export MODEL_ID="/snapshots/${mdl#"$snap"/}"
fi

# --- cache ---------------------------------------------------------------
mkdir -p "$HOME/cache"
cache_real="$(realpath "$HOME/cache")"
if [[ "$(stat -f -c %T "$cache_real")" == "nfs" && "${ALLOW_ROOT_CACHE_ON_EFS:-0}" != "1" ]]; then
  echo "\$HOME/cache resolves to NFS/EFS ($cache_real). The server runs as root and would" >&2
  echo "leave root-owned files there. Use a local cache, or ALLOW_ROOT_CACHE_ON_EFS=1." >&2; exit 1
fi

# --- native-stack settings (openai_server/vllm/neuron_native.sh) ---------
mkdir -p "$HOME/scripts"
cat > "$HOME/scripts/neuron_native.env" <<ENV
PREFIX_CACHING=${PREFIX_CACHING:-false}
NUM_REPLICAS=${NUM_REPLICAS:-}
BUCKET_PROFILE=${BUCKET_PROFILE:-ladder}
${MAX_NUM_BATCHED_TOKENS:+MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS}
ENV

echo "MODEL_ID=$MODEL_ID TP=$TENSOR_PARALLEL_SIZE MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS" \
     "stack=$NEURON_VLLM_STACK"
cd "$REPO"
bash launch.sh down >/dev/null 2>&1 || true
bash launch.sh up

# --- wait until every replica is up --------------------------------------
S=$(date +%s)
until docker logs "$CONTAINER" 2>&1 | grep -q "All server instances started successfully"; do
  if ! docker ps -q -f name="$CONTAINER" | grep -q .; then
    echo "server container exited:" >&2; docker logs --tail 40 "$CONTAINER" >&2; exit 1
  fi
  if docker logs "$CONTAINER" 2>&1 | grep -q "Failed to start server instance"; then
    echo "a replica failed to start:" >&2; docker logs --tail 60 "$CONTAINER" >&2; exit 1
  fi
  sleep 15
done
docker logs "$CONTAINER" 2>&1 | grep -E "is ready after" || true
echo "all replicas up after $(( $(date +%s) - S ))s; nginx on http://localhost:8080"

if [[ "$ACTION" == "precompile" ]]; then
  bash launch.sh down
  echo "compile cache warm in $cache_real"
fi
