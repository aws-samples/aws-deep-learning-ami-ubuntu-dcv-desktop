#!/usr/bin/env bash
# Unattended pre-staging for a Neuron desktop session.
#
# Runs ON the desktop, detached, triggered over SSM right after the
# CloudFormation stack completes. Its job is to get through everything slow
# BEFORE a human sits down -- packages, a large checkpoint download, a cold
# compile -- so the session starts with a warm, validated server.
#
# Phases: wait -> clone -> cache-restore -> image -> checkpoint
#         -> compile+serve (launch.sh) -> cache-backup -> validate
#
# Progress is written to $STATUS so a remote poller can report phase without
# streaming the whole log. Everything durable goes to EFS and S3 as it is
# produced, not at the end -- the root volume does not outlive the instance.
#
# UNVERIFIED end-to-end. Individual steps were run on trn2 separately; this
# orchestration as a whole has not been run since the move to launch.sh.
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop.git}"
BRANCH="${BRANCH:-main}"
# The durable lab store bucket. Resolved after the clone from the store stack
# (labstore/aws-defaults.sh) unless set explicitly; a wrong or empty bucket
# makes the run re-download the whole checkpoint from Hugging Face.
S3_BUCKET="${S3_BUCKET:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
BF16_REVISION="${BF16_REVISION:-main}"
DLC="${DLC:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04}"
IMAGE_TAG="${IMAGE_TAG:-openai_server-vllm:neuron}"
# Serving configuration passed to bench/serve_neuron.sh; override any of them.
# Defaults: Qwen3-32B on trn2 with room for an SLO sweep (up to 16 concurrent
# requests per replica, prompts up to 8k tokens).
SERVE_TP="${SERVE_TP:-8}"
SERVE_MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-10240}"
SERVE_MAX_NUM_SEQS="${SERVE_MAX_NUM_SEQS:-16}"
SERVE_BUCKET_PROFILE="${SERVE_BUCKET_PROFILE:-ladder}"

HOME_DIR="${HOME_DIR:-/home/ubuntu}"
EFS="${EFS:-$HOME_DIR/efs}"
WORK="$EFS/trn2-qwen3"
CACHE_DIR="$WORK/compile-cache"
MODEL_DIR="$WORK/models/$(basename "$MODEL_ID")"
REPO_DIR="$HOME_DIR/aws-deep-learning-ami-ubuntu-dcv-desktop"
STATUS="$HOME_DIR/bootstrap-status"
LOG="$HOME_DIR/bootstrap.log"

exec >>"$LOG" 2>&1

phase() { echo "PHASE=$1 AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) ${2:-}" | tee -a "$STATUS"; }
fail()  { echo "PHASE=FAILED AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) WHERE=$1" | tee -a "$STATUS"; exit 1; }

echo "=============================================================="
echo "bootstrap start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="
: >"$STATUS"

# --- 1. wait for the instance to actually be usable ----------------------
# cloud-init installs the Neuron driver via DKMS and reboots; /dev/neuron*
# does not exist until after that. Docker may also still be installing.
phase WAIT "waiting for neuron devices + docker"
for i in $(seq 1 90); do
  NODES=$(ls -1 /dev/neuron* 2>/dev/null | wc -l | tr -d ' ')
  DOCKER_OK=$(docker info >/dev/null 2>&1 && echo yes || echo no)
  if [[ "$NODES" -gt 0 && "$DOCKER_OK" == yes ]]; then
    phase WAIT_OK "devices=$NODES docker=ready after ${i}0s"
    break
  fi
  sleep 10
done
NODES=$(ls -1 /dev/neuron* 2>/dev/null | wc -l | tr -d ' ')
[[ "$NODES" -gt 0 ]] || fail "no neuron devices after 15min"
docker info >/dev/null 2>&1 || fail "docker not ready after 15min"

/opt/aws/neuron/bin/neuron-ls 2>&1 | head -8 || true
mkdir -p "$WORK" "$CACHE_DIR" "$MODEL_DIR" || fail "cannot write to EFS at $WORK"

# --- 2. clone the branch -------------------------------------------------
phase CLONE
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" \
    && git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
else
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR" || fail clone
fi
GAIT="$REPO_DIR/gen-ai-inference-testing"
BS="$GAIT/labstore/bootstrap"
[[ -d "$BS" ]] || fail "labstore/bootstrap missing after clone"
# shellcheck source=../aws-defaults.sh
. "$GAIT/labstore/aws-defaults.sh"
S3_BUCKET="${S3_BUCKET:-$(lab_bucket)}"
[[ -n "$S3_BUCKET" ]] || fail "cannot resolve the lab store bucket; set S3_BUCKET"
chmod +x "$BS"/*.sh "$GAIT"/bench/*.sh 2>/dev/null || true

# --- 3. durable store: NVMe, then restore cache and weights --------------
# Delegated to labstore (gen-ai-inference-testing/labstore), which keys the
# compile cache by accelerator class/model + stack hash + topology, so a
# restore is only reported warm when the fingerprint inputs actually match.
# If labstore is unavailable this falls back to the old flat S3 prefix, so
# the bootstrap still works standalone.
LABSTORE="$REPO_DIR/gen-ai-inference-testing/labstore/labstore.sh"
export LAB_EFS_ROOT="${LAB_EFS_ROOT:-$EFS/lab}"
export LAB_S3_BUCKET="${LAB_S3_BUCKET:-$S3_BUCKET}"
export MODEL_ID MODEL_REVISION="$BF16_REVISION"
export TOPOLOGY="${TOPOLOGY:-tp8-lnc2}"

if [[ -x "$LABSTORE" ]]; then
  # 7.6 TB of instance-store NVMe on trn2.48xlarge is unused by default;
  # weights belong there, not on EFS or the 1 TB root volume.
  phase NVME
  sudo -E "$LABSTORE" nvme-setup 2>&1 | tail -3 || phase NVME_WARN "continuing on EBS"
  if mountpoint -q /mnt/nvme 2>/dev/null; then
    export LAB_LOCAL_ROOT=/mnt/nvme/lab
    phase NVME_OK "$(df -h /mnt/nvme | tail -1 | awk '{print $2" usable"}')"
  else
    export LAB_LOCAL_ROOT="$HOME_DIR/lab-local"
    phase NVME_SKIP "no instance store; using $LAB_LOCAL_ROOT"
  fi

  phase STORE_INIT
  "$LABSTORE" init 2>&1 | tail -4

  phase STORE_RESUME "restoring cache + hydrating weights"
  RESUME_OUT=$("$LABSTORE" resume 2>&1) || true
  echo "$RESUME_OUT"
  # resume emits the exports it wants us to use; adopt them.
  eval "$(printf '%s\n' "$RESUME_OUT" | grep -E '^export (VLLM_CACHE_ROOT|MODEL_PATH)=')" || true
  CACHE_DIR="${VLLM_CACHE_ROOT:-$CACHE_DIR}"
  MODEL_DIR="${MODEL_PATH:-$MODEL_DIR}"
  phase STORE_RESUME_OK "cache=$CACHE_DIR model=$MODEL_DIR"
else
  phase CACHE_RESTORE "labstore unavailable; flat S3 fallback"
  if aws s3 ls "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" >/dev/null 2>&1; then
    aws s3 sync "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" "$CACHE_DIR/" --only-show-errors \
      && phase CACHE_RESTORE_OK "files=$(find "$CACHE_DIR" -type f | wc -l | tr -d ' ')"
  else
    phase CACHE_RESTORE_SKIP "no prior cache in s3"
  fi
fi
mkdir -p "$CACHE_DIR" "$MODEL_DIR"

# --- 4. container image -------------------------------------------------
# Built through the repo's standard build script, the same one launch.sh runs
# (it refuses to build on anything but trn2/trn3). The old prebuilt ECR image
# (:qwen3-trn2-validated) predates the launch.sh layout -- no resources/serve,
# no nginx -- so it cannot serve this path and is not used.
phase IMAGE_BUILD "scripts/openai_server-vllm-neuron.sh (stack chosen by instance type)"
docker pull "$DLC" >/dev/null || fail "docker pull $DLC"
bash "$GAIT/scripts/openai_server-vllm-neuron.sh" \
  || fail "image build (scripts/openai_server-vllm-neuron.sh)"
# Confirm the plugin reports the neuron platform. Needs devices, hence
# --privileged, and it is cheap insurance before a long compile.
PLAT=$(docker run --rm --privileged -v /dev:/dev --entrypoint python "$IMAGE_TAG" \
  -c "import vllm;from vllm.platforms import current_platform;print(current_platform.device_name)" 2>&1 | tail -1)
phase IMAGE_OK "platform=$PLAT"
[[ "$PLAT" == "neuron" ]] || fail "platform probe returned '$PLAT', expected 'neuron'"

# --- 5. checkpoint ------------------------------------------------------
# Onto EFS so it outlives the instance and a later run skips the download.
phase CHECKPOINT "downloading $MODEL_ID -> $MODEL_DIR"
if [[ -f "$MODEL_DIR/config.json" ]]; then
  phase CHECKPOINT_SKIP "already present ($(du -sh "$MODEL_DIR" | cut -f1))"
else
  MODEL_ID="$MODEL_ID" BF16_REVISION="$BF16_REVISION" DEST="$MODEL_DIR" \
    bash "$BS/download_model.sh" || fail "checkpoint download"
  phase CHECKPOINT_OK "$(du -sh "$MODEL_DIR" | cut -f1)"
fi

# Seed S3 with the checkpoint so every future session skips the Hugging Face
# download entirely -- `labstore resume` then hydrates from S3 in-region,
# which is far faster and cannot be rate-limited by a third party. This runs
# before anyone starts testing.
if [[ -x "$LABSTORE" ]]; then
  phase WEIGHTS_PUSH "seeding S3 for future sessions"
  if WEIGHTS_SRC="$MODEL_DIR" "$LABSTORE" push-weights 2>&1 | tail -3; then
    phase WEIGHTS_PUSH_OK
  else
    phase WEIGHTS_PUSH_WARN "push failed; weights still on local/EFS"
  fi
fi

# --- 6. compile + serve through launch.sh -------------------------------
# launch.sh mounts $HOME/snapshots and $HOME/cache. The server container runs
# as root (compose/neuron.yaml), so the compile cache is kept LOCAL and only
# copied to the store as ubuntu -- root-owned entries on the shared EFS block
# ubuntu (PLAN.md 2b). The store copy seeds the local cache when present.
LOCAL_CACHE="${LAB_LOCAL_ROOT:-$HOME_DIR/lab-local}/cache/neuron"
mkdir -p "$LOCAL_CACHE"
if [[ -d "$CACHE_DIR" && -n "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]]; then
  cp -a "$CACHE_DIR/." "$LOCAL_CACHE/" 2>/dev/null \
    && phase CACHE_SEED_OK "files=$(find "$LOCAL_CACHE" -type f | wc -l | tr -d ' ')" \
    || phase CACHE_SEED_WARN "partial copy from $CACHE_DIR"
fi
if [[ ! -e "$HOME_DIR/cache" ]]; then
  ln -s "$LOCAL_CACHE" "$HOME_DIR/cache"
elif [[ "$(realpath "$HOME_DIR/cache")" != "$(realpath "$LOCAL_CACHE")" ]]; then
  phase CACHE_LINK_WARN "\$HOME/cache already exists ($(realpath "$HOME_DIR/cache")); using it"
fi

phase PRECOMPILE "launch.sh up (first start compiles; replicas start one by one)"
env -u MODEL_ID MODEL_DIR="$MODEL_DIR" TENSOR_PARALLEL_SIZE="$SERVE_TP" \
  MAX_MODEL_LEN="$SERVE_MAX_MODEL_LEN" MAX_NUM_SEQS="$SERVE_MAX_NUM_SEQS" BUCKET_PROFILE="$SERVE_BUCKET_PROFILE" \
  PREFIX_CACHING="${PREFIX_CACHING:-false}" NUM_REPLICAS="${NUM_REPLICAS:-}" HOME="$HOME_DIR" \
  bash "$GAIT/bench/serve_neuron.sh" up || fail "bench/serve_neuron.sh up"
phase PRECOMPILE_OK "cache files=$(find "$(realpath "$HOME_DIR/cache")" -type f | wc -l | tr -d ' ')"
curl -sf http://localhost:8080/health >/dev/null 2>&1 && phase SERVE_OK "nginx :8080 healthy" \
  || fail "server never became healthy on :8080"

# --- 7. back the cache up immediately -----------------------------------
# labstore also writes a MANIFEST.json recording the exact versions and
# topology, without which the next run cannot tell whether the cache is
# reusable. Runs as ubuntu, so the store copies are ubuntu-owned.
phase CACHE_BACKUP
if [[ -x "$LABSTORE" ]]; then
  VLLM_CACHE_ROOT="$(realpath "$HOME_DIR/cache")" "$LABSTORE" save 2>&1 | tail -5 \
    && phase CACHE_BACKUP_OK "manifest written, synced to S3" \
    || phase CACHE_BACKUP_WARN "labstore save failed; cache still local at $LOCAL_CACHE"
else
  aws s3 sync "$(realpath "$HOME_DIR/cache")/" "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" --only-show-errors \
    && phase CACHE_BACKUP_OK || phase CACHE_BACKUP_WARN "s3 sync failed, cache still local"
fi

# --- 8. correctness -----------------------------------------------------
# A fast wrong answer is worthless; prove it before anyone benchmarks it.
phase VALIDATE
SERVED=$(curl -s http://localhost:8080/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
ANS=$(curl -s http://localhost:8080/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":20,\"temperature\":0}" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
echo "answer: $ANS"
if echo "$ANS" | grep -qi paris; then
  phase VALIDATE_OK "answer mentions Paris"
else
  phase VALIDATE_WARN "unexpected answer: $ANS"
fi

phase READY "server up on :8080 (nginx over all replicas), cache backed up"
echo "=============================================================="
echo "bootstrap complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  model      : $MODEL_DIR (served as $SERVED)"
echo "  cache      : $(realpath "$HOME_DIR/cache") (store copy via labstore save)"
echo "  next       : cd $GAIT && bash bench/client.sh vllm <run-name>"
echo "=============================================================="
