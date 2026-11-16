#!/usr/bin/env bash
# labstore -- durable session store for ephemeral accelerator instances.
#
# Instances can be short-lived; when one is terminated its root volume goes
# with it. This moves the slow-to-recreate artifacts somewhere that outlives the
# host, and restores them onto a fresh one in minutes.
#
# Three tiers, because putting an artifact in the wrong one is the main way
# this gets slower instead of faster:
#   S3    durable source of truth        weights, cache backups, results
#   EFS   warm cache, instantly mounted  compile caches, results, manifests
#   local hot working set (NVMe)         weights during a session
#
# Hardware-agnostic. Compile caches are keyed by accelerator class/model and a
# stack hash (see provenance.py), so a Trainium NEFF cannot collide with a CUDA
# TensorRT engine. Weights and results are keyed without accelerator fields, so
# a GPU box and a Trainium box share one weight cache.
#
# Non-disruptive: nothing here is invoked unless you invoke it. It writes only
# under $LAB_EFS_ROOT, $LAB_LOCAL_ROOT and its S3 prefix.
#
# Usage:
#   labstore.sh nvme-setup                 # RAID0 + mount local NVMe (idempotent)
#   labstore.sh init                        # create the store layout
#   labstore.sh status                      # what is warm vs cold
#   labstore.sh push-weights                # local -> S3 (run on a cheap box)
#   labstore.sh resume                      # restore cache + hydrate weights
#   labstore.sh save                         # cache + results -> EFS and S3
#
# Env:
#   MODEL_ID        default Qwen/Qwen3-32B
#   MODEL_REVISION  default main
#   TOPOLOGY        default tp8-lnc2 (neuron) / tp8 (cuda)
#   LAB_EFS_ROOT    default $HOME/efs/lab
#   LAB_LOCAL_ROOT  default /mnt/nvme/lab, else $HOME/lab-local
#   LAB_S3_BUCKET   default resolved at runtime (aws-defaults.sh): the store
#                   stack's BucketName export, else dl-lab-store-<account>-<region>
#   AWS_REGION      default the CLI's configured region, else instance metadata
set -uo pipefail

# HOME is not guaranteed. cloud-init UserData runs with a minimal environment,
# and under `set -u` an unset HOME is fatal -- which is exactly how this failed
# on first contact with a real instance ("line 42: HOME: unbound variable").
# Resolve it from passwd rather than assuming /root, so the fallback is still
# correct when invoked as a non-root user.
if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
  [[ -n "$HOME" ]] || HOME=/tmp
  export HOME
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
TOPOLOGY="${TOPOLOGY:-}"
LAB_EFS_ROOT="${LAB_EFS_ROOT:-$HOME/efs/lab}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=aws-defaults.sh
. "$HERE/aws-defaults.sh"
AWS_REGION="$(lab_region)"
LAB_S3_BUCKET="$(lab_bucket)"
[[ -n "$LAB_S3_BUCKET" ]] || { echo "labstore: cannot resolve the S3 bucket; set LAB_S3_BUCKET" >&2; exit 1; }
LAB_S3_PREFIX="${LAB_S3_PREFIX:-lab}"
PROV="$HERE/provenance.py"
PY=(python3 -I)

# Prefer local NVMe; fall back to home so this works on instance types with no
# instance store (inf2.xlarge, m5, most GPU dev boxes).
if [[ -z "${LAB_LOCAL_ROOT:-}" ]]; then
  if mountpoint -q /mnt/nvme 2>/dev/null; then LAB_LOCAL_ROOT=/mnt/nvme/lab
  else LAB_LOCAL_ROOT="$HOME/lab-local"; fi
fi

say()  { printf '%s\n' "$*"; }
step() { printf '\n=== %s ===\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "$PROV" ]] || die "provenance.py not found beside this script"
command -v aws >/dev/null || die "aws cli not found"

prov() { MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" TOPOLOGY="$TOPOLOGY" \
         "${PY[@]}" "$PROV" --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
         ${TOPOLOGY:+--topology "$TOPOLOGY"} "$@"; }

CACHE_KEY=""; WEIGHTS_KEY=""; STACK_HASH=""
resolve_keys() {
  CACHE_KEY=$(prov --key cache)   || die "provenance failed"
  WEIGHTS_KEY=$(prov --key weights) || die "provenance failed"
  STACK_HASH=$(prov --key hash)   || die "provenance failed"
}

s3_uri() { printf 's3://%s/%s/%s' "$LAB_S3_BUCKET" "$LAB_S3_PREFIX" "$1"; }

# ---------------------------------------------------------------------------
cmd_nvme_setup() {
  step "local NVMe setup"
  # trn2.48xlarge exposes 4 x 1,900 GB instance-store SSDs. Nothing in the
  # project sets these up, so by default the 7.6 TB sits unused while the
  # 1 TB root EBS volume does all the work. Instance store is ephemeral --
  # that is fine, it is only ever the working set.
  if mountpoint -q /mnt/nvme 2>/dev/null; then
    say "already mounted:"; df -h /mnt/nvme | tail -1; return 0
  fi
  # SAFETY: select ONLY disks whose device model reports EC2 instance storage.
  #
  # An earlier version selected any unmounted whole disk. That is dangerous:
  # a blank, unmounted secondary EBS volume looks identical to instance store
  # by that test, and this function runs mdadm --create + mkfs.xfs under sudo.
  # A simulated topology with one spare EBS volume confirmed the old logic
  # would have reformatted it. Model matching eliminates the whole class:
  # instance store reports "Amazon EC2 NVMe Instance Storage", EBS reports
  # "Amazon Elastic Block Store".
  #
  # If no instance-store disk is found we do nothing and fall back to $HOME --
  # never guess. LABSTORE_NVME_DEVICES can override explicitly.
  DISKS=()
  if [[ -n "${LABSTORE_NVME_DEVICES:-}" ]]; then
    read -r -a DISKS <<<"$LABSTORE_NVME_DEVICES"
    say "using explicitly supplied devices: ${DISKS[*]}"
  else
    while IFS= read -r d; do
      [[ -n "$d" ]] && DISKS+=("$d")
    done < <(lsblk -dn -o NAME,MODEL 2>/dev/null \
      | grep -i 'Instance Storage' \
      | awk '{print "/dev/"$1}' \
      | while read -r dev; do
          # Belt and braces: skip anything with a mountpoint on it or any
          # of its partitions.
          if ! lsblk -n -o MOUNTPOINT "$dev" 2>/dev/null | grep -q .; then echo "$dev"; fi
        done)
  fi

  if (( ${#DISKS[@]} == 0 )); then
    say "no EC2 instance-store disks found -- falling back to \$HOME"
    say "  (expected on m5/inf2/most GPU types; trn2.48xlarge should report 4)"
    return 0
  fi
  say "found ${#DISKS[@]} instance-store disk(s): ${DISKS[*]}"

  # Refuse outright if anything selected is not instance store, in case an
  # override was passed carelessly.
  for d in "${DISKS[@]}"; do
    if lsblk -n -o MOUNTPOINT "$d" 2>/dev/null | grep -q .; then
      die "$d has a mountpoint -- refusing to reformat it"
    fi
    model=$(lsblk -dn -o MODEL "$d" 2>/dev/null || true)
    if [[ -n "$model" ]] && ! printf '%s' "$model" | grep -qi 'Instance Storage'; then
      die "$d model is '$model', not instance storage -- refusing to reformat it"
    fi
  done

  [[ $EUID -eq 0 ]] || die "nvme-setup needs root: sudo -E $0 nvme-setup"

  if (( ${#DISKS[@]} == 1 )); then
    TARGET="${DISKS[0]}"
    mkfs.xfs -f "$TARGET" >/dev/null || die "mkfs failed on $TARGET"
  else
    command -v mdadm >/dev/null || { apt-get update -qq && apt-get install -y -qq mdadm; }
    # RAID0: we want aggregate throughput, and redundancy is pointless on
    # ephemeral disks holding a re-downloadable working set.
    mdadm --create /dev/md0 --level=0 --raid-devices="${#DISKS[@]}" "${DISKS[@]}" \
      --run --force >/dev/null 2>&1 || die "mdadm create failed"
    TARGET=/dev/md0
    mkfs.xfs -f "$TARGET" >/dev/null || die "mkfs failed on $TARGET"
  fi
  mkdir -p /mnt/nvme
  mount -o noatime,nodiratime "$TARGET" /mnt/nvme || die "mount failed"
  # Deliberately NOT added to /etc/fstab: instance-store disks are blank after
  # a stop/start, and a stale fstab entry can block boot.
  chown -R "${SUDO_USER:-ubuntu}:${SUDO_USER:-ubuntu}" /mnt/nvme
  say "mounted:"; df -h /mnt/nvme | tail -1
}

# ---------------------------------------------------------------------------
cmd_init() {
  step "initialising store"
  [[ -d "$(dirname "$LAB_EFS_ROOT")" ]] || die "EFS parent $(dirname "$LAB_EFS_ROOT") missing -- is EFS mounted?"
  mkdir -p "$LAB_EFS_ROOT"/{cache,results,manifests,wheelhouse,state} \
           "$LAB_LOCAL_ROOT"/{models,cache,results} || die "mkdir failed"
  say "EFS   : $LAB_EFS_ROOT"
  say "local : $LAB_LOCAL_ROOT"
  say "S3    : $(s3_uri '')"
  if aws s3api head-bucket --bucket "$LAB_S3_BUCKET" --region "$AWS_REGION" >/dev/null 2>&1; then
    say "bucket reachable"
  else
    say "WARN: cannot reach s3://$LAB_S3_BUCKET"
  fi
  resolve_keys
  prov --json > "$LAB_EFS_ROOT/manifests/live-$(date -u +%Y%m%dT%H%M%SZ).json" 2>/dev/null \
    && say "wrote live provenance snapshot"
}

# ---------------------------------------------------------------------------
cmd_status() {
  resolve_keys
  step "provenance"
  prov
  step "store status"
  local ec="$LAB_EFS_ROOT/$CACHE_KEY" lw="$LAB_LOCAL_ROOT/$WEIGHTS_KEY"
  printf '%-34s %s\n' "EFS compile cache" \
    "$([[ -d $ec ]] && echo "WARM ($(find "$ec" -type f 2>/dev/null | wc -l | tr -d ' ') files)" || echo COLD)"
  printf '%-34s %s\n' "local weights" \
    "$([[ -f $lw/config.json ]] && echo "PRESENT ($(du -sh "$lw" 2>/dev/null | cut -f1))" || echo ABSENT)"
  printf '%-34s %s\n' "S3 weights" \
    "$(aws s3 ls "$(s3_uri "$WEIGHTS_KEY")/config.json" --region "$AWS_REGION" >/dev/null 2>&1 && echo PRESENT || echo ABSENT)"
  printf '%-34s %s\n' "S3 compile cache" \
    "$(aws s3 ls "$(s3_uri "$CACHE_KEY")/" --region "$AWS_REGION" 2>/dev/null | head -1 | grep -q . && echo PRESENT || echo ABSENT)"

  # Explain a near-miss: a cache for this model+accel but a different stack
  # hash is the single most confusing failure mode, so name it explicitly.
  local family="$LAB_EFS_ROOT/cache/$(prov --key cache | cut -d/ -f2-3)"
  if [[ ! -d "$ec" && -d "$family" ]]; then
    step "near-miss analysis"
    say "no cache for stack hash $STACK_HASH, but other hashes exist for this model+accelerator:"
    # Manifests live at <hash>/<topology>/MANIFEST.json -- topology is the last
    # component of the cache key, so search rather than assuming one level.
    local m
    while IFS= read -r m; do
      [[ -n "$m" ]] || continue
      say "  $(printf '%s' "${m#"$family"/}" | sed 's#/MANIFEST.json$##')"
      "${PY[@]}" -c "
import json,sys
sys.path.insert(0,'$HERE')
from provenance import collect
man=json.load(open('$m'))
p=collect('$MODEL_ID','$MODEL_REVISION','$TOPOLOGY')
rs=p.diff(man)
print('      (identical stack fields -- key should have matched)' if not rs else '')
for r in rs[:6]: print('      '+r)
"
    done < <(find "$family" -name MANIFEST.json -maxdepth 3 2>/dev/null | sort)
  fi
}

# ---------------------------------------------------------------------------
cmd_push_weights() {
  resolve_keys
  step "pushing weights to S3"
  local src="${WEIGHTS_SRC:-$LAB_LOCAL_ROOT/$WEIGHTS_KEY}"
  [[ -f "$src/config.json" ]] || die "no checkpoint at $src (set WEIGHTS_SRC)"
  say "$src -> $(s3_uri "$WEIGHTS_KEY")"
  # Weights are large and immutable; S3 is 13x cheaper than EFS at rest and
  # faster to fan out from. This is the step that removes the HuggingFace
  # download from every future accelerator session.
  aws s3 sync "$src" "$(s3_uri "$WEIGHTS_KEY")" --region "$AWS_REGION" \
    --only-show-errors --no-progress || die "s3 sync failed"
  say "done: $(du -sh "$src" | cut -f1)"
}

cmd_resume() {
  resolve_keys
  step "resuming session"
  say "stack hash : $STACK_HASH"
  mkdir -p "$LAB_LOCAL_ROOT/$WEIGHTS_KEY" "$LAB_EFS_ROOT/$CACHE_KEY"

  # 1. weights: S3 -> local. Parallelised; S3 to in-region instance is far
  #    faster than reading a large checkpoint over NFS.
  local lw="$LAB_LOCAL_ROOT/$WEIGHTS_KEY"
  if [[ -f "$lw/config.json" ]]; then
    say "weights already local ($(du -sh "$lw" | cut -f1)) -- skipping"
  elif aws s3 ls "$(s3_uri "$WEIGHTS_KEY")/config.json" --region "$AWS_REGION" >/dev/null 2>&1; then
    say "hydrating weights from S3 ..."
    aws configure set default.s3.max_concurrent_requests 32 2>/dev/null || true
    local t0=$SECONDS
    aws s3 sync "$(s3_uri "$WEIGHTS_KEY")" "$lw" --region "$AWS_REGION" \
      --only-show-errors --no-progress || die "weight hydration failed"
    say "hydrated $(du -sh "$lw" | cut -f1) in $((SECONDS-t0))s"
  else
    say "weights not in S3 -- will need a HuggingFace download (run push-weights next time)"
  fi

  # 2. compile cache: EFS first (already local-ish), else S3.
  local ec="$LAB_EFS_ROOT/$CACHE_KEY"
  if [[ -n "$(find "$ec" -type f 2>/dev/null | head -1)" ]]; then
    say "compile cache WARM on EFS: $(find "$ec" -type f | wc -l | tr -d ' ') files"
  elif aws s3 ls "$(s3_uri "$CACHE_KEY")/" --region "$AWS_REGION" 2>/dev/null | head -1 | grep -q .; then
    say "restoring compile cache from S3 ..."
    aws s3 sync "$(s3_uri "$CACHE_KEY")" "$ec" --region "$AWS_REGION" --only-show-errors --no-progress
    say "restored $(find "$ec" -type f | wc -l | tr -d ' ') files"
  else
    say "compile cache COLD -- expect a full compile (~20 min). Run 'status' for a near-miss diff."
  fi

  cat <<EOF

export VLLM_CACHE_ROOT=$ec
export MODEL_PATH=$lw
EOF
}

cmd_save() {
  resolve_keys
  step "saving session artifacts"
  local ec="$LAB_EFS_ROOT/$CACHE_KEY"
  mkdir -p "$ec"

  # Manifest first: a cache without provenance is a trap, because the next
  # run cannot tell whether it is valid.
  prov --json > "$ec/MANIFEST.json" && say "wrote $ec/MANIFEST.json"

  if [[ -n "${VLLM_CACHE_ROOT:-}" && -d "${VLLM_CACHE_ROOT:-}" && "$VLLM_CACHE_ROOT" != "$ec" ]]; then
    say "copying $VLLM_CACHE_ROOT -> $ec"
    cp -a "$VLLM_CACHE_ROOT/." "$ec/" 2>/dev/null || say "WARN: cache copy incomplete"
  fi

  local n; n=$(find "$ec" -type f 2>/dev/null | wc -l | tr -d ' ')
  if (( n > 1 )); then
    say "backing cache up to S3 ($n files)"
    aws s3 sync "$ec" "$(s3_uri "$CACHE_KEY")" --region "$AWS_REGION" --only-show-errors --no-progress \
      && say "cache backed up" || say "WARN: S3 backup failed (still on EFS)"
  else
    say "no cache files to back up"
  fi

  if [[ -n "${RESULTS_DIR:-}" && -d "${RESULTS_DIR:-}" ]]; then
    local rid; rid="$(basename "$RESULTS_DIR")"
    cp -a "$RESULTS_DIR" "$LAB_EFS_ROOT/results/$rid" 2>/dev/null
    aws s3 sync "$RESULTS_DIR" "$(s3_uri "results/$rid")" --region "$AWS_REGION" --only-show-errors --no-progress
    say "results saved: results/$rid"
  fi
  say "save complete"
}

case "${1:-}" in
  nvme-setup)    cmd_nvme_setup ;;
  init)          cmd_init ;;
  status)        cmd_status ;;
  push-weights)  cmd_push_weights ;;
  resume)        cmd_resume ;;
  save)          cmd_save ;;
  *) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
