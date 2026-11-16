#!/usr/bin/env bash
# Download a Qwen3 checkpoint to local disk before serving (default:
# $HOME/snapshots, the directory launch.sh mounts).
#
# Pin BF16_REVISION for reproducibility -- benchmark numbers are not
# comparable across checkpoint revisions.
# Qwen/Qwen3-32B is not gated: no HF_TOKEN is needed.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
BF16_REVISION="${BF16_REVISION:-main}"
DEST="${DEST:-$HOME/snapshots/$(basename "$MODEL_ID")}"

mkdir -p "$DEST"
echo "=== downloading $MODEL_ID @ $BF16_REVISION -> $DEST ==="
[[ -n "${HF_TOKEN:-}" ]] && echo "HF_TOKEN is set" || echo "HF_TOKEN not set (fine if the repo is public)"

hf download "$MODEL_ID" --revision "$BF16_REVISION" --local-dir "$DEST"

echo
echo "=== checkpoint manifest ==="
# Hash the files so the exact checkpoint can be verified across environments
# (the AWS tutorial recommends this) and attached to benchmark results.
find "$DEST" -type f \( -name '*.safetensors' -o -name '*.json' \) -print0 \
  | sort -z | xargs -0 shasum -a 256 | tee "$DEST/../$(basename "$DEST").sha256"

du -sh "$DEST"
echo "done. Serve with: MODEL_DIR=$DEST bash bench/serve_neuron.sh up"
