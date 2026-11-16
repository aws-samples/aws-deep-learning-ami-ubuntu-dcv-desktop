#!/usr/bin/env bash
# Opt-in bridge between the durable lab store and the EXISTING test harness.
#
# The harness (launch.sh + compose/{cuda,neuron}.yaml) mounts $HOME/snapshots
# and $HOME/cache into its containers. Both currently sit on the root EBS
# volume, so nothing survives instance termination. This script redirects them
# into the durable store by symlink, without editing a single harness file.
#
# Deliberately opt-in and reversible:
#   link-harness.sh link      point $HOME/{snapshots,cache} at the store
#   link-harness.sh unlink    restore plain local directories
#   link-harness.sh status    show what they currently point at
#
# Works for GPU and Trainium alike: $HOME/snapshots holds the HuggingFace cache
# (hardware-independent, so a GPU box and a Trainium box share it), while
# $HOME/cache holds accelerator-specific compile output and is therefore placed
# under a per-accelerator subdirectory.
#
# Run nothing and the harness behaves exactly as it does today.
set -uo pipefail

if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
  [[ -n "$HOME" ]] || HOME=/tmp
  export HOME
fi
LAB_EFS_ROOT="${LAB_EFS_ROOT:-$HOME/efs/lab}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="$HOME/snapshots"
CACHE="$HOME/cache"

accel() {
  python3 -I -c "
import sys; sys.path.insert(0,'$HERE')
from provenance import detect_accel_class, detect_neuron, detect_cuda
c = detect_accel_class()
m = {'neuron': lambda: detect_neuron().get('accel_model','unknown'),
     'cuda':   lambda: detect_cuda().get('accel_model','unknown'),
     'cpu':    lambda: 'none'}[c]()
print(f'{c}-{m}')
" 2>/dev/null || echo "unknown-unknown"
}

# Move an existing real directory aside rather than deleting it -- it may hold
# a checkpoint someone just spent 40 minutes downloading.
relocate() {
  local path="$1" target="$2"
  mkdir -p "$target"
  if [[ -L "$path" ]]; then
    echo "  $path is already a symlink -> $(readlink "$path")"
    rm -f "$path"
  elif [[ -d "$path" ]]; then
    if [[ -n "$(ls -A "$path" 2>/dev/null)" ]]; then
      echo "  migrating existing contents of $path into the store (not deleting)"
      cp -an "$path/." "$target/" 2>/dev/null || true
      mv "$path" "${path}.pre-labstore.$(date -u +%Y%m%d%H%M%S)"
      echo "  original preserved at ${path}.pre-labstore.*"
    else
      rmdir "$path"
    fi
  fi
  ln -s "$target" "$path"
  echo "  $path -> $target"
}

case "${1:-status}" in
  link)
    [[ -d "$(dirname "$LAB_EFS_ROOT")" ]] || { echo "ERROR: $(dirname "$LAB_EFS_ROOT") missing -- is EFS mounted?" >&2; exit 1; }
    A="$(accel)"
    echo "accelerator: $A"
    echo "linking harness paths into $LAB_EFS_ROOT"
    # Weights/HF cache: shared across all hardware.
    relocate "$SNAP"  "$LAB_EFS_ROOT/harness/snapshots"
    # Compile output: per-accelerator, because a NEFF and a TRT engine must
    # never be handed to each other.
    relocate "$CACHE" "$LAB_EFS_ROOT/harness/cache/$A"
    echo
    echo "Done. launch.sh and compose/*.yaml are unmodified; they now resolve"
    echo "through these symlinks. NOTE: EFS is slower than local disk for large"
    echo "sequential reads -- for heavy weight loading prefer 'labstore resume',"
    echo "which hydrates to local NVMe instead."
    ;;
  unlink)
    for p in "$SNAP" "$CACHE"; do
      if [[ -L "$p" ]]; then rm -f "$p"; mkdir -p "$p"; echo "  $p restored as a local directory"
      else echo "  $p is not a symlink; left alone"; fi
    done
    ;;
  status)
    for p in "$SNAP" "$CACHE"; do
      if [[ -L "$p" ]]; then echo "  $p -> $(readlink "$p")  [store]"
      elif [[ -d "$p" ]]; then echo "  $p  [local dir, not persisted]"
      else echo "  $p  [absent]"; fi
    done
    ;;
  *) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
