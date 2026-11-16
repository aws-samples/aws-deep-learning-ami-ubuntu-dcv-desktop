#!/usr/bin/env bash
# Run a benchmark client from the server image, as your own uid, inside the
# server container's network namespace -- so it reaches nginx on :8080 AND
# every replica on 8000+128*i, which compose does not publish.
#
#   bench/client.sh vllm   <run-name> [extra docker -e args...]   # run_vllm_bench.sh
#   bench/client.sh locust <run-name> <in> <out> <users> <num_requests>
#
# TARGET=nginx (default) or TARGET=<i> for replica i only.
# Results: $RESULTS_ROOT/<run-name> (default /tmp/results), mirrored after
# every cell to $MIRROR_ROOT/<run-name> (default $HOME/efs/lab/results) when
# that directory exists. The harness is snapshotted per run: bash reads
# scripts incrementally, so editing a mounted script mid-run changes what runs.
set -uo pipefail
KIND=$1; NAME=$2; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-compose-server_0-1}"
# Client image: by default the running server's own image, which has vLLM (for
# `vllm bench serve`) -- and, for the Neuron native stack, Locust and matplotlib.
IMAGE="${IMAGE:-$(docker inspect -f '{{.Config.Image}}' "$CONTAINER" 2>/dev/null)}"
[[ -n "$IMAGE" ]] || { echo "no server container '$CONTAINER'; start one or set IMAGE" >&2; exit 1; }
TARGET="${TARGET:-nginx}"
RESULTS_ROOT="${RESULTS_ROOT:-/tmp/results}"
MIRROR_ROOT="${MIRROR_ROOT:-$HOME/efs/lab/results}"
if [[ "$TARGET" == "nginx" ]]; then PORT=8080; else PORT=$((8000 + 128 * TARGET)); fi
URL="http://localhost:$PORT"
# Counter snapshots must cover exactly the servers this run loads.
NSERV=$(docker exec "$CONTAINER" grep -c "server localhost:" /opt/program/nginx.conf 2>/dev/null || echo 1)
if [[ "$TARGET" == "nginx" ]]; then
  METRICS_URLS=$(for i in $(seq 0 $(( ${NSERV:-1} - 1 ))); do printf "http://localhost:%d " $((8000 + 128 * i)); done)
else
  METRICS_URLS="$URL"
fi

D="$RESULTS_ROOT/$NAME"; mkdir -p "$D"
MIRROR_ARGS=()
if [[ -d "$MIRROR_ROOT" ]]; then
  mkdir -p "$MIRROR_ROOT/$NAME"
  MIRROR_ARGS=(-v "$MIRROR_ROOT/$NAME:/mirror" -e MIRROR_DIR=/mirror)
fi
rm -rf "$D/.bench" && cp -r "$HERE" "$D/.bench"
( cd "$HERE" && git rev-parse HEAD && git status --porcelain . ) > "$D/harness-rev.txt" 2>/dev/null

# Served model name = MODEL_ID; a /snapshots/... path doubles as the tokenizer.
MODEL=$(docker exec "$CONTAINER" curl -sf "$URL/v1/models" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])') \
  || { echo "no server answering at $URL in $CONTAINER" >&2; exit 1; }

# Record the run's context from the host side; parse_results.py reads it, so no
# model, dtype, TP or instance is hardcoded in the analysis.
CFG=$(docker exec "$CONTAINER" cat /tmp/config.yaml 2>/dev/null || true)
STACK=$(docker exec "$CONTAINER" cat /opt/program/neuron-stack 2>/dev/null || true)
CACHE_INFO=$(docker exec "$CONTAINER" curl -sf "$URL/metrics" 2>/dev/null | grep '^vllm:cache_config_info' | head -1 || true)
if command -v neuron-ls >/dev/null 2>&1; then
  CORES=$(neuron-ls --json-output 2>/dev/null | python3 -c 'import json,sys;d=json.load(sys.stdin);print(sum(x.get("nc_count",0) for x in d))' 2>/dev/null)
else
  CORES=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
CFG="$CFG" STACK="$STACK" CACHE_INFO="$CACHE_INFO" CORES="${CORES:-0}" NSERV="$NSERV" MODEL="$MODEL" \
INSTANCE="$(cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null)" python3 - > "$D/server-context.json" <<'PY'
import json, os, re
cfg = {}
for line in os.environ["CFG"].splitlines():
    m = re.match(r"^([a-z][a-z0-9-]*):\s*(\S.*)$", line)
    if m:
        cfg[m.group(1)] = m.group(2).strip()
tp = int(cfg["tensor-parallel-size"]) if cfg.get("tensor-parallel-size", "").isdigit() else None
cap = re.search(r'kv_cache_max_concurrency="([0-9.]+)"', os.environ["CACHE_INFO"])
cores = int(os.environ["CORES"] or 0)
print(json.dumps({
    "model": os.environ["MODEL"],
    "dtype": cfg.get("dtype"),
    "tp": tp,
    "max_model_len": cfg.get("max-model-len"),
    "max_num_seqs": cfg.get("max-num-seqs"),
    "instance_type": os.environ["INSTANCE"] or None,
    "neuron_stack": os.environ["STACK"] or None,
    "servers_running": int(os.environ["NSERV"]),
    "replica_slots": cores // tp if tp and cores else None,
    "kv_admission_cap": int(float(cap.group(1))) if cap else None,
}, indent=2))
PY

COMMON=(--rm --network "container:$CONTAINER" --user "$(id -u):$(id -g)" -e HOME=/tmp -w /tmp
        -v "$HOME/snapshots:/snapshots:ro" -v "$D/.bench:/bench:ro" -v "$D:/results"
        -e MODEL="$MODEL" -e TOKENIZER="$MODEL" "${MIRROR_ARGS[@]}")
case "$KIND" in
  vllm)
    docker run "${COMMON[@]}" --name "bench-$(echo "$NAME" | tr -c 'a-zA-Z0-9\n' '-' | tail -c 40)" \
      -e BASE_URL="$URL" -e METRICS_URLS="$METRICS_URLS" -e RESULTS_DIR=/results "$@" \
      --entrypoint bash "$IMAGE" /bench/run_vllm_bench.sh > "$D/client.log" 2>&1
    rc=$? ;;
  locust)
    IN=$1; OUT=$2; U=$3; NREQ=$4; F="locust_in${IN}_out${OUT}_c${U}"
    docker run "${COMMON[@]}" --name "locust-$U-$TARGET" \
      -e INPUT_TOKENS="$IN" -e OUTPUT_TOKENS="$OUT" -e NUM_REQUESTS="$NREQ" \
      -e LOCUST_SUMMARY_JSON="/results/$F.json" --entrypoint locust "$IMAGE" \
      -f /bench/locustfile.py -H "$URL" --headless -u "$U" -r "$U" -t 60m --only-summary > "$D/$F.log" 2>&1
    rc=$?
    [[ -d "$MIRROR_ROOT/$NAME" ]] && cp -p "$D/$F.json" "$D/$F.log" "$MIRROR_ROOT/$NAME/" 2>/dev/null ;;
  *) echo "usage: client.sh vllm|locust <run-name> ..." >&2; exit 2 ;;
esac
[[ -d "$MIRROR_ROOT/$NAME" ]] && cp -p "$D"/*.log "$D/harness-rev.txt" "$D/server-context.json" "$MIRROR_ROOT/$NAME/" 2>/dev/null
echo "exit=$rc results=$D"
exit $rc
