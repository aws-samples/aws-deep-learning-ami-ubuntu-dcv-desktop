#!/usr/bin/env bash
# SLO sweep using `vllm bench serve` -- the source-of-truth harness.
#
# Emits one JSON per (token-shape, concurrency) cell into $RESULTS_DIR, which
# bench/parse_results.py folds into an SLO table.
#
# --ignore-eos is NOT optional. Without it, output length becomes a function
# of the model's stopping behaviour, so ITL and throughput stop being
# comparable across cells.
#
# Usage: normally via bench/client.sh, which runs this in a client container
# inside the server's network namespace:
#   bench/client.sh vllm <run-name> -e SHAPES="4096:256" -e CONCURRENCIES="1 2 4"
#   TARGET=1 bench/client.sh vllm <run-name> ...     # replica 1 only
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"   # nginx; replica i is 8000+128*i
# Served model name = MODEL_ID (launch.sh path); ask the server when unset.
MODEL="${MODEL:-$(curl -sf "${BASE_URL}/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)}"
# The served name (MODEL_ID) may be a local path or an alias rather than a
# Hugging Face repo id; bench/client.sh points this at the checkpoint.
TOKENIZER="${TOKENIZER:-}"
# Optional: copy results here after every cell (e.g. an EFS path), so a
# terminated instance loses at most one cell.
MIRROR_DIR="${MIRROR_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

# "input:output" token shapes -- spanning prefill-bound to decode-bound.
#
# IMPORTANT: Neuron's torch.compile backend forces dynamic=False, so each
# distinct shape needs a compiled bucket. These shapes and concurrencies MUST
# be covered by the server's neuron_config buckets, or the sweep triggers
# recompiles mid-run and the latency numbers become meaningless.
#
# On the Neuron native stack every shape is a compiled bucket: start the server
# with buckets that cover this matrix (e.g. a *-sweep profile with
# bench/serve_neuron.sh), and keep CONCURRENCIES <= the server's MAX_NUM_SEQS
# and below the scheduler's KV admission cap (recorded in server-context.json);
# requests above it only queue.
SHAPES="${SHAPES:-1024:128 4096:256 8192:1024}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16}"
# Enough prompts for percentiles to mean something: >= 10x concurrency.
# parse_results.py trims the first and last N requests of each cell (closed-loop
# ramp and drain, see PLAN.md 5a), so the floor is set high enough that the
# steady-state window still holds >= 30 samples at low concurrency.
PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-10}"
MIN_PROMPTS="${MIN_PROMPTS:-40}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
# vllm bench serve 0.29 no longer defaults to greedy; without this the server
# applies the model's generation_config sampling. Pin it so every cell, and the
# Locust cross-check, does identical work.
TEMPERATURE="${TEMPERATURE:-0}"
# Unique prompts per cell AND per run. The random dataset is seeded (default
# seed 0), so without this every cell and every rerun sends the SAME prompts;
# the server has prefix caching on, and on Neuron a prefix-cache hit makes
# TTFT slower, not faster (PLAN.md 5b). The base is recorded in metadata.
SEED_BASE="${SEED_BASE:-$(date +%s)}"

mkdir -p "$RESULTS_DIR"

if ! curl -sf "${BASE_URL}/health" >/dev/null; then
  echo "ERROR: no healthy server at ${BASE_URL}. Start it with launch.sh up or bench/serve_neuron.sh up." >&2
  exit 1
fi

# Capture the environment alongside results -- both TorchNeuron and the vLLM
# plugin are Beta, so resolved versions are part of the measurement.
{
  echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_url: $BASE_URL"
  echo "model: $MODEL"
  echo "NEURON_LOGICAL_NC_CONFIG: ${NEURON_LOGICAL_NC_CONFIG:-unset}"
  echo "VLLM_NEURON_BACKEND: ${VLLM_NEURON_BACKEND:-unset}"
  echo "server_version: $(curl -sf "${BASE_URL}/version" || echo unknown)"
  echo "server_models: $(curl -sf "${BASE_URL}/v1/models" || echo unknown)"
  echo "seed_base: $SEED_BASE"
  echo "server_cache_config: $(curl -sf "${BASE_URL}/metrics" | grep '^vllm:cache_config_info' || echo unknown)"
  # Non-fatal: a client container without /dev/neuron* has no devices to list,
  # and under pipefail a failing neuron-ls would abort the sweep silently.
  { command -v neuron-ls >/dev/null && neuron-ls 2>&1 | head -30; } || echo "neuron-ls: unavailable here"
  python -m pip freeze 2>/dev/null | grep -iE "vllm|neuron|^torch" || true
} > "$RESULTS_DIR/run-metadata.txt"
echo "metadata -> $RESULTS_DIR/run-metadata.txt"

# Server counters that can silently invalidate a cell: prefix-cache hits
# (prompt reuse) and preemptions (KV exhaustion). Snapshotted around each cell.
# Through nginx, /metrics is answered by whichever replica nginx picks, so a
# before/after pair would mix replicas. METRICS_URLS lists every server to sum
# over (bench/client.sh sets it); default is the target itself.
METRICS_URLS="${METRICS_URLS:-$BASE_URL}"
server_counters() {
  for u in $METRICS_URLS; do curl -sf "$u/metrics"; done | awk '
    /^vllm:prefix_cache_queries_total/ {q+=$NF}
    /^vllm:prefix_cache_hits_total/    {h+=$NF}
    /^vllm:num_preemptions_total/      {p+=$NF}
    END {printf "{\"prefix_queries\": %s, \"prefix_hits\": %s, \"preemptions\": %s}", q+0, h+0, p+0}'
}

cell=0
for shape in $SHAPES; do
  IN="${shape%%:*}"
  OUT="${shape##*:}"
  for c in $CONCURRENCIES; do
    n=$(( c * PROMPT_MULTIPLIER ))
    (( n < MIN_PROMPTS )) && n=$MIN_PROMPTS
    name="vllmbench_in${IN}_out${OUT}_c${c}"
    out_file="$RESULTS_DIR/${name}.json"

    if [[ -f "$out_file" ]]; then
      echo "--- skip ${name} (exists) ---"
      continue
    fi

    cell=$((cell + 1))
    seed=$((SEED_BASE + cell))
    echo "=== ${name}: in=${IN} out=${OUT} concurrency=${c} prompts=${n} seed=${seed} ==="
    before=$(server_counters || echo '{}')
    # `random` dataset with fixed lengths gives controlled token shapes
    # rather than observed ones.
    if ! vllm bench serve \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        ${TOKENIZER:+--tokenizer "$TOKENIZER"} \
        --dataset-name random \
        --random-input-len "$IN" \
        --random-output-len "$OUT" \
        --num-prompts "$n" \
        --seed "$seed" \
        --max-concurrency "$c" \
        --request-rate "$REQUEST_RATE" \
        --ignore-eos \
        --temperature "$TEMPERATURE" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,95,99 \
        --save-result \
        --save-detailed \
        --result-filename "$out_file"; then
      echo "WARN: ${name} failed; continuing sweep" >&2
    fi
    after=$(server_counters || echo '{}')
    echo "{\"seed\": $seed, \"before\": $before, \"after\": $after}" > "$RESULTS_DIR/${name}.server.json"
    if [[ -n "$MIRROR_DIR" ]]; then
      mkdir -p "$MIRROR_DIR" && cp -p "$RESULTS_DIR"/* "$MIRROR_DIR"/ 2>/dev/null || \
        echo "WARN: mirror to $MIRROR_DIR failed" >&2
    fi
    sleep 5
  done
done

echo
echo "=== sweep complete: $RESULTS_DIR ==="
echo "Next: python bench/parse_results.py --results-dir $RESULTS_DIR"
echo "Copy results off-instance before the instance is terminated."
