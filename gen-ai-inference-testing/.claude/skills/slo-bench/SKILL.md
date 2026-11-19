---
name: slo-bench
description: Measure latency SLOs (TTFT, ITL, per-request stalls) of a vLLM server started by launch.sh, using the bench/ tools - a vllm bench serve sweep over shapes x concurrencies, an optional streaming Locust cross-check and multi-replica contention run, then SLO tables and a proposed SLO. Use when asked to benchmark, sweep, find the knee, derive or check an SLO, or measure TTFT/ITL. Not for the throughput-only Locust notebooks.
argument-hint: "[model] [shapes=IN:OUT,...] [conc=1,2,4,...] [replicas=N] [locust] [ttft_p99=MS itl_p99=MS] [keep]"
---

# SLO benchmark with bench/

Run everything from `gen-ai-inference-testing/`. The tools and their options are described in
`bench/README.md`; read `docs/neuron/README.md` section 4 before interpreting results on Neuron.

Arguments: `$ARGUMENTS`. Anything not given takes the default below. Confirm the resolved plan
(model, server settings, shapes, concurrencies, optional steps, rough duration) with the user before
starting a server or a sweep, since both occupy the accelerators for a long time.

| Argument | Default |
|---|---|
| model | the model of the server already running; otherwise ask |
| `shapes` | `1024:128,4096:256,8192:1024` (input:output tokens) |
| `conc` | `1,2,4,8,16` (closed-loop clients per replica) |
| `replicas` | not run; `replicas=N` adds an N-replica contention run |
| `locust` | not run; adds a streaming Locust cross-check of up to 3 cells |
| `ttft_p99`, `itl_p99`, ... | none; with targets, also report the max load inside them |
| `keep` | stop a server this skill started; `keep` leaves it running |

## Rules

- **Report, don't work around.** If a step fails, show the actual error and your diagnosis to the
  user before trying a fix. Say "I do not know yet" rather than guess.
- **No measured numbers in git.** Results stay in `/tmp/results/<run>` (mirrored to
  `$HOME/efs/lab/results/<run>` when that exists). Never commit them or paste them into tracked files.
- **Don't edit bench scripts while a run is going**: bash reads scripts incrementally; `client.sh`
  snapshots the harness per run for this reason.
- Use a new run name per sweep (`<UTC timestamp>-<label>`); the tools give every cell unique prompts,
  so do not reuse fixed seeds or prompt files.
- Run as the normal user, not with sudo.

## 1. Server

Check for a running server: `docker ps --format '{{.Names}}'` shows `compose-server_0-1`.

- **Running**: use it. Read its settings (`docker exec compose-server_0-1 cat /tmp/config.yaml`) and
  check the plan against them (below). If they do not fit, tell the user instead of restarting a
  server you did not start.
- **Not running, Neuron** (`neuron-ls` works): start it with `bench/serve_neuron.sh`, settings as
  environment variables (`MODEL_ID=<hf id>` or `MODEL_DIR=$HOME/snapshots/<dir>`,
  `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `NUM_REPLICAS`, `PREFIX_CACHING`,
  `BUCKET_PROFILE`; header of the script). The first start of a configuration compiles and can take
  tens of minutes; tell the user.
  ```bash
  MODEL_ID=<model> TENSOR_PARALLEL_SIZE=8 MAX_MODEL_LEN=<len> MAX_NUM_SEQS=<seqs> bash bench/serve_neuron.sh up
  ```
- **Not running, GPU**: start it the notebook way (`launch.sh up` with the notebook's environment)
  or ask the user to. The client tools have only been verified on Neuron: on CUDA use `TARGET=nginx`
  only (replica ports differ) and treat any failure as a tooling gap to report.

The plan must fit the server:
- `MAX_MODEL_LEN` >= the largest `input + output` (e.g. 10240 for `8192:1024`);
- `MAX_NUM_SEQS` >= the largest concurrency;
- Neuron native stack: every shape is a compiled bucket. The default `BUCKET_PROFILE=ladder` covers
  prefills up to 8192 and batch sizes 1, 2, 4, ...; shapes outside that recompile mid-run and
  invalidate the latencies;
- concurrencies above the KV admission cap (`kv_cache_max_concurrency`, recorded per run in
  `server-context.json`) only queue; `parse_results.py` marks them OVER CAP.

## 2. Sweep (always)

One replica, so the numbers are per replica (`TARGET=0`; `TARGET=nginx` loads all replicas behind
nginx):
```bash
TARGET=0 bash bench/client.sh vllm <run> -e SHAPES="<in:out ...>" -e CONCURRENCIES="<c ...>"
```
Long sweeps: run it in the background and wait for it to finish; do not poll in a tight loop.

## 3. Locust cross-check (if `locust`)

Up to three cells from the sweep, e.g. the smallest shape at the top concurrency and a middle shape
at concurrency 1 and 8. Requests = max(40, 10 x users):
```bash
TARGET=0 bash bench/client.sh locust <run> <in> <out> <users> <requests>
python3 bench/compare_harnesses.py --results-dir /tmp/results/<run> --out /tmp/results/<run>/harness_compare.md
```

## 4. Multi-replica contention (if `replicas=N`)

`N` <= the number of replicas the server runs (upstreams in nginx). Keep it to two shapes and three
concurrencies unless asked:
```bash
N=<N> SHAPES="<in:out ...>" CONCURRENCIES="<c ...>" bash bench/multi_replica.sh <run>-mr
python3 bench/compare_replicas.py --baseline /tmp/results/<run> \
  --replica-dirs $(ls -d /tmp/results/<run>-mr-r<N>-rep*) > /tmp/results/<run>/replica_compare.md
```

## 5. Analysis

```bash
python3 bench/parse_results.py --results-dir /tmp/results/<run> [--slo-ttft-p99 MS --slo-itl-p99 MS ...]
```
Writes `slo_table.csv`, `slo_knee.md` and, with targets, `slo_feasibility.md`. Without matplotlib
on the host it skips `slo_curves.png`; for the plot, run it in the server image:
```bash
docker run --rm -v /tmp/results:/tmp/results -v $PWD/bench:/bench --entrypoint python3 \
  "$(docker inspect -f '{{.Config.Image}}' compose-server_0-1)" /bench/parse_results.py --results-dir /tmp/results/<run>
```

## 6. Report

Summarise for the user, citing the files:
- the context line (model, dtype, TP, instance, stack, replicas) from `slo_knee.md`;
- per shape: latency floor (concurrency 1), knee by p99 TTFT and by p99 ITL, the proposed
  latency-optimal and capacity SLOs, and the worst gap per request at the top concurrency;
- the Locust and multi-replica verdicts, if run;
- on Neuron at concurrency > 1: TTFT is mostly queueing behind serialized prefills and streams pause
  during other requests' prefills (`docs/neuron/README.md` section 4.1), so judge stalls by
  "worst gap/req", not p99 ITL;
- anything abnormal: failed requests, OVER CAP cells, preemptions or prefix-cache hits in the
  per-cell counters, recompiles in the server log.

Then, unless `keep` was given, stop a server this skill started (`bash bench/serve_neuron.sh down`).
