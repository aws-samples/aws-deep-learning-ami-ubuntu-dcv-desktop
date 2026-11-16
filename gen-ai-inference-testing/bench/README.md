# SLO benchmarking tools

For latency SLOs (time to first token, inter-token latency) rather than the throughput view of
`locust_decoder.ipynb`. The notebook's Locust user (`endpoint_user.py`) does not stream, so it cannot
measure TTFT or ITL.

| Tool | Purpose |
|---|---|
| `serve_neuron.sh` | start the Neuron vLLM server via `launch.sh` (settings as environment variables), wait for every replica |
| `client.sh` | run a client container in the server's network namespace, against nginx (`TARGET=nginx`) or one replica (`TARGET=<i>`); records the run context (`server-context.json`) |
| `run_vllm_bench.sh` | `vllm bench serve` sweep over shapes x concurrencies (source of truth); unique prompts per cell, prefix-cache and preemption counters per cell |
| `locustfile.py` | streaming Locust cross-check with a workload identical to the sweep; single-process |
| `multi_replica.sh` | load N replicas at once |
| `parse_results.py` | SLO table, steady-state percentiles, knee per metric, target feasibility, curves |
| `compare_harnesses.py` | Locust vs `vllm bench serve`, ±10% gate |
| `compare_replicas.py` | per-replica SLO under N-replica contention |

The client tools work against any OpenAI-compatible vLLM server started by `launch.sh`; by default
the client container is the running server's own image. Only `serve_neuron.sh` is Neuron-specific.

```bash
cd gen-ai-inference-testing
MODEL_DIR=$HOME/snapshots/Qwen3-32B MAX_MODEL_LEN=10240 MAX_NUM_SEQS=16 NUM_REPLICAS=1 bash bench/serve_neuron.sh up
TARGET=0 bash bench/client.sh vllm my-sweep -e SHAPES="4096:256" -e CONCURRENCIES="1 2 4 8"
TARGET=0 bash bench/client.sh locust my-sweep 4096 256 8 80
python3 bench/compare_harnesses.py --results-dir /tmp/results/my-sweep
python3 bench/parse_results.py --results-dir /tmp/results/my-sweep
bash bench/serve_neuron.sh down
```

Results go to `/tmp/results/<run>`, mirrored after every cell to `$HOME/efs/lab/results/<run>` when
that exists. Keep measured numbers out of git. On the Neuron native stack, read
`docs/neuron/README.md` before interpreting TTFT or ITL at concurrency > 1.
