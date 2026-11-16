# Qwen3 on Trainium2 — vLLM + native TorchNeuron

> **Paths in this document are historical (2026-11-16 restructure).** The work below was done under
> `gen-ai-inference-testing/trn2-qwen3-torchneuron/`, which no longer exists. Current locations:
>
> | Then | Now |
> |---|---|
> | engine `vllm_native`, `containers/openai_server-vllm_native-neuron/`, `openai_server/vllm_native/run.sh` | the existing `vllm` engine: `containers/openai_server-vllm-neuron/` builds the **native** stack on trn2/trn3 (`scripts/openai_server-vllm-neuron.sh`), `openai_server/vllm/neuron_native.sh` configures it |
> | `serve/run_qwen3_32b.sh`, `serve/precompile.sh`, `serve/launch_qwen3_32b.sh` (`CONFIG=recipe\|sweep`) | `bench/serve_neuron.sh` with the settings as environment variables (recipe: `MAX_MODEL_LEN=10240 MAX_NUM_SEQS=4 BUCKET_PROFILE=recipe`; sweep: `MAX_MODEL_LEN=10240 MAX_NUM_SEQS=16`) |
> | `SWEEP_MODE=1` | the sweep settings above |
> | `$HOME/scripts/vllm_native.env` | `$HOME/scripts/neuron_native.env` |
> | `bench/*` | top-level `bench/` (engine-agnostic) |
> | `bootstrap.sh`, `ops/`, `serve/download_model.sh` | `labstore/bootstrap/` |
> | `LAB-STORE-PLAN.md` | removed; its rationale is in `labstore/README.md` |
> | `notebooks/qwen3_32b_trn2_vllm.ipynb` | removed; use `locust_decoder.ipynb` (`openai_server` + `vllm`) |
>
> Engine-level findings are summarised in `docs/neuron/README.md`. Measured numbers stay on the lab
> store, not in git.

Serve Qwen3 models on a single `trn2.48xlarge` through vLLM running on the **native TorchNeuron**
backend, and derive defensible SLOs (TTFT, ITL, TPOT, throughput).

> **Status:** Qwen3-32B serving verified on `trn2.48xlarge` (2026-11-14). SLO sweep, harness
> cross-check and 2/4-replica runs done 2026-11-15. Moved onto the repo's `launch.sh` path
> 2026-11-16. Read [`PLAN.md`](./PLAN.md) first: pinned versions, the risk register,
> open items O-1..O-8, and **Section 5a**, which explains why TTFT and p99 ITL at concurrency > 1
> on Neuron mostly measure prefill queueing and stalls. Measured numbers are kept out of the repo;
> they are stored with their run artifacts on the lab store.

## Why this is separate from `containers/openai_server-vllm-neuron`

The existing container is the **legacy** path: vLLM 0.7.2 patched, `torch-neuronx` + XLA
(`PJRT_DEVICE=NEURON`), NxD Inference for model implementations, Neuron SDK 2.28, trn1/inf2 era.

This is the **forward** path: vLLM 0.29.0 via the `vllm-neuron` plugin (no patches), TorchNeuron
native compilation (`VLLM_NEURON_BACKEND=neuron_native`), model implementations inside the plugin
(NxD Inference was dropped as a dependency at 0.21.0), Neuron SDK 2.33.0, trn2/trn3 only.

The old patch files do not apply and the compilation model is different, so this is a new example
rather than an upgrade. The legacy path is left untouched and working.

## Stack (Neuron SDK 2.33.0, 2026-10-19)

| Component | Version |
|---|---|
| TorchNeuron (Beta) | `libtorch-neuronx-lite 2.13.0.1.0` (vendored by the plugin; `torch-neuronx` not installed) |
| vLLM Neuron plugin (Beta) | `0.29.0.1.2.0` (vLLM `0.29.0`) |
| PyTorch | `2.13.0` |
| `neuronx-cc` | `2.28.7631.0` |
| Neuron Runtime | `2.35.30.0` |
| Python / OS in DLC | `3.14` / Ubuntu 26.04 |
| NKI / Neuron driver (host) | `0.7.0` / `2.31.8.0` |

Versions as installed in the DLC and recorded on 2026-11-15. Earlier drafts listed PyTorch and
TorchNeuron 2.14; the DLC ships 2.13.

Both headline components are **Beta** — pin exactly and record `pip freeze` with every result.

## How this fits the repo

The server is a normal `gen-ai-inference-testing` combination, `openai_server` + `vllm_native` +
`neuron`, so `launch.sh`, `compose/neuron.yaml`, `locust_decoder.ipynb` and `litellm.ipynb` all
drive it like any other engine:

```
gen-ai-inference-testing/
  scripts/openai_server-vllm_native-neuron.sh        build; skips (exit 3) on anything but trn2/trn3
  containers/openai_server-vllm_native-neuron/
    Dockerfile                    FROM the Neuron 2.33 vLLM DLC; Trainium env baked in
    Dockerfile.from-source        reference full-stack build (not yet built)
    resources/serve               one vLLM per TP core group, nginx on :8080
  openai_server/vllm_native/run.sh   MODEL_ID/TP/MAX_* -> vLLM config; Trainium-only knobs
                                     from $HOME/scripts/vllm_native.env
  config/decoder/text_only/openai_server-vllm_native.yaml   Locust/LiteLLM request config
```

This directory holds what is specific to Qwen3-32B on trn2 and to SLO measurement, which the
repo's harness does not cover. The repo's Locust user (`endpoint_user.py`) does not stream, so it
cannot measure TTFT or ITL.

```
trn2-qwen3-torchneuron/
  serve/launch_qwen3_32b.sh   named configs (recipe | sweep | notebook) on top of launch.sh
  serve/download_model.sh     pinned checkpoint download into $HOME/snapshots
  bench/client.sh             run a client in the server's network namespace (nginx or one replica)
  bench/run_vllm_bench.sh     `vllm bench serve` sweep (source of truth)
  bench/locustfile.py         streaming cross-check harness (single-process)
  bench/multi_replica.sh      load N replicas at once
  bench/parse_results.py      JSON -> SLO table, knee, feasibility, curves
  bench/compare_harnesses.py  Locust vs vllm bench, +/-10% gate
  bench/compare_replicas.py   per-replica SLO under N-replica contention
  bootstrap.sh, ops/          unattended pre-staging of the desktop
  PLAN.md                     plan, findings (2b, 5a-5c), open items
  TEST-SUMMARY.md             versions and PASS/FAIL per test (no performance numbers)
```

## Quick start on a trn2.48xlarge

Interactive: open `gen-ai-inference-testing/locust_decoder.ipynb`, set
`hf_model_id = 'Qwen/Qwen3-32B'` and `inference_engine = 'vllm_native'`, and run it. The
Trainium-only cell sets prefix caching (off by default), the replica count and the bucket profile.

Scripted SLO sweep:

```bash
cd gen-ai-inference-testing/trn2-qwen3-torchneuron
bash serve/download_model.sh                         # -> $HOME/snapshots/Qwen3-32B

# Builds the image, compiles on first start, waits until every replica is up.
CONFIG=sweep PREFIX_CACHING=false bash serve/launch_qwen3_32b.sh up
#   CONFIG=recipe  = the published AWS control (concurrency <= 4 per replica)
#   PREFIX_CACHING=true is a separate compile; see PLAN.md 5b before using it
#   NUM_REPLICAS=1 for a single-replica baseline

TARGET=0 bash bench/client.sh vllm sweep-r1 -e SHAPES="4096:256" -e CONCURRENCIES="1 2 4"
TARGET=0 bash bench/client.sh locust sweep-r1 4096 256 8 80
python3 bench/compare_harnesses.py --results-dir /tmp/results/sweep-r1
python3 bench/parse_results.py --results-dir /tmp/results/sweep-r1 --admission-cap 21
N=4 bash bench/multi_replica.sh contention

bash serve/launch_qwen3_32b.sh down
```

Results go to `/tmp/results/<run>` and, when `$HOME/efs/lab/results` exists, are mirrored there
after every cell. Keep measured numbers out of git.

## Three things that will bite you

1. **Precompile, always.** First run traces on CPU and compiles to NEFF (docs default the timeout
   to 1200s). The cache fingerprint invalidates on *any* version, compiler-flag or config change,
   so freeze your sweep matrix, then precompile once per config
   (`serve/launch_qwen3_32b.sh precompile`). On a time-limited instance
   this is the difference between collecting data and watching a compiler.
2. **`--ignore-eos` (and greedy decoding) are mandatory in benchmarks.** Without it, output length depends on the model's
   stopping behaviour and ITL comparisons across cells become meaningless. `vllm bench serve` 0.29
   no longer defaults to greedy, so `run_vllm_bench.sh` pins `--temperature 0`.
3. **BF16 only on trn2.** The Qwen3-32B recipe advertises MXFP8, but that requires **Trn3**.

Also: do not install `torch-neuronx` into the serving image. The plugin uses
`libtorch-neuronx-lite`, which vendors the TorchNeuron compile components — the two together are a
conflict risk. Standalone TorchNeuron work gets its own venv.

## Capacity math

`trn2.48xlarge` = 16 Trainium2 devices x 8 NeuronCore-v3 = **128 physical cores**, 1,536 GiB HBM.
At `NEURON_LOGICAL_NC_CONFIG=2` that is **64 logical cores**.

Qwen3-32B at TP8/LNC2 uses 8 logical cores = **2 of 16 devices**, so **8 independent replicas fit
on one instance**. That makes a single box a real throughput-scaling testbed, and it is the number
that matters for EKS bin-packing later.

## Model coverage

| Model | Status | Notes |
|---|---|---|
| Qwen3-32B Dense | served, cross-validated SLO sweep at 1/2/4 replicas (2026-11-15) | published Trn2 recipe (`CONFIG=recipe`) is the control; SLOs come from the widened-bucket sweep (PLAN.md 5a-5c) |
| Qwen3-VL-32B | planned | report text vs image TTFT separately |
| Qwen3-Embedding-8B | planned | TTFT/ITL do not apply; latency + embeddings/sec instead |

## References

- TorchNeuron — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html>
- vLLM Neuron plugin — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/index.html>
- Qwen3-32B tutorial — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/tutorials/tutorial-qwen3-32b.html>
- Precompile cache — <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/how-to-precompile-cache.html>
- Plugin source — <https://github.com/vllm-project/vllm-neuron> (`release-0.29.0.1.2.0`)
