# Test summary: Qwen3-32B on Trainium2, vLLM + native TorchNeuron

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

Pass/fail record only. **No performance numbers in this repository**: measured TTFT, ITL,
throughput and the like live with the run artifacts on the lab store
(`/home/ubuntu/efs/lab/results/2026111*`, mirrored to `s3://<lab-store-bucket>/lab/results/`).

## Environment

| Item | Value |
|---|---|
| Instance | `trn2.48xlarge` |
| Topology | TP8 at LNC2 = 8 logical NeuronCores = 2 Trainium2 devices per replica; 8 replicas fit |
| Model | `Qwen/Qwen3-32B` (dense), revision `main` (not pinned), BF16 |
| Base image | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` |
| Legacy-stack base image (regression test) | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.13.0-neuronx-py312-sdk2.28.0-ubuntu24.04` |
| vLLM (native stack) | `0.29.0` |
| vLLM Neuron plugin | `vllm-neuron 0.29.0.1.2.0`, `VLLM_NEURON_BACKEND=neuron_native` |
| PyTorch | `2.13.0` |
| TorchNeuron | `libtorch-neuronx-lite 2.13.0.1.0` (vendored by the plugin; `torch-neuronx` not installed) |
| `torch-xla` | `2.13.0` installed in the DLC, **not used** (native route) |
| Neuron compiler / NKI | `neuronx-cc 2.28.7631.0` / `nki 0.7.0` |
| Neuron runtime / collectives / tools | `2.35.30.0` / `2.35.43.0` / `2.33.38.0` |
| Neuron driver (host) | `2.31.8.0` |
| Neuron SDK | 2.33 per the DLC tag and tools version (not separately confirmed) |
| Python | `3.14` in the DLC; host Python `3.14` |
| Load clients | `vllm bench serve` (vLLM 0.29.0), Locust `2.47.0` |
| Other inference frameworks | none. NxD Inference is not a dependency of this plugin line; Triton/DJL not used |

## Results

### 2026-11-14 — first hardware run (standalone container)
| Test | Result | Notes |
|---|---|---|
| Serve published Qwen3-32B Trn2 recipe (TP8/LNC2) | **PASS** | after adding `NEURON_SKIP_EFA_AFFINITY=1` (all TP workers died without it) |
| Real completion returned | **PASS** | |

### 2026-11-15 — SLO work (standalone containers)
| Test | Result | Notes |
|---|---|---|
| `/health`, "capital of France" → Paris, 16 concurrent non-empty (published config) | **PASS** | |
| Same checks on the widened-bucket sweep config | **PASS** | greedy text differs from the control config (different compiled graphs) |
| O-7: plugin scheduler behaviour understood before measuring | **PASS** | PLAN.md 5a |
| Benchmark harness, narrow run | **PASS** after fix | `neuron-ls` under `pipefail` silently aborted the client |
| First Locust vs `vllm bench` cross-check (±10%) | **FAIL** | root cause: prompts reused across runs → prefix-cache hits (PLAN.md 5b), not instrumentation |
| Cross-check re-run with unique prompts (3 cells) | **PASS** | within tolerance on median TTFT and throughput |
| Single-replica sweep, 3 shapes × 6 concurrencies | **PASS** | 0 prefix-cache hits within cells, 0 preemptions; c=32 is above the KV admission cap |
| 2 replicas concurrently, 9 cells vs baseline (±10%) | **PASS** with caveat | 8/9 hold; 1 flag is a single tail gap at c=1 |
| 4 replicas concurrently, 9 cells vs baseline (±10%) | **PASS** with caveat | same cell flagged, rare decode hiccups under contention |

### 2026-11-16 — repo `launch.sh` path (`INFERENCE_ENGINE=vllm_native`)
| Test | Result | Notes |
|---|---|---|
| Build script skips on simulated trn1 / inf2 (exit 3) | **PASS** | `build-containers.sh` reports "skipped" |
| `run.sh` refuses non-trn2 instance and `DEVICE=cuda` | **PASS** | |
| Notebook engine/parameter cells, 8 server/engine/device combinations × 2 notebooks | **PASS** | existing paths unchanged; `vllm_native` refused off trn2 |
| Image build via `scripts/openai_server-vllm_native-neuron.sh` | **PASS** | |
| `launch.sh up`, 8 replicas, prefix caching off, config from env file | **PASS** | all replicas started one by one; nginx on :8080 |
| Paris check via nginx and on each of 8 replicas | **PASS** | |
| `locust_decoder.ipynb` load-test cells on host Python 3.14 | **FAIL** (existing) | `datasets==3.6.0` → `dill 0.3.8` breaks on Python 3.14; affects every engine |
| Same cells on Python 3.12 (`python:3.12-slim`) | **PASS** with caveat | 0 failures, load reached all 8 replicas; HF Hub rate-limited (429) the 64 workers' dataset downloads, so effective concurrency was below target |
| Prefix caching on, `MAX_MODEL_LEN=8192` | **FAIL** → fixed | plugin: "APC requires segmented prefill"; `run.sh` now sets segment buckets |
| Prefix caching on, after a prefix-off container on the same cache | **FAIL** → fixed | `neuronx-cc NCC_EVRF059`: NKI kernels were in container `/var/tmp`; now `NKI_COMPILE_CACHE_URL=/cache/nki` |
| Prefix off → down → prefix on, fresh cache, with fix | **PASS** | 0 `NCC_EVRF059`; flag reported `True`; repeated prompt produced cache hits |
| `bench/client.sh` (one replica and nginx), Locust client, `compare_harnesses.py` | **PASS** | counter snapshots through nginx fixed to sum all replicas |

### 2026-11-16 (later) — single `vllm` engine with two Neuron stacks, on top of upstream `main` (`9f0d35e`)
The `vllm_native` engine above was replaced: `openai_server` + `vllm` builds the native stack on
trn2/trn3 and the legacy stack elsewhere (`docs/neuron/README.md`).

| Test | Result | Notes |
|---|---|---|
| Upstream `aws-samples` `main` merged (24 commits, incl. GPU container updates) | **PASS** | one conflict (`build-containers.sh`), resolved in upstream's favour |
| Stack selection, simulated trn2 / trn3 / inf2 / trn1, `ENCODER_TYPE`, overrides | **PASS** | trn2/trn3 → native; inf2/trn1 → legacy; encoder tasks → legacy; overrides win |
| Native image via `scripts/openai_server-vllm-neuron.sh` on trn2 (auto) | **PASS** | stack label/marker `native`, `PJRT_DEVICE` empty; legacy apt list installs on the 26.04 base |
| Native via `INFERENCE_ENGINE=vllm`, prefix caching off, 2 replicas | **PASS** | `run.sh` → `neuron_native.sh`; per-replica `NEURON_VISIBLE_DEVICES`; no `PJRT_DEVICE`; warm reuse of the compile and NKI caches; Paris via nginx and each replica |
| Native, prefix caching on | **PASS** | segmented-prefill buckets set; 0 `NCC_EVRF059`; repeated prompt produced cache hits |
| **Legacy** forced on trn2 (`NEURON_VLLM_STACK=legacy`), notebook default model `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, 8 replicas | **PASS** | legacy build, `run.sh` and `serve` branches unchanged in behaviour (`NEURON_RT_NUM_CORES`); Paris on all 8 replicas |
| `locust_decoder.ipynb` load-test cells on the **host's Python 3.14** with upstream's deps (`datasets 5.2.0`), legacy stack, 64 users | **PASS** | 0 failures, load on all 8 replicas, no HF 429s (upstream's staggered workers) |
| Same cells, native stack, 1 replica | **PASS** | 0 failures |
| Notebook engine/Neuron cells, 9 combinations × 2 notebooks | **PASS** | CUDA/Triton/DJL unaffected; `vllm_native` refused by upstream's own engine check |
| `bench/serve_neuron.sh` with Qwen3-32B sweep settings (TP 8, 10240 tokens, 16 sequences, ladder buckets; then a profile file, since removed) | **PASS** | settings applied; cold compile of the sweep buckets |
| `bench/client.sh` (vllm + locust), `compare_harnesses.py`, `parse_results.py` | **PASS** | `server-context.json` recorded model, dtype, TP, instance, stack, replica slots and KV admission cap from the live server; parser used them (no hardcoded context) |

## Not tested
- **inf2 / trn1 hardware.** The legacy branches are unchanged in behaviour and were exercised on trn2
  with `NEURON_VLLM_STACK=legacy`, but a real regression needs those instances.
- **CUDA paths** (unchanged by this work; upstream's GPU updates were merged, not run here).
- `bench/client.sh` against a legacy or CUDA image (the native image carries Locust; others may not).
- `Dockerfile.native-from-source` (never built).
- `labstore/bootstrap/` end to end (it runs `labstore nvme-setup`, which must not be re-run on this
  instance).
- `litellm.ipynb` end to end (only its engine and parameter cells).
- GPU, inf2 and trn1 paths on real hardware (logic exercised with stubs only).
- `code_gen` / `text_image` request configs on the native stack; native encoder tasks (they keep legacy).
- The standalone TorchNeuron smoke test (PLAN Phase 1).
- 8-replica SLO, open-loop arrivals, and SLO numbers on the `launch.sh` path (the sweep used
  standalone containers, prefix caching on, and slightly different buckets).
- Qwen3-VL-32B, Qwen3-Embedding-8B.
