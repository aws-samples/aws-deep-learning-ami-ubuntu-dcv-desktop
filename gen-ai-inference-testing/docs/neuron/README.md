# vLLM on AWS Neuron: legacy and native TorchNeuron stacks

`openai_server` + `vllm` on Neuron is one engine with two stacks, built into the same image
(`openai_server-vllm:neuron`) by `scripts/openai_server-vllm-neuron.sh`:

| Stack | Hardware | What it is |
|---|---|---|
| **native** | trn2, trn3 (`NEURON_VLLM_STACK=auto` default there, decoder tasks) | vLLM 0.29 + `vllm-neuron` plugin on the native TorchNeuron backend (`torch.compile` -> StableHLO -> `neuronx-cc`), Neuron SDK 2.33 DLC |
| **legacy** | inf2, trn1, anything else (default) | the earlier vLLM DLC (Neuron SDK 2.28, torch-neuronx / XLA, NxD Inference), unchanged |

Neuron documents TorchNeuron and the `vllm-neuron` plugin for **trn2 and trn3 only** ("supported on
Trainium 2 (trn2) and Trainium 3 (trn3) instances", Neuron 2.33). The plugin's hardware table lists
only trn2/trn3. Running the native stack on inf2/trn1 is not supported, and the native run script
refuses to start there.

Override with `NEURON_VLLM_STACK=native|legacy` (environment of `launch.sh` / the build script, or
the notebooks' Neuron parameter cell). The image records its stack in `/opt/program/neuron-stack`
and the label `org.opencontainers.image.neuron-vllm-stack`. Each host builds its own image. An
image built on trn2 and pushed to a registry is the native stack, and does not run on inf2.

## Where the code is

```
scripts/openai_server-vllm-neuron.sh              picks the stack, builds the image
containers/openai_server-vllm-neuron/
  Dockerfile                                      both stacks (build args: base image, stack)
  Dockerfile.native-from-source                   reference full native build (not yet built)
  resources/serve                                 replicas + nginx :8080; core pinning per stack
openai_server/vllm/run.sh                         shared CUDA/Neuron script; hands native images to:
openai_server/vllm/neuron_native.sh               native config, buckets, prefix caching, env
config/neuron_profiles/                           optional per-model serving profiles (scripted path)
bench/                                            SLO tooling: serve_neuron.sh, client.sh, sweep, Locust,
                                                  parse/compare (engine-agnostic except serve_neuron.sh)
labstore/bootstrap/                               unattended pre-staging of a Neuron desktop
docs/neuron/qwen3-32b/                            the Qwen3-32B trn2 work: plan, findings, test record
```

## Native-stack settings

`compose/neuron.yaml` passes the standard variables (`MODEL_ID`, `TENSOR_PARALLEL_SIZE`,
`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `TRUST_REMOTE_CODE`). The native-only settings go in
`$HOME/scripts/neuron_native.env`, which `launch.sh` mounts as `/scripts` (written by the notebooks'
Neuron cell or `bench/serve_neuron.sh`):

| Setting | Default | Notes |
|---|---|---|
| `PREFIX_CACHING` | `false` | each setting compiles separately; see "Prefix caching" below |
| `NUM_REPLICAS` | all | servers started; one per `TENSOR_PARALLEL_SIZE` logical cores |
| `BUCKET_PROFILE` | `ladder` | `ladder`: batch 1,2,4..MAX_NUM_SEQS and prefill 1024..8192; `recipe`: one bucket each |
| `MAX_NUM_BATCHED_TOKENS` | min(8192, MAX_MODEL_LEN) | prefill chunk size |

## What holds for any model on the native stack

Found on trn2 with Qwen3-32B (details and evidence in `qwen3-32b/PLAN.md`), but properties of the
stack rather than of the model:

- **Every input shape is a compiled bucket.** The backend forces `dynamic=False`. A prefill is padded
  up to the next `num_batched_tokens` bucket and a decode batch up to the next `num_seqs` bucket.
  Anything outside the buckets recompiles. The bucket choice sets both latency and compile time.
- **One prefill at a time, and prefill pauses decode.** The plugin's scheduler runs one prefill per
  step, never mixes prefill and decode in a step, and hides running decodes while a prefill waits.
  Consequences for measurement: TTFT at concurrency > 1 is mostly queueing; a stream can freeze for
  the length of other requests' prefills while p99 inter-token latency still looks fine. Measure the
  per-request longest token gap (`bench/parse_results.py` reports it).
- **Admission is capped by worst-case KV.** The scheduler admits at most
  `kv_cache_max_concurrency` requests (KV budget / `max_model_len`), reported in the server's
  `vllm:cache_config_info` metric. `MAX_NUM_SEQS` above it only queues; it is not more concurrency.
- **Prefix caching needs segmented prefill.** The plugin serves a prefix-cache hit only through
  segmented prefill and refuses to start with prefix caching on otherwise. `neuron_native.sh` sets the
  segment buckets when `PREFIX_CACHING=true`. A hit was measured to take *longer* to the first token
  than a fresh prefill for 4k prompts; test with real traffic before enabling it.
- **Keep the NKI kernel cache with the compile cache.** `NKI_COMPILE_CACHE_URL` defaults to the
  container's `/var/tmp`. A later container that reuses cached graphs but compiles a new shape then
  fails in `neuronx-cc` (`NCC_EVRF059 Kernel file ... does not exist`). `neuron_native.sh` points it
  at `/cache/nki`.
- **Container settings:** `NEURON_SKIP_EFA_AFFINITY=1` is mandatory (without it every TP worker
  fails with "No EFA device found"). Pin cores with `NEURON_VISIBLE_DEVICES` (logical core ids).
  The plugin rejects `NEURON_RT_VISIBLE_CORES`. Replicas sharing a compile cache must start one
  after another, which `serve` does.
- **Root-owned caches.** `compose/neuron.yaml` runs the server as root, so `$HOME/cache` should be
  local disk, copied to shared storage as your own user. `bench/serve_neuron.sh` refuses an
  EFS-backed `$HOME/cache`.

## Adding a model

- Usually nothing: set the model in the notebook (or `MODEL_ID` for `bench/serve_neuron.sh`) with
  `openai_server` + `vllm`. Defaults come from the standard variables and the `ladder` buckets.
- For a model with a published Neuron recipe, or a benchmark configuration you want to pin, add
  `config/neuron_profiles/<model>-<family>-<purpose>.env`.
- Encoder tasks (embeddings, rerankers, ...; `ENCODER_TYPE` set by `locust_encoder.ipynb`) keep the
  **legacy** stack even on trn2/trn3: `auto` picks legacy whenever `ENCODER_TYPE` is set, because the
  `run_<task>.sh` scripts only implement the legacy path. Native encoder support is future work.

## trn3

The native path is enabled on trn3 by the same rules (stack `auto` -> native; `neuron_native.sh`
accepts `trn3*`), but **nothing has been run on trn3**. Expect to revisit per model: the default LNC
and logical cores per device, HBM per device (which changes the right TP), and MXFP8, which AWS's
recipes list for Trn3 only (no dtype/quantization switch exists yet).
