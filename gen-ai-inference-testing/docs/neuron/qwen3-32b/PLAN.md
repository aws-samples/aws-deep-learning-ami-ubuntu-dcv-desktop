# Qwen3 on Trainium2 with vLLM + TorchNeuron — Implementation Plan

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

**Status:** serving verified on `trn2.48xlarge` (2026-11-14, Section 2b). O-7 resolved from plugin source
(Section 5a). Benchmark harness exercised on hardware 2026-11-15. Measured performance numbers are
kept **out of this repository** on purpose: they live with the run artifacts on the lab store
(`/home/ubuntu/efs/lab/results/<run>`), each with its config and `pip freeze`.
**Author:** drafted ahead of hardware access to a trn2.48xlarge.
**Goal:** a runnable, reproducible example that serves Qwen3 models on a single `trn2.48xlarge`
through vLLM on the **native TorchNeuron** backend, and produces defensible SLO numbers
(TTFT, ITL, TPOT, throughput) — then lifts to EKS.

---

## 1. Why this is a new example and not a patch of the existing one

The repo already has `gen-ai-inference-testing/containers/openai_server-vllm-neuron`. It is **not**
a starting point for this work:

| | Existing container | This example |
|---|---|---|
| Base image | `...vllm-neuronx:0.13.0-neuronx-py312-sdk2.28.0-ubuntu24.04` | `...vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` |
| vLLM | 0.7.2 / 0.6.6.post1, **patched** (`patch/vllm-neuron-*.patch`) | 0.29.0 via the **`vllm-neuron` plugin**, no patches |
| Neuron backend | `torch-neuronx` + XLA (`PJRT_DEVICE=NEURON`) | **TorchNeuron native** (`VLLM_NEURON_BACKEND=neuron_native`) |
| Model impls | NxD Inference (`neuronx-distributed-inference`) | **In the plugin** — NxDI dropped as a dependency at 0.21.0 |
| Hardware | trn1 / inf2 era | trn2 / trn3 only |

The compilation model changed (XLA graph tracing → `torch.compile` to NEFF through TorchNeuron's
Dynamo backend), the dependency graph changed, and the patch files no longer apply. Treat the old
container as the *legacy* path and this as the forward path.

**A dependency trap to avoid:** the plugin depends on **`libtorch-neuronx-lite`**, which *vendors*
the TorchNeuron compile components into its own wheel instead of depending on `torch-neuronx`.
Installing `torch-neuronx` alongside it in the same environment is a conflict risk. Use
`torch-neuronx` only for standalone TorchNeuron work (Section 6), not inside the serving image.

---

## 2. Pinned stack (AWS Neuron SDK 2.33.0, released 2026-10-19)

| Component | Version | Notes |
|---|---|---|
| Neuron SDK | `2.33.0` | |
| TorchNeuron | `libtorch-neuronx-lite 2.13.0.1.0` (Beta) | vendored by the plugin; measured in the DLC 2026-11-15 (this row said `2.14.3.0` before hardware access) |
| PyTorch | `2.13.0` | measured in the DLC 2026-11-15 (this row said `2.14` before hardware access) |
| vLLM Neuron plugin | `0.29.0.1.2.0` (Beta) | targets vLLM `0.29.0` |
| Neuron Compiler (`neuronx-cc`) | `2.28.7631.0` | |
| Neuron Runtime | `2.35.30.0` | |
| Python | `3.14` in the DLC / DLAMI | plugin requires ≥ 3.11 |
| OS | Ubuntu 26.04 in the DLC | |

Both TorchNeuron and the vLLM plugin are **Beta**. Expect API churn and pin exactly.

**Primary references**
- TorchNeuron: `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html`
- TorchNeuron quickstart: `.../torch/get-started/quickstart-torch-neuron.html`
- Native PyTorch inference: `.../inference-native-pytorch.html`
- vLLM Neuron plugin: `.../vllm-neuron/docs/index.html`
- Setup guide: `.../vllm-neuron/docs/getting-started/setup-guide.html`
- Qwen3-32B recipe: `.../vllm-neuron/docs/model-recipes/qwen3-32b.html`
- Qwen3-32B tutorial: `.../vllm-neuron/docs/tutorials/tutorial-qwen3-32b.html`
- Precompile cache: `.../vllm-neuron/docs/guides/how-to-precompile-cache.html`
- Config reference: `.../vllm-neuron/docs/guides/reference-configuration.html`
- Benchmark methodology: `.../vllm-neuron/docs/tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.html`
- Plugin source: `https://github.com/vllm-project/vllm-neuron` (branch `release-0.29.0.1.2.0`)
- Agent-friendly URL index: `https://awsdocs-neuron.readthedocs-hosted.com/llms.txt`

---

## 2a. How TorchNeuron and vLLM actually connect

The compile pipeline, end to end:

```
your model → TorchDynamo → FX graphs → AOT Autograd → Neuron FX passes
           → StableHLO MLIR (NO TorchInductor) → neuronx-cc → NEFF → Neuron Runtime
```

vLLM does not reimplement any of this. The plugin compiles each model *through*
TorchNeuron's `torch.compile` backend, so the serving stack and a standalone
`@torch.compile(backend="neuron")` script share one compiler path. That is why the Phase 1
smoke test is diagnostic: if `torch.compile(backend="neuron")` fails on a matmul, the fault is
TorchNeuron, not vLLM.

**`VLLM_NEURON_BACKEND` values** (two compilation routes):

| Value | Route | Lowering | Default opt level |
|---|---|---|---|
| unset | **native (default)** | AOTAutograd + torch-mlir → StableHLO → `neuronx-cc` | O2 |
| `neuron_native` | native | as above | O2 |
| `neuron` | native | as above | O2 |
| `vllm_neuron` | **XLA (deprecated)** | PyTorch/XLA → HLO → `neuronx-cc` | O1 |

Both routes compile identical `torch.compile` graphs onto the same runtime; only the lowering
differs. The XLA route is deprecated and slated for removal — use it only if the native route
produces results you cannot accept.

> Subtlety worth knowing: the native route is already the default, so setting
> `VLLM_NEURON_BACKEND=neuron_native` does not change the route. But the **precompile cache
> only activates when the variable is explicitly set**. Set it anyway.

### Three things this pipeline implies

**1. Dynamic shapes are not supported.** The backend forces `dynamic=False`; "each distinct
input shape requires separate compilation." This is the single most operationally important
fact in the whole stack: **buckets are not a tuning knob, they are the list of shapes that
exist.** A benchmark sweep over input lengths and concurrency must be covered by the server's
buckets or it recompiles mid-run and the latency numbers are garbage. Hence the separate
`CONFIG=sweep` (`serve/launch_qwen3_32b.sh`; called `SWEEP_MODE=1` in the 2026-11-15 records) and
`CONFIG=recipe`, the published control config.

**2. Graph breaks degrade but do not fail.** When Dynamo hits an unsupported op or a
data-dependent branch it splits the graph and compiles each segment independently — narrower
optimisation scope, not an error. Trivial pass-through segments are skipped entirely.

**3. Caching is three-layered**, and one layer is explicitly designed for what the lab store
does:

| Layer | Keyed by | Portable? |
|---|---|---|
| Persistent NEFF cache | lowered HLO + compiler config (content-addressed) | **yes — "reusable across instances via shared storage"** |
| HLO cache | shared between eager and compiled modes | within a host |
| Dynamo precompile cache | embeds StableHLO bytes for warm reload | fingerprint-bound |

That middle column is why `labstore` keys on resolved versions and topology rather than on a
friendly name, and the quoted phrase is direct documentation support for putting the cache on
EFS/S3.

Also: only `torch.compile(mode="default")` is supported — other modes warn and fall back.

## 2b. First run on real Trainium2 — 2026-11-14

Qwen3-32B served successfully on `trn2.48xlarge`. Verified
with a real completion: `vllm-0.29.0-tp8`, 20 tokens, `finish_reason: length`.

### `NEURON_SKIP_EFA_AFFINITY=1` is mandatory in a container

Without it every TP worker dies at startup:

```
RuntimeError: No EFA device found at /sys/bus/pci/devices/0000:c9:00.0/infiniband
              for BDF 0000:c9:00.0
-> Engine core initialization failed
```

The CloudFormation template *does* attach an EFA interface for `trn2.48xlarge`, but the
InfiniBand sysfs nodes are not visible inside the container. The Neuron runtime's own message
states EFA affinity is a CPU performance optimisation, **not** a correctness requirement. Set in
`containers/openai_server-vllm_native-neuron/Dockerfile` (2026-11-16; originally in the run script). **Revisit for multi-node**, where EFA is
not optional.

### The cold compile is ~5 minutes, not ~20

Measured end to end: container start 14:43:45Z → `Application startup complete` 14:49:07Z, with
**56 cache files / 371 MB** written. Earlier sections of this plan assumed ~20 min from the docs'
1200s timeout; that was pessimistic by about 4x. Budget accordingly — the compile is no longer the
dominant time sink, which weakens the case for treating cache reuse as urgent.

### Two warnings that matter later

**1. Chunked prefill:**
```
Chunked prefill is enabled on Neuron platform. Currently Neuron only supports chunking
prefills with batch size of 1. Mixing prefill and decode in the same batch is not supported.
```
This shapes how concurrency behaves and must be understood before any SLO number is trusted.
**Open item O-7.**

**2. Rank-less compile cache — a direct threat to the lab store:**
```
Kept the shared package directory /tmp/torchinductor_root/dynamo/topo_none. This process has
no known rank, so each rank reads the same torch.compile package. A warm restore can stop
with an error on a process group name. Set NEURON_RT_VISIBLE_CORES or RANK, or initialize
torch.distributed before the first compile.
```
**Correction (2026-11-15):** do **not** set `NEURON_RT_VISIBLE_CORES`. The plugin rejects it under
vLLM's multiprocess executor (`RuntimeError: NEURON_RT_VISIBLE_CORES cannot be used with
multi-processing execution on vLLM. Set NEURON_VISIBLE_DEVICES ...`). `NEURON_VISIBLE_DEVICES`
takes **logical core ids** (e.g. `8,9,...,15`) despite its name, and defaults to `0..TP-1`.
With it set, the precompile package is written per rank (`topo_<hash>`) and no longer
`topo_none`. See O-8.

The warm restore worked on 2026-11-14, but this warns it may fail *across* processes or
instances — exactly the cache-reuse path `labstore` exists to provide. Resolve before relying on
a warm restore in a future session. **Open item O-8.**

### Operational notes from the same run

- Publish the port: a `docker run` without `-p 8000:8000` leaves the server reachable only via
  `docker exec`, which looks identical to a failed startup from the host. (Standalone runs only. The `launch.sh` path
  publishes nginx on `:8080` through `compose/neuron.yaml`, and `bench/client.sh` reaches the
  per-replica ports through the container's network.)
- Run as `ubuntu`, not root. A validation instance that ran `labstore` as root left root-owned
  directories on the shared EFS which then blocked `ubuntu` from creating the cache path.
  **The same applies inside the container.** The DLC runs as root by default, so the 2026-11-14
  cold compile left the warm control cache under
  `cache/qwen-qwen3-32b/neuron-trainium2/86b6d4b6e8da/tp8-lnc2/{neuron,modelinfos}` owned by
  `root:root` (755). `ubuntu` can read it but cannot add to it. Fixing it needs a
  `chown -R ubuntu:ubuntu`, which is deliberately not done from an agent. The server runs fine
  **as uid 1000** (verified 2026-11-15: `--user 1000:1000 -e HOME=/tmp`; `/dev/neuron*` are
  0666), so start every container that way and new cache entries stay ubuntu-owned.
- SSM injects `AWS_SHARED_CREDENTIALS_FILE=/var/lib/amazon/ssm/credentials`, pointing at the
  account's Default Host Management role, which has no S3 access. Anything run through SSM Run
  Command must use `su - ubuntu` (a login shell) so credentials fall back to IMDS.
- DCV uses `authentication="system"`, and the template leaves `ubuntu` **locked** (`passwd -S`
  reports `L`). A password must be set before a DCV login will succeed.

## 3. The hardware, and the core math that drives every config

`trn2.48xlarge`:

- **16** Trainium2 devices → `/dev/neuron0` … `/dev/neuron15`
- **8** NeuronCore-v3 per device → **128 physical cores**
- **96 GiB** HBM per device → **1,536 GiB** total HBM
- 192 vCPU, 2,048 GiB host RAM
- EFA-capable (the repo's CFN template enables it for `trn2.48xlarge`)

**Logical NeuronCore (LNC)** fuses physical cores into one logical unit:

| LNC | Logical cores/device | Logical cores/instance | Meaning |
|---|---|---|---|
| 1 | 8 | **128** | max parallelism, less memory+compute per unit |
| 2 | 4 | **64** | fewer units, more memory+compute each |

`NEURON_LOGICAL_NC_CONFIG=2` is what the published Qwen3-32B recipe uses.

**The consequence worth internalizing:** Qwen3-32B wants **TP8 at LNC2** = 8 logical cores =
**2 physical devices**. One `trn2.48xlarge` has 64 logical cores at LNC2, so it fits
**8 independent TP8 replicas**. That makes a single instance a legitimate throughput-scaling
testbed, not just a single-replica box — and the repo's existing `serve` script already builds
replica "core groups" automatically, so this is reachable.

> **O-1 RESOLVED on hardware** (trn2.48xlarge):
> `neuron-ls --json-output` reports **`"nc_count": 4`** — i.e. **logical** cores per device at
> LNC2, not physical. `neuron-ls` also confirms `logical-neuroncore-config: 2`, so **LNC2 is the
> instance default** and needs no explicit setting. 16 devices x 4 = 64 logical cores, core IDs
> `0-63`, 96 GB per device.
>
> Consequence: the repo's existing `serve` replica math (`len(devices) * nc_count`) is **correct
> as-is** on trn2 — no `NEURON_CORES_PER_DEVICE` override needed. With
> `TENSOR_PARALLEL_SIZE=8` it yields 8 core-groups of 8 logical cores, matching the 8-replica
> figure below.

**BF16 only on trn2.** The Qwen3-32B recipe offers MXFP8, but the docs state it needs **Trn3**;
Trn2 is BF16 for this model. Do not plan FP8 numbers on this instance.

---

## 4. Published Qwen3-32B recipe (the baseline to reproduce first)

Requires "at least eight available logical NeuronCores at LNC2".

```bash
export NEURON_LOGICAL_NC_CONFIG=2
export VLLM_NEURON_BACKEND=neuron_native
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

vllm serve "$MODEL_PATH" \
  --served-model-name qwen3-32b \
  --dtype bfloat16 \
  --optimization-level 2 \
  --tensor-parallel-size 8 \
  --max-model-len 10240 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4
```

with `neuron_config` buckets:

```json
{"neuron_config": {
  "kv_segment_size_buckets":   [8192],
  "num_batched_tokens_buckets":[8192],
  "num_seqs_buckets":          [4]
}}
```

> **Observed 2026-11-15:** vLLM 0.29 logs `Unknown vLLM environment variable detected:
> VLLM_NEURON_COMPILATION_TIMEOUT`, and the plugin's Python source (0.29.0.1.2.0) never reads it.
> It appears to be a no-op in this release; whether a native component reads it is unknown. The
> recipe's value is kept for fidelity, but don't rely on it to bound a compile.

Checkpoint: `hf download Qwen/Qwen3-32B --revision "$BF16_REVISION" --local-dir "$BF16_MODEL"`

Validation per the tutorial: `curl localhost:8000/health`, then a completion for
"The capital of France is" expecting Paris, then 16 concurrent requests all returning non-empty.

**Reproduce this verbatim before changing a single flag.** It is the only configuration with
published AWS correctness + performance validation, so it is the control for everything after.

---

## 5. Phases

### Phase 0 — Prerequisites (off-instance, done)
- [x] trn2 instance available
- [x] DCV desktop stack ready to launch
- [ ] `HF_TOKEN` available on the instance (Qwen3 is gated on some mirrors — verify)
- [ ] EFS or S3 path chosen for the **compile cache** (Phase 3) and benchmark artifacts

### Phase 1 — Verify the hardware and the native backend (~20 min on instance)
Cheap, fast, and it fails loudly if the driver stack is wrong. Do it before any model work.

1. `neuron-ls` and `neuron-ls --json-output` → confirm 16 devices, record `nc_count` (**O-1**).
2. `cat /proc/modules | grep neuron`, `modinfo neuron` → driver (`aws-neuronx-dkms`) loaded.
3. Standalone TorchNeuron smoke test in its own venv (**not** the serving image):
   ```bash
   python -m pip install "torch-neuronx==2.14.*" "neuronx-cc==2.*" nki \
     --extra-index-url https://pip.repos.neuron.amazonaws.com \
     --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   then the quickstart assertion: `torch.neuron.is_available()`, `torch.neuron.device_count()`,
   and a matmul on `.to("neuron")` compared against CPU with `torch.testing.assert_close`.
4. `@torch.compile(backend="neuron")` on the same matmul → exercises the Dynamo path that vLLM
   itself uses, so a failure here localises the problem to TorchNeuron rather than vLLM.

Deliverable: not done (the standalone TorchNeuron smoke test was never run; serving was the
first hardware test).

### Phase 2 — Containers
**Primary — DLC base** (`gen-ai-inference-testing/containers/openai_server-vllm_native-neuron/Dockerfile`):
`FROM public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`,
plus nginx, `resources/serve` and the Trainium environment. Built by
`scripts/openai_server-vllm_native-neuron.sh`, which skips (exit 3) on anything but trn2/trn3.

**Reference — from source** (`Dockerfile.from-source` in the same directory): Ubuntu 26.04 +
Python 3.14, `neuronx-cc`, `nki`, `libtorch-neuronx-lite`, and the plugin built from
`git clone -b release-0.29.0.1.2.0 https://github.com/vllm-project/vllm-neuron.git`. Pins follow
the validated DLC. **Not yet built.** Build it ahead of time, not on an accelerator instance:
dependency resolution can take a long time to get right.

Verify either image with:
```bash
python -c "import vllm; from vllm.platforms import current_platform; print(current_platform.device_name)"
# expect: neuron
```

### Phase 3 — Compile cache strategy (the highest-leverage step)
First run traces on CPU and compiles; the docs default both timeouts to **1200s**, which tells you
the expected order of magnitude. On a time-limited instance this is the difference between collecting
data and watching a compiler.

- Set **before Python starts** (setting them in-process has no effect):
  `VLLM_NEURON_BACKEND=neuron_native`, `VLLM_CACHE_ROOT=/cache`.
- Cache path: `$VLLM_CACHE_ROOT/neuron/torch_compile/<fingerprint>/dp_<i>/rank_<r>`.
- Keep `/cache` (`$HOME/cache`) on **local** disk and copy it to the lab store (EFS + S3) as
  `ubuntu` with `labstore save`. The server container runs as root under `compose/neuron.yaml`,
  and root-owned entries on the shared EFS block `ubuntu` (Section 2b). `launch_qwen3_32b.sh`
  refuses an EFS-backed `$HOME/cache`.
- Confirm warm start from startup logs: cold shows misses + saves > 0; warm shows Dynamo hits and
  **zero** saves.
- **The fingerprint invalidates on any version, compiler-flag, or config change.** So freeze the
  serve config *before* the sweep, precompile once per config you intend to benchmark, and treat
  each new bucket/TP combination as a fresh compile. Budget for this explicitly.
- Multi-node has no cache coordination → `TORCH_CACHING_PRECOMPILE=0` there. Also incompatible
  with encoder-disaggregated (EPD) serving.

`serve/launch_qwen3_32b.sh precompile` warms the cache for a config (up, wait for every replica, down).

### Phase 4 — Serve integration (done 2026-11-16, O-2 resolved)
The example is a standard repo combination; `launch.sh`, `compose/neuron.yaml` and the GPU paths
are unchanged:

| Convention slot | Value |
|---|---|
| `INFERENCE_SERVER` / `INFERENCE_ENGINE` / `DEVICE` | `openai_server` / `vllm_native` / `neuron` |
| build script | `scripts/openai_server-vllm_native-neuron.sh` (trn2/trn3 only; exit 3 = skipped) |
| image | `openai_server-vllm_native:neuron` |
| container | `containers/openai_server-vllm_native-neuron/{Dockerfile,resources/serve}` |
| run script | `openai_server/vllm_native/run.sh` |
| request config | `config/decoder/text_only/openai_server-vllm_native.yaml` |
| notebooks | `locust_decoder.ipynb`, `litellm.ipynb`: `inference_engine = 'vllm_native'` |

How the constraints were met without touching `compose/neuron.yaml`:
- The variables compose does not pass (`VLLM_NEURON_BACKEND`, `NEURON_LOGICAL_NC_CONFIG`,
  `NEURON_SKIP_EFA_AFFINITY`, `VLLM_CACHE_ROOT`) are baked into the image, where no other engine
  sees them.
- Trainium-only knobs (`PREFIX_CACHING`, `NUM_REPLICAS`, `BUCKET_PROFILE`) go to
  `$HOME/scripts/vllm_native.env`. `launch.sh` mounts that directory as `/scripts` and never
  cleans it; `run.sh` prints what it applied.
- `resources/serve` keeps the repo contract (TP-sized core groups, servers on 8000+128*i started
  one at a time, nginx round-robin on 8080). It pins cores with `NEURON_VISIBLE_DEVICES`, because
  the plugin rejects `NEURON_RT_VISIBLE_CORES`. Its nginx read timeout is 1800 s, not 120 s:
  long-prompt requests under load ran for minutes in the sweep.
- Divergences from `openai_server/vllm/run.sh`, on purpose: `block-size` is left to the plugin
  (it resolves 32 on Neuron), `max-num-batched-tokens` is capped at 8192 rather than
  `MAX_MODEL_LEN`, and prefix caching is a switch (default off, like the vllm engine; see 5b).

### Phase 5 — Benchmarking and SLO determination
Two harnesses, cross-validated.

**Source of truth — `vllm bench serve`** (built into vLLM 0.29). Reports `median_ttft_ms`,
`p99_ttft_ms`, ITL, TPOT, `output_throughput`; supports `--max-concurrency`, `--request-rate`,
`--ignore-eos`, `--save-result`.

Fix the token shape so input/output lengths are controlled rather than observed — `--ignore-eos`
is essential or output length becomes a function of the model's stopping behaviour and ITL
comparisons stop meaning anything.

Sweep (`bench/run_vllm_bench.sh`):
- Token shapes: 1k/128, 4k/256, 8k/1k input/output — spanning prefill- and decode-bound regimes
- Concurrency: 1, 2, 4, 8, 16, 32 (1 = latency floor; raise until TTFT degrades)
- `--request-rate inf` for saturation, plus fixed rates for steady-state
- `--num-prompts` ≥ 10x concurrency so percentiles mean something

Then `bench/parse_results.py` folds the JSON into one CSV/table.

**Cross-validation — Locust** (`bench/locustfile.py`), run **single-process** (results are
collected in-process, so the repo's master/worker `run_locust.sh` would summarise one shard).
Workload is matched to `vllm bench serve` exactly: unique random token-id prompts per request
(prefix caching is on, so a repeated prompt turns TTFT into a cache-hit time), `/v1/completions`,
greedy, `ignore_eos`. Only the measurement is independent: ITL from streamed chunk arrival
timestamps, tokens from the server's `usage` block. Agreement within
~10% on median TTFT and throughput validates both; disagreement means one harness is
mis-instrumented and the numbers are not yet publishable.

**Deriving the SLO** — the actual deliverable:
1. At concurrency 1, record the **floor**: best achievable TTFT and ITL. No SLO can beat this.
2. Sweep concurrency and plot TTFT-p99 and ITL-p99 against throughput.
3. Find the **knee** — the concurrency beyond which latency rises faster than throughput.
4. Set the SLO just inside the knee, e.g. "p99 TTFT ≤ X ms and p99 ITL ≤ Y ms at N concurrent
   requests per TP8 replica, 4k input / 256 output, BF16".
6. Re-run at 2 and 4 replicas to confirm the per-replica SLO holds under instance-level contention
   — this is what makes the number meaningful for EKS capacity planning rather than a lab curio.

An SLO without its token shape, concurrency, replica count and dtype attached is not reusable.
Record all five with every number.

### 5c. What the 2026-11-15 sweep established (numbers live on the lab store, not here)

Artifacts: `/home/ubuntu/efs/lab/results/20261115T*` (mirrored to `s3://<lab-store-bucket>/lab/results/`).
Each run directory has the raw per-cell JSON with per-request detail, `<cell>.server.json`
counter snapshots, `run-metadata.txt` (versions, seed base, server cache config),
`slo/slo_table.csv`, `slo/slo_knee.md` and `slo/slo_curves.png`. Feasibility reports and the
harness and replica checks are alongside.

- **Harnesses agree.** On the `SWEEP_MODE` server, Locust and `vllm bench serve` matched well
  inside ±10% on median TTFT and throughput for all cells compared (`xval.md`), once prompt reuse
  was removed (5b).
- **Concurrency above the KV admission cap is strictly worse.** At `max_num_seqs=32` only about 21
  requests run at once. The c=32 cell had *lower* throughput than c=16 and much higher TTFT, so
  don't configure `max_num_seqs` above the cap the scheduler reports.
- **Within the cap, the ITL knee was not reached.** Throughput still outgrew p99 ITL at c=16 for
  every shape, so the capacity knee lies between 16 and the cap. Testing c≈20 is the cheap
  follow-up.
- **The TTFT knee is at c=1 for every shape** (5a), so TTFT SLOs come from target feasibility.
- **Per-replica SLOs hold at 2 and 4 replicas.** Per-replica TTFT and ITL percentiles and
  throughput stayed within ±10% of the single-replica baseline in every cell, and instance
  throughput scaled linearly. The one consistent contention effect: at c=1 on the longest shape,
  rare decode hiccups (a handful of gaps out of ~10^5) appear that the single replica never
  showed. They only matter for a hard per-stream max-gap SLO in the tens of milliseconds.
  8 replicas were not tested.
- **Starting replicas:** one container per replica, `NEURON_VISIBLE_DEVICES=8k..8k+7`,
  `PORT=8000+k`, sharing one cache directory. The topology hash depends on the TP rank layout, not
  the physical cores, so later replicas reuse NEFFs but still re-run graph extraction and rewrite
  the shared package files. That makes them partially warm, so **start replicas one at a time**.
  (Section 3's remark that the repo's `serve` script builds core groups automatically was not
  used or verified here.)
- **Not done:** fixed-rate (open-loop) arrivals, 8 replicas, c between 16 and the cap, prefix
  caching off, and the Qwen3-VL / Embedding variants.
- **Naming:** these runs used `SWEEP_MODE=1` (tokens `[1024,4096,8192]`, seqs up to 32, prefix
  caching on) through standalone containers. The repo path's `CONFIG=sweep` differs: seqs up to 16,
  tokens `[1024,2048,4096,8192]`, prefix caching off by default. Its numbers are not the same
  measurement until re-run.

### Phase 6 — The notebook
Superseded 2026-11-16. The interactive path is the repo's own `locust_decoder.ipynb` /
`litellm.ipynb` with `inference_engine = 'vllm_native'`. The SLO sweep is scripted
(`serve/launch_qwen3_32b.sh`, `bench/client.sh`, see README). The example's separate notebook,
never run and out of date, was removed.

### Phase 7 — The other two Qwen3 variants
Different architectures, so each needs its own pass. Sequence them after the dense model works —
they share the container and the harness, so the marginal effort is low once Phase 5 is solid.

| Model | What it exercises | Benchmark caveat |
|---|---|---|
| **Qwen3-32B Dense** | baseline decoder, TP8/LNC2/BF16 | the control |
| **Qwen3-VL-32B** | vision encoder, M-RoPE, on-device encoder cache, block-packing attention | TTFT includes image encode; report text and image TTFT separately or the number is meaningless |
| **Qwen3-Embedding-8B** | pooling models, no autoregressive decode | TTFT/ITL **do not apply** — SLO is request latency and embeddings/sec vs batch size |

### Phase 8 — EKS (scoped later, not designed here)
What this phase should carry forward, noted now so Phase 1–6 artifacts don't block it:
- The **from-source Dockerfile** becomes the production image; push to private ECR.
- Neuron device plugin + scheduling: `aws.amazon.com/neuron` resource requests, node selectors for
  `trn2.48xlarge`, and the fact that **TP8 at LNC2 = 2 devices**, so a pod requests a slice, not
  the whole node — up to 8 pods per instance.
- The **compile cache must be a shared RWX volume** (EFS CSI) or baked into the image. Per-pod cold
  compiles (measured ~5 min for the control config, Section 2b; longer for every extra bucket) would
  make autoscaling slow. Baking it in is probably correct for prod.
- `TORCH_CACHING_PRECOMPILE=0` if any deployment spans nodes.
- Readiness probes must tolerate a multi-minute warm start; `/health` plus a generous
  `initialDelaySeconds`/`failureThreshold`.
- The repo's `ray-vllm-trn2` and `ai-on-eks` work next door are the likely integration points.

---

## 6. TorchNeuron direct usage (beyond serving)

Worth a short notebook section independent of vLLM, since it is the part of the stack the user
specifically wants to evaluate and it is where the "native PyTorch" claim is actually visible:

- Device: `torch.device("neuron")`; autoloads on `import torch` via `PrivateUse1` — no explicit
  `import torch_neuronx` needed for basic use.
- Port pattern: `.to('cuda')` → `.to('neuron')`; `torch.autocast(device_type="neuron")`.
- Compile: `@torch.compile(backend="neuron")`; drop CUDA-specific kwargs.
- Distributed: DDP, FSDP2, DTensor TP/SP, PP, CP all work through stock `torch.distributed`,
  sharding over NeuronCores. Relevant to the training side of this repo (`gen-ai-training-testing`),
  and a plausible follow-on.

---

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| R-1 | Compile time consumes the session | Precompile first, persist cache to EFS, freeze configs before sweeping |
| R-2 | Both key components are **Beta**; APIs shift | Pin every version exactly; record `pip freeze` in results |
| R-3 | `nc_count` ambiguity breaks replica math (**O-1**) | Resolve in Phase 1 before any serving; set `NEURON_CORES_PER_DEVICE` explicitly |
| R-4 | `torch-neuronx` vs `libtorch-neuronx-lite` conflict | Separate venvs: standalone TorchNeuron vs serving image |
| R-5 | Benchmarks disagree | Treat as a blocker, not an average; `--ignore-eos` and fixed token shapes in both |
| R-6 | MXFP8 assumed available | It needs Trn3. BF16 only here. |
| R-7 | Block expires with artifacts on root EBS | Everything durable goes to EFS/S3 *during* the run, not at the end |
| R-8 | Folding into `launch.sh` breaks the legacy path | Done without editing `launch.sh`/compose; new engine is additive, gated to trn2/trn3 (**O-2 resolved**) |

## 8. Open items

- ~~**O-1** Does `neuron-ls --json-output` report `nc_count` as physical (8) or logical (4) per device at LNC2?~~
  **RESOLVED on hardware: logical (4).** LNC2 is the instance default; existing replica math is correct. See Section 3.
- ~~**O-2** When to fold into `launch.sh`/`compose/neuron.yaml` vs stay standalone.~~
  **RESOLVED 2026-11-16:** folded in as engine `vllm_native` with no change to `launch.sh` or
  compose (Phase 4).
- ~~**O-3** Is `Qwen/Qwen3-32B` gated? Confirm `HF_TOKEN` need and pin `BF16_REVISION` for reproducibility.~~
  **RESOLVED 2026-11-13: not gated.** Downloaded with no `HF_TOKEN` supplied. 62 GB on disk,
  **65,540,308,523 bytes / 84 objects / 17 safetensors shards + 1 index** staged to
  `s3://<lab-store-bucket>/lab/models/qwen-qwen3-32b`.
  (An earlier note said 18 shards; that was a miscount from `grep -c safetensors`, which also
  matched `model.safetensors.index.json`. The index itself declares 17, and the hydrated copy
  matches the S3 byte count exactly.)
  Pinning a revision SHA is still worth doing before publishing numbers; `main` was used here.
- **O-4** Exact `--optimization-level` semantics (recipe uses O2 single-server, O3 for disaggregated decode).
- ~~**O-5** Does a 1k-input shape need its own bucket config, given buckets are pinned at 8192?~~
  **RESOLVED from docs: yes.** The Neuron `torch.compile` backend forces `dynamic=False`, so
  every distinct shape compiles separately. Buckets must enumerate every shape the sweep will
  use. Handled by `SWEEP_MODE=1`, which is a separate fingerprint from the control config. See
  Section 2a.
- **O-6** Confirm the plugin's own `benchmarks/` directory contents — may supersede `bench/` here.
- ~~**O-7** Chunked prefill on Neuron only chunks at batch size 1 and cannot mix prefill with decode
  in a batch. Determine how this shapes the concurrency sweep before trusting any SLO number.~~
  **RESOLVED 2026-11-15 from plugin source** (`vllm_neuron/vllm/core/scheduler.py`,
  `NeuronScheduler`, release 0.29.0.1.2.0). See **Section 5a** for the consequences. In short:
  the warning is accurate, and on top of it **prefill pre-empts decode**. The sweep design stays,
  but how TTFT and ITL are read and reported changes.

## 5a. O-7: what prefill/decode separation does to a concurrency sweep

The warning in `platform.py` is only a log line. The behaviour lives in `NeuronScheduler.schedule()`:

1. **One prefill at a time, hard-coded.** `self.max_prefills_per_batch: int = 1`, and
   `can_schedule()` refuses a new prefill while any prefill is in `running`. A long prompt is
   chunked by `max_num_batched_tokens`, and while its chunks run every other prefill waits
   (branch 3.1).
2. **A step is either all-prefill or all-decode.** `_apply_padding_and_log_stats` raises
   `Neuron constraint violated: Mixed prefill ... and decode` if both appear.
3. **Prefill wins.** When a prefill can start (branch 3.2), the scheduler *hides every running
   decode* (`running_holdback = self.running; self.running = []`) for the whole prefill, all
   chunks included. Decode resumes only when no admissible prefill is waiting (branch 3.3).
4. **Padding to buckets.** A prefill is padded to the smallest `num_batched_tokens_buckets` entry
   that fits (`_calculate_padded_count`). With the control buckets `[8192]`, **a 1k or 4k prompt
   costs a full 8k prefill**. The earlier `SWEEP_MODE` buckets `[2048, 8192]` padded 1k→2k and
   4k→8k, so the "4k/256" cell measured an 8k prefill. `kv_segment_size_buckets` only matters for
   *segmented* prefill (prompt > one chunk), which none of the planned shapes (≤ 8192) trigger.
5. **Admission is capped at worst-case KV.** `_max_kv_concurrent` comes from
   `get_max_concurrency_for_kv_cache_config`, which assumes every request uses `max_model_len`
   tokens. The KV budget is `gpu_memory_utilization × per-core HBM × VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`
   (default 0.30, i.e. 6.62 GiB per logical core in the 2026-11-14 log). Qwen3-32B at TP8 stores
   1 KV head per rank: 64 layers × 2 × 128 × 2 bytes = 32 KiB/token/rank, which gives roughly 21
   concurrent requests at `max_model_len=10240`. **Above that cap, `max_num_seqs=32` does not
   give 32 running requests; the rest queue.** Read the real cap from the
   `Effective concurrency will be capped` warning at startup and do not report a cell above it as
   running at that concurrency.

### What this means for the numbers

- **TTFT at concurrency > 1 is mostly queueing, not compute.** TTFT = (wait for the prefill in
  flight, plus any queued ahead of it) + own padded prefill. Only concurrency 1 gives compute-only
  TTFT. That is still the latency a user sees, so it is a valid SLO input, but report
  `TTFT(c) - TTFT(1)` as the queueing share and do not read the curve as prefill slowing down.
- **The first N requests of every closed-loop cell are a burst artifact.** With
  `--request-rate inf --max-concurrency N`, N requests arrive together and are prefilled one
  after another, so the k-th waits about k prefills. `--num-warmups` does not help: warm-ups
  finish before the measured run starts. At the old `10 × N` prompts the burst is ~10% of
  samples, which is enough to set p90/p99 TTFT. Fix: `--save-detailed`, then compute
  **steady-state** percentiles with the first N and last N requests (ramp and drain) removed.
  Report raw and steady-state side by side.
- **p99 ITL measures prefill stalls, not decode speed.** Each new arrival freezes every running
  stream for one full prefill. In a closed loop a stream sees about (N−1) such stalls over its
  `OUT` tokens, so once (N−1)/OUT > 1%, **p99 ITL ≈ one padded prefill time**. For 4k/256 that
  happens from N≈4, for 1k/128 from N≈3, for 8k/1k from N≈12. p99 ITL is a step function, not a
  smooth curve, and an ITL-p99 target below the single-prefill time cannot be met above that N.
  Report **ITL p50** (decode-step time at the padded `num_seqs` bucket), **ITL p99** (stall) and
  **TPOT** (the mean, which spreads the stalls) separately.
  *Corrected by measurement (2026-11-15, control config, 4k/256):* the measured stall fraction was
  several times **below** (N−1)/OUT. A closed loop **phase-locks**: requests start together,
  finish at about the same time, and their replacements arrive together and are prefilled back to
  back. Each stream then sees about one *long* stall per wave (several prefills) instead of N−1
  short ones. So p99 ITL crosses into stall territory at a higher N than the estimate above, but
  the stall it eventually reports is longer. Open-loop (Poisson) arrivals do not phase-lock, so
  expect them to look closer to the estimate. Read the parser's `stall frac` column; don't
  predict it.
  **Consequence: p99 ITL is the wrong stall metric.** The `SWEEP_MODE` sweep (2026-11-15) kept the
  stall fraction under 1% at every in-cap concurrency, so p99 ITL stayed close to the decode-step
  time. Meanwhile the typical request's single longest gap grew to *seconds* at high
  concurrency, which is a visible freeze of the stream. The parser therefore reports
  `worst gap/req` (each request's longest gap, p50/p99 over requests) and accepts
  `--slo-maxgap-p99`. A streaming-UX SLO should bound that, not p99 ITL.
- **The "latency rises faster than throughput" knee always lands at concurrency 1 for TTFT.**
  Prefill is serialized, so TTFT grows roughly in proportion to N while throughput grows less than
  that. This describes the stack correctly; it is not a parser bug. TTFT SLOs therefore have to
  come from target feasibility (`--slo-ttft-p99`), not from the TTFT knee. The ITL/TPOT knee is the
  one that says something about capacity.
- **In the control config, c=1 is not a batch-1 measurement.** With `num_seqs_buckets [4]`, a
  single decode stream is padded to batch 4 and a 4k prompt to the 8k prefill bucket. The control
  "floor" is a floor *for that config*. A narrower `SWEEP_MODE` bucket set can sit lower.
- **Replicas do help ITL p99, unlike most latency limits.** The stall rate per stream depends on
  *per-replica* concurrency, so spreading load across replicas lowers it. They still cannot lower
  the concurrency-1 floor.
- **Decode is padded too.** A batch of 5 runs at the `num_seqs` bucket of 8, so ITL p50 changes in
  steps at bucket edges.

## 5b. Prefix caching makes TTFT worse on this stack, and contaminated the first cross-check

Found 2026-11-15 when the first Locust/`vllm bench` cross-check failed the 10% gate on median
TTFT at concurrency 1.

- **Cause of the disagreement: prompts were reused, not mis-instrumented.** `vllm bench serve
  --dataset-name random` is seeded (default `--seed 0`), so every run regenerates the *same*
  prompts, and cells with more prompts reuse the first ones of smaller cells. The old
  `locustfile.py` also defaulted to a fixed seed. vLLM 0.29 runs with
  `enable_prefix_caching=True` by default, so reused prompts were cache hits. The server's
  `Prefix cache hit rate` log line and the `vllm:prefix_cache_hits_total` counter show it. The
  exact same cell, rerun after the cache had turned over, agreed with Locust.
- **On Neuron a prefix-cache hit is *slower* than a fresh prefill** (control config, 4k prompts,
  measured with controlled A/B requests). The penalty was the same whether the cached prefix was
  half the prompt, most of it, or all of it. That fits the cached-prefix path dispatching a
  different compiled graph (segmented prefill over prior KV, Section 5a point 4) that costs more
  than one padded full prefill at this size. Not yet measured: other prompt lengths and the
  `SWEEP_MODE` buckets.
- **Production consequence:** with the control config, a workload with a shared system prompt
  gets *worse* TTFT with prefix caching on. Measure both settings with a realistic shared-prefix
  workload before deciding. `--no-enable-prefix-caching` may require its own compile.
- **Harness fixes:** `run_vllm_bench.sh` passes a unique `--seed` per cell, from a per-run
  `SEED_BASE` recorded in `run-metadata.txt`. It snapshots `prefix_cache_{queries,hits}_total` and
  `num_preemptions_total` around every cell (`<cell>.server.json`). `parse_results.py` marks any
  cell above 1% hits **CONTAMINATED** and leaves it out of knee and feasibility analysis.
  `locustfile.py` defaults its seed to the run's start time.
- **Prefix caching requires segmented prefill (found 2026-11-16).** The plugin serves a
  prefix-cache hit only through segmented prefill, and refuses to start without it:
  `ValueError: Automatic Prefix Caching (APC) requires segmented prefill to be enabled`. It turns
  segmented prefill on by default only when `max_num_batched_tokens < max_model_len`. So the
  2026-11-15 sweep (10240 > 8192) worked, while the notebook's 8192/8192 failed.
  `openai_server/vllm_native/run.sh` now always sets `kv_segment_size_buckets` when
  `PREFIX_CACHING=true`. This also supports the earlier explanation of the slower hit path.
- **Data affected:** the 2026-11-15 control-config narrow run (cells c=2 and c=4) and the first
  cross-check run. Both are marked on the lab store and not used for SLOs.

### Sweep corrections applied

- `SWEEP_MODE=1` buckets: `num_batched_tokens_buckets` = `[1024, 4096, 8192]` so each planned
  shape prefills **unpadded**, and `kv_segment_size_buckets` = `[8192]` (the plugin's diagonal
  default). Off-diagonal segment buckets only add compile work for these shapes, and the plugin
  documents that too many targets can hit `Shared compilation timeout after 600s`.
- `bench/run_vllm_bench.sh` passes `--save-detailed`. `bench/parse_results.py` reports
  steady-state percentiles next to the raw ones, along with the queueing share of TTFT.
- Fixed request-rate (open-loop) runs are still worth doing for steady-state SLOs, since a closed
  loop of N users is not Poisson arrivals.
- **O-8** The compile cache is written rank-less (`topo_none`), and the runtime warns a warm
  restore "can stop with an error on a process group name". ~~Set `NEURON_RT_VISIBLE_CORES`~~ (rejected
  by the plugin, see Section 2b correction) — set `NEURON_VISIBLE_DEVICES` or `RANK`,
  or init `torch.distributed` before first compile, then re-verify a cross-instance warm restore.
  *Narrowed 2026-11-16:* a cross-container reuse failure, reproduced. The NKI kernel binaries
  that compiled graphs reference live in `/var/tmp/nki-intermediate-cache` by default, inside the
  container, while the compile cache lives on `/cache`. After `launch.sh down`/`up`, a start that
  reuses cached graphs but must compile a *new* shape (prefix caching on, i.e. segmented-prefill
  graphs) failed in `neuronx-cc` with `NCC_EVRF059 Kernel file ... does not exist on the host`.
  Warm restarts that compile nothing never hit it. Fix: `NKI_COMPILE_CACHE_URL=/cache/nki` (image
  and run.sh), so kernels persist with the graphs. A labstore cache copy must include `nki/`.
  *Narrowed 2026-11-15:* with `NEURON_VISIBLE_DEVICES` set, the TP **workers** write per-rank
  packages (`dp_0/rank_N`, `topo_<hash>`) and a warm restart in a fresh container hit the cache
  (0 artifacts saved). The `topo_none` warning still appears, but it comes from the **EngineCore**
  process (no rank), not from a worker. Still open: whether `topo_<hash>`, which embeds the core
  ids, lets a replica on other cores reuse the cache (multi-replica check), and a restore on a
  different instance.
  This is the assumption `labstore` rests on.
