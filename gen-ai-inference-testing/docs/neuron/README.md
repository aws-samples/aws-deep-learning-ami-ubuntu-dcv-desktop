# vLLM on AWS Neuron: design and the native TorchNeuron stack

This document explains how `openai_server` + `vllm` runs on AWS Neuron in this repository, how the
**native TorchNeuron** stack works underneath, and what that means on Trainium2 (trn2) and, later,
Trainium3 (trn3).

## 1. Design: one engine, two stacks

On Neuron, `openai_server` + `vllm` is a single engine (`INFERENCE_ENGINE=vllm`, image
`openai_server-vllm:neuron`) that can be built with one of two software stacks:

| | **native** | **legacy** |
|---|---|---|
| Software | vLLM 0.29 + `vllm-neuron` plugin on **TorchNeuron** (Neuron SDK 2.33 DLC) | earlier vLLM DLC (Neuron SDK 2.28): torch-neuronx / PyTorch-XLA with NxD Inference |
| Hardware | trn2, trn3 (the only hardware Neuron supports this stack on) | inf2, trn1 (also runs on trn2/trn3) |
| Selected | `auto` on trn2/trn3 for **decoder** tasks | `auto` everywhere else, and for **encoder** tasks on every instance |

`scripts/openai_server-vllm-neuron.sh` makes the choice at build time from the instance type
(`NEURON_VLLM_STACK=auto|native|legacy` overrides it; native with an encoder task is refused). The
image records its stack in `/opt/program/neuron-stack` and the label
`org.opencontainers.image.neuron-vllm-stack`.

```
launch.sh up
  └─ scripts/openai_server-vllm-neuron.sh        picks the stack, builds openai_server-vllm:neuron
  └─ compose/neuron.yaml  →  /scripts/run.sh     (openai_server/vllm/, shared with CUDA)
        ├─ legacy image  →  existing Neuron setup in run.sh (unchanged)
        └─ native image  →  /scripts/neuron_native.sh   native env, buckets, prefix caching
                               └─ /opt/program/serve    one vLLM per core group, nginx on :8080
```

Nothing in `launch.sh`, `compose/` or the CUDA paths changes. Native-only settings, which
`compose/neuron.yaml` has no slot for, go in `$HOME/scripts/neuron_native.env` (written by the
notebooks' Neuron cell or `bench/serve_neuron.sh`; parsed, not executed):

| Setting | Default | Meaning |
|---|---|---|
| `PREFIX_CACHING` | `false` | vLLM automatic prefix caching (see section 4.4) |
| `NUM_REPLICAS` | all | number of servers, one per `TENSOR_PARALLEL_SIZE` logical cores |
| `BUCKET_PROFILE` | `ladder` | compiled shapes: `ladder` = batch 1, 2, 4, ... and prefill 1024 ... 8192; `recipe` = one each |
| `MAX_NUM_BATCHED_TOKENS` | min(8192, `MAX_MODEL_LEN`) | prefill chunk size |
| `OPT_LEVEL`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, `OMP_NUM_THREADS`, `NGINX_TIMEOUT_S` | 2, 1200, 16, 1800 | tuning defaults |

Any other line in the file is logged and ignored. `launch.sh` never cleans `$HOME/scripts`, so a
direct `launch.sh up` reuses the last file; the server log prints the values applied and the file's
timestamp.

## 2. The native stack, layer by layer

```
 HTTP (OpenAI API)     nginx :8080  ──►  vLLM API server (one per replica)
 Serving engine        vLLM 0.29 engine + vllm-neuron platform plugin
                         ├─ NeuronScheduler       (Neuron-specific batching rules, section 4)
                         ├─ Neuron model runner   (bucket padding, KV cache layout)
                         └─ model code in the plugin: Llama 3, Qwen3, Qwen3-VL, Qwen3-ASR, GPT-OSS
 Framework             PyTorch + TorchNeuron     (device "neuron" via PyTorch's PrivateUse1 backend)
 Compilation           torch.compile(backend="neuron")
                         TorchDynamo → FX graph → AOTAutograd → Neuron FX passes
                         → StableHLO (no TorchInductor) → neuronx-cc → NEFF
 Kernels               NKI kernels (attention, MLP, ...) compiled alongside the graphs
 Runtime / hardware    Neuron Runtime → NeuronCores on Trainium devices
```

- **TorchNeuron** is a native PyTorch backend: tensors live on device `"neuron"`, and code written for
  `cuda` mostly ports by changing the device. It provides eager mode and `torch.compile`; only
  `mode="default"` is supported. vLLM does not reimplement any of this; the plugin compiles each model
  *through* TorchNeuron's `torch.compile` backend.
- **The plugin** (`vllm-neuron`) registers Neuron as a vLLM platform and carries its own model
  implementations. NxD Inference is not a dependency. It ships with its own build of TorchNeuron's
  compile components (`libtorch-neuronx-lite`), so the serving image must not also install
  `torch-neuronx`.
- **Two lowering routes exist** (`VLLM_NEURON_BACKEND`): the native route (default; this repo sets
  `neuron_native` explicitly because the precompile cache only activates when it is set) and a
  deprecated XLA route. Both compile the same graphs onto the same runtime.
- **The legacy stack** differs at every layer below the API: PyTorch-XLA graph tracing, NxD Inference
  model classes, an older SDK. It supports decoder families only (Llama, Mistral, Mixtral, Qwen2/3,
  DBRX, GPT-OSS and vision variants).

## 3. Compilation and caching

**No dynamic shapes.** The Neuron `torch.compile` backend forces `dynamic=False`: every distinct input
shape is a separately compiled graph (NEFF). The plugin therefore compiles a fixed set of **buckets**,
pairs of (prefill length, batch size), and pads each request up to the next bucket. Buckets decide
both the compile time and how much padding requests pay; a shape outside them is never served without
a recompile. `BUCKET_PROFILE=ladder` compiles several sizes so short prompts and small batches are not
padded to the largest shape; `recipe` compiles one, as in AWS's published model recipes.

**Three caches**, all of which must persist together on `/cache` (`$HOME/cache` on the host):

| Cache | Contents | Keyed by | Notes |
|---|---|---|---|
| NEFF cache | compiled graphs | lowered HLO + compiler flags (content-addressed) | reusable across instances with the same stack |
| Precompile package | Dynamo artefacts for a warm restart | model, config and TP rank layout | invalidated by any version or config change |
| NKI kernel cache | compiled kernel binaries the graphs reference | kernel source and parameters | defaults to the container's `/var/tmp`; `neuron_native.sh` moves it to `/cache/nki` |

If the NKI cache is not kept with the others, a later container that reuses cached graphs but has to
compile a new shape fails in `neuronx-cc` (`NCC_EVRF059 Kernel file ... does not exist on the host`).

The first start of a configuration compiles; later starts with the same settings are warm. Changing
TP, lengths, batch size, buckets, prefix caching or any version means a new compile. Replicas that
share a cache start one after another (`serve` does this), because later replicas rewrite the same
package files.

## 4. Runtime behaviour on the native stack

These follow from the plugin's design, not from a particular model.

### 4.1 One prefill at a time, and prefill pauses decode

The plugin's scheduler (`NeuronScheduler`) runs **at most one prefill per step**, never mixes prefill
and decode in one step, and while a prefill is waiting **pauses every running decode** until that
prefill (all its chunks) completes. Consequences:

- **Time to first token at concurrency > 1 is mostly queueing.** A new request waits for the prefill
  in progress (and any queued ahead) before its own. Only concurrency 1 shows pure compute TTFT.
- **Streams freeze while other requests prefill.** Each new arrival stops every running stream for one
  prefill. In a closed loop requests tend to arrive in waves, so a stream sees about one long freeze
  rather than many short ones, which percentile inter-token latency across all gaps does not show.
  Measure each request's longest gap (`bench/parse_results.py` reports it).
- **The first requests of a closed-loop benchmark arrive together** and are prefilled one by one, so
  raw percentiles include that ramp; `bench/` trims the ramp and drain for steady-state figures.

### 4.2 Padding

A prefill is padded up to the smallest `num_batched_tokens` bucket that fits, and a decode batch up to
the next `num_seqs` bucket. A configuration with a single 8k bucket processes a 1k prompt as 8k.

### 4.3 Admission is capped by worst-case KV

The scheduler admits at most as many requests as fit in the KV cache at `max_model_len` each
(`kv_cache_max_concurrency` in the server's `vllm:cache_config_info` metric). `MAX_NUM_SEQS` above
that cap is not more concurrency: the excess only queues, and a decode batch padded to a larger bucket
can make things slower. Size `MAX_NUM_SEQS` and `MAX_MODEL_LEN` together.

### 4.4 Prefix caching

The plugin serves a prefix-cache hit only through **segmented prefill** (prefill over KV that is
already cached) and refuses to start with prefix caching on otherwise; `neuron_native.sh` adds the
segment buckets when `PREFIX_CACHING=true`. A hit takes a different compiled path than a fresh
prefill and was observed to be slower to first token for short prompts, so prefix caching is off by
default; measure with real traffic before enabling it. Benchmarks must use unique prompts per run, or
cached prefixes from earlier runs change the results (`bench/run_vllm_bench.sh` seeds every cell).

### 4.5 Encoder models

- Decoder-backbone **embedding** models can run on the native stack: the plugin converts a supported
  decoder into a pooling model (for example Qwen3-Embedding with `--runner pooling`). The repo does not
  wire this up yet; encoder tasks keep the legacy stack.
- **Classification and reranking** are not implemented in the plugin, and encoder-only architectures
  (BERT, RoBERTa, XLM-RoBERTa) have no Neuron implementation in either stack. Use
  `triton_inference_server` + `python` for those on Neuron.

## 5. Running in containers

- **`NEURON_SKIP_EFA_AFFINITY=1` is mandatory.** The runtime tries to map each Neuron device to its
  EFA interface; inside a container the InfiniBand sysfs entries are not visible and every TP worker
  fails at start ("No EFA device found"). EFA affinity is a CPU-placement optimisation, not needed for
  single-node inference. Revisit for multi-node serving, where EFA is required.
- **Pin cores with `NEURON_VISIBLE_DEVICES`**, a list of logical core ids. The plugin rejects
  `NEURON_RT_VISIBLE_CORES` under vLLM's multiprocess executor and otherwise places every server on
  cores `0..TP-1`.
- **`PJRT_DEVICE` must not be `NEURON`** on the native stack (that selects the XLA route).
- **Root-owned caches.** `compose/neuron.yaml` runs the server as root, so files it writes into
  `$HOME/cache` or the Hugging Face cache under `$HOME/snapshots` are root-owned. Keep `$HOME/cache`
  on local disk and copy it to shared storage as your own user (`bench/serve_neuron.sh` refuses an
  EFS-backed cache). Download models before `launch.sh up` (the notebooks do), so the container does
  not create the Hugging Face cache as root.

## 6. Trainium2

| Property | trn2.48xlarge |
|---|---|
| Devices | 16 Trainium2 devices, 8 physical NeuronCores each (128 total) |
| Logical NeuronCore config (LNC) | **2 by default**: two physical cores form one logical core, so 4 logical cores per device and **64 per instance** (`neuron-ls` reports logical cores) |
| Memory | 96 GiB HBM per device |
| Data type | BF16 (the FP8/MXFP8 options in AWS's recipes are listed for Trn3) |

**Tensor parallelism and replicas.** `TENSOR_PARALLEL_SIZE` counts logical cores. TP 8 uses two
devices, so one instance holds 8 independent replicas, each behind nginx; `serve` starts them one
after another. A larger model needs a higher TP (fewer replicas); a small one can use a lower TP for
more replicas, as long as the KV cache at `MAX_MODEL_LEN` still fits the concurrency you need
(section 4.3).

## 7. Trainium3

The design already routes trn3 to the native stack (`auto` → native; `neuron_native.sh` accepts
`trn3*`), and the plugin carries trn3-specific code: separate instance-family definitions for its EFA
mapping, and expert-parallel collectives that require trn3's NeuronSwitch topology. **Nothing has been
run on trn3 in this repository yet.** Expect to revisit, per model:

- **LNC and logical cores per device**: read from `neuron-ls` at start (`run.sh` already does), but the
  right TP may differ because HBM per device differs.
- **Quantization**: AWS's recipes list MXFP8 for Trn3. `neuron_native.sh` fixes the dtype to BF16 and
  has no quantization setting yet; adding one is planned together with trn3 testing.
- **Kernels and buckets**: the plugin ships architecture-specific NKI kernels; compile caches are not
  shared between trn2 and trn3 (the stack hash and NEFFs differ).
- **Container settings** (section 5) should carry over; verify EFA behaviour on the target instance.

## 8. Where the code is

```
scripts/openai_server-vllm-neuron.sh          stack selection and image build
containers/openai_server-vllm-neuron/
  Dockerfile                                  both stacks (build args: base image, stack)
  Dockerfile.native-from-source               reference full native build (not yet built)
  resources/serve                             replicas + nginx; per-stack core pinning
openai_server/vllm/run.sh                     shared CUDA/Neuron entry; hands native images to:
openai_server/vllm/neuron_native.sh           native settings, buckets, prefix caching, environment
bench/                                        SLO measurement tools (see bench/README.md)
```
