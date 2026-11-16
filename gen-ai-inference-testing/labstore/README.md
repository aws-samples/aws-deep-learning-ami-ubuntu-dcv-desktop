# labstore — durable session store for ephemeral accelerator instances

## Why

Accelerator instances are often short-lived: they are launched for a session and terminated
afterwards, and the root volume goes with them. Downloaded weights, compiled artifacts (for example
Neuron NEFFs or TensorRT engines) and benchmark results are slow to recreate. Without somewhere
durable to keep them, every new instance starts from scratch: a large download and a cold compile
before anything can be served.

`labstore` keeps those artifacts somewhere that outlives the host and restores them onto a fresh
instance in minutes, so a new instance reaches "serving a model" quickly.

It works for **GPU and Neuron alike** and is **entirely opt-in**: nothing here runs unless you run
it, and the existing harness (`launch.sh`, `compose/`) is not modified.

## Three tiers

Putting an artifact in the wrong tier is the main way this gets *slower* rather than faster.

| Tier | Holds | Why |
|---|---|---|
| **S3** | weights, cache backups, results | durable source of truth, fast parallel fan-out |
| **EFS** (Elastic) | compile caches, manifests, results | instantly mounted, no hydration step |
| **Local NVMe** | weights during a session | fastest for large sequential reads; instance-store, so it does not survive termination |

Weights do **not** belong on EFS: bursting-mode throughput scales with *stored* data, and local
NVMe beats NFS for large sequential reads.

## Commands

```bash
labstore.sh nvme-setup     # RAID0 + mount instance-store NVMe (idempotent, needs sudo)
labstore.sh init           # create the store layout, write a provenance snapshot
labstore.sh status         # what is warm vs cold -- and WHY, if cold
labstore.sh push-weights   # local checkpoint -> S3 (run on a general-purpose instance)
labstore.sh resume         # restore compile cache + hydrate weights to local
labstore.sh save           # cache + manifest + results -> EFS and S3
```

`provenance.py` can be used on its own:

```bash
python3 provenance.py --model-id Qwen/Qwen3-32B --topology tp8-lnc2
python3 provenance.py --key cache      # just the cache key, for shell use
python3 provenance.py --json           # full manifest
```

## Why caches are keyed, not named

A compiled artifact is reusable only where the *relevant* environment matches. A Neuron NEFF
invalidates on Neuron SDK version, plugin version, torch version, LNC and TP degree; a CUDA
TensorRT engine invalidates on GPU model and driver. None of that is visible in a filename.

So entries are keyed by a **stack hash** over exactly those facts:

```
cache/<model-slug>/<accel-class>-<accel-model>/<stack-hash>/<topology>/
models/<model-slug>/                                  # no accelerator fields
```

Consequences, each verified by test:

- **Weights are shared across hardware** — `models/qwen-qwen3-32b` is identical on a GPU box and
  a Neuron box, so one download serves both.
- **Compile caches cannot collide** — `neuron-trainium2/...` and `cuda-h100/...` are distinct keys.
- **Host size does not invalidate** — `device_count` is excluded from the hash, so a TP8 cache
  built on a 16-device host still hits on a 4-device host. The parallelism degree lives in
  `topology`, which *is* hashed.
- **A version bump invalidates, and says so** — `status` diffs the stored `MANIFEST.json` against
  the live machine and names the field:
  ```
  frameworks.vllm-neuron: cached='0.29.0.1.2.0' live='0.29.0.1.3.0'
  ```
  A silent cache miss means a full recompile and looks like a mystery; this is the whole point of
  the manifest.

## Typical use

**Before an accelerator session, on a general-purpose instance** — takes the download off the
accelerator instance entirely:
```bash
MODEL_ID=Qwen/Qwen3-32B labstore.sh push-weights
```

**On a fresh accelerator instance:**
```bash
sudo -E labstore.sh nvme-setup
labstore.sh init
eval "$(labstore.sh resume | grep ^export)"   # sets VLLM_CACHE_ROOT and MODEL_PATH
# ... run the workload ...
RESULTS_DIR=./results labstore.sh save
```

`bootstrap/` automates this end to end for a Neuron desktop (see below).

## Bridging the existing harness (optional)

`launch.sh` and `compose/{cuda,neuron}.yaml` mount `$HOME/snapshots` and `$HOME/cache`, both of
which live on the root EBS volume — so nothing persists. `link-harness.sh` redirects them into the
store by symlink, **without editing any harness file**:

```bash
link-harness.sh link      # $HOME/snapshots and $HOME/cache -> store
link-harness.sh status
link-harness.sh unlink    # revert to plain local directories
```

It migrates (never deletes) existing contents, preserving the original as
`*.pre-labstore.<timestamp>`. `$HOME/snapshots` maps to a shared location since the HF cache is
hardware-independent; `$HOME/cache` maps to a per-accelerator subdirectory so a NEFF is never
handed to a CUDA runtime.

Two cautions: EFS is slower than local disk for large sequential reads, so for heavy weight loading
prefer `labstore resume`, which hydrates to NVMe. And the server container runs as root under
`compose/`, so a `$HOME/cache` on EFS collects root-owned files that other users cannot write to;
keep the live cache local and copy it to the store with `labstore save`.

## Infrastructure

Provisioned by `lab-store.yaml` at the repo root (default stack name `dl-lab-store`), which is
permanent and independent of any desktop stack. It creates an EFS file system (Elastic, encrypted,
with a mount target in each AZ you give it), optionally an S3 bucket and an ECR repository, and
exports them as `<stack>-EfsId`, `<stack>-BucketName` and `<stack>-EcrRepositoryUri`.

Nothing account-specific is kept in this repo. The scripts resolve these at runtime
(`aws-defaults.sh`): an explicit `LAB_S3_BUCKET` / `LAB_EFS_ID` / `LAB_ECR_URI` wins, otherwise the
store stack's exports, otherwise (bucket only) the convention `dl-lab-store-<account>-<region>`.
The CloudFormation templates here take a `LabStoreStackName` parameter and import the same exports
when their bucket/EFS/ECR parameters are left blank.

Desktop stacks consume the store by passing `EFSFileSystemId` to the unmodified
`deep-learning-ubuntu-desktop.yaml`. Create mount targets in every AZ a desktop may be launched in;
an AZ without one cannot reach the store.

## Contents

| Path | Purpose |
|---|---|
| `labstore.sh`, `provenance.py`, `aws-defaults.sh` | the store CLI, cache keys and manifests, runtime lookups |
| `link-harness.sh` | optional `$HOME/snapshots` / `$HOME/cache` redirection |
| `prestage-instance.yaml` | general-purpose instance that pushes a checkpoint to S3 |
| `validate-instance.yaml` | non-accelerator instance that validates the workflow and pushes a known-good image to ECR |
| `bootstrap/` | unattended pre-staging of a Neuron desktop: launch into a capacity reservation (`launch_and_bootstrap.sh`, `remote-launcher.yaml`, `preflight.sh`, `verify_ready.sh`), then on the instance restore, build, compile, serve and validate (`bootstrap.sh`, `download_model.sh`) |
