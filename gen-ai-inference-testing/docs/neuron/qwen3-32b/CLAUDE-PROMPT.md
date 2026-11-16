# Claude Code prompt — Qwen3 on Trainium2 with vLLM + TorchNeuron

> **Paths in this document are historical (2026-11-16 restructure).** The work below was done under
> `gen-ai-inference-testing/trn2-qwen3-torchneuron/`, which no longer exists. Current locations:
>
> | Then | Now |
> |---|---|
> | engine `vllm_native`, `containers/openai_server-vllm_native-neuron/`, `openai_server/vllm_native/run.sh` | the existing `vllm` engine: `containers/openai_server-vllm-neuron/` builds the **native** stack on trn2/trn3 (`scripts/openai_server-vllm-neuron.sh`), `openai_server/vllm/neuron_native.sh` configures it |
> | `serve/run_qwen3_32b.sh`, `serve/precompile.sh`, `serve/launch_qwen3_32b.sh` (`CONFIG=recipe\|sweep`) | `bench/serve_neuron.sh` with `config/neuron_profiles/qwen3-32b-trn2-{recipe,sweep}.env` |
> | `SWEEP_MODE=1` | the `-sweep` profile |
> | `$HOME/scripts/vllm_native.env` | `$HOME/scripts/neuron_native.env` |
> | `bench/*` | top-level `bench/` (engine-agnostic) |
> | `bootstrap.sh`, `ops/`, `serve/download_model.sh` | `labstore/bootstrap/` |
> | `LAB-STORE-PLAN.md` | removed; its rationale is in `labstore/README.md` |
> | `notebooks/qwen3_32b_trn2_vllm.ipynb` | removed; use `locust_decoder.ipynb` (`openai_server` + `vllm`) |
>
> Engine-level findings are summarised in `docs/neuron/README.md`. Measured numbers stay on the lab
> store, not in git.
> In the prompt below, task G's "Include the raw results JSON" was later **overridden**: measured
> numbers stay out of git.

For the **Claude Code CLI running on the trn2 desktop itself**. That agent starts with no
knowledge of how the instance was set up, so the prompt below is self-contained: it states what is
already done, what must not be redone, and the non-obvious settings that took real debugging time
to find.

Paste it into `claude` with this repo as the working directory.

---

## Prompt

```
You are working on a trn2.48xlarge (Trainium2) AWS Deep Learning Desktop. Your job is to
benchmark Qwen3-32B served by vLLM on the native TorchNeuron backend, and derive defensible
SLOs from the results.

Read these first, in order, and tell me your understanding before you change anything:
  gen-ai-inference-testing/trn2-qwen3-torchneuron/PLAN.md       (esp. sections 2a, 2b, 3, 5)
  gen-ai-inference-testing/trn2-qwen3-torchneuron/README.md
  gen-ai-inference-testing/labstore/README.md

== ALREADY DONE - do not redo ==
- Qwen3-32B weights are on local NVMe at /mnt/nvme/lab/models/qwen-qwen3-32b
  (62 GB, 17 safetensors shards + index). Also in S3 at
  s3://<lab-store-bucket>/lab/models/qwen-qwen3-32b
- Container image openai_server-vllm_native:neuron is built locally and in ECR as
  <lab-store ECR repository>:qwen3-trn2-validated
- The NEFF compile cache is WARM at
  /home/ubuntu/efs/lab/cache/qwen-qwen3-32b/neuron-trainium2/86b6d4b6e8da/tp8-lnc2
  A cold compile takes about 5 minutes; a warm start is under a minute.
- Serving was already verified end to end: vllm-0.29.0-tp8 returned a real
  completion. Do not treat "does it work at all" as an open question.
- Local NVMe is RAID0 across 4 instance-store disks, 7.0 TB, mounted at /mnt/nvme.
  Do NOT run labstore nvme-setup again; it is destructive and already done.

== HARD CONSTRAINTS ==
1. NEURON_SKIP_EFA_AFFINITY=1 is MANDATORY. Without it all 8 TP workers die at startup with
   "No EFA device found at /sys/bus/pci/devices/.../infiniband". serve/run_qwen3_32b.sh sets
   it by default; do not remove it. (EFA affinity is a CPU optimisation, not a correctness
   requirement, and the container cannot see the InfiniBand sysfs nodes.)
2. Run everything as the ubuntu user. Do NOT use sudo for anything touching /home/ubuntu/efs.
   A previous instance ran as root and left root-owned directories on the shared EFS that
   blocked writes.
3. Always publish the port: docker run needs -p 8000:8000. Without it the server is reachable
   only via docker exec and looks identical to a failed startup from the host.
4. Neuron's torch.compile forces dynamic=False, so EVERY distinct input shape needs its own
   compiled bucket. The default config is the published AWS control recipe with
   num_seqs_buckets [4], valid for concurrency <= 4 ONLY. For a wider sweep you must start the
   server with SWEEP_MODE=1, which widens the buckets so one compile covers concurrency 1-32.
   Mixing the two gives mid-run recompiles and meaningless latency numbers.
5. Save results to /home/ubuntu/efs or S3 AS YOU GO, not at the end. The instance is
   time-limited and its root volume is deleted when it is terminated. /home/ubuntu/efs and S3
   survive; everything else does not.
6. Do not modify launch.sh, compose/*.yaml, or either deep-learning-ubuntu-desktop*.yaml.
   Those are the existing shared harness and must keep working unchanged for GPU users.
7. Do not delete or modify the instance's reservation, the lab store stack, or the EFS.

== TASKS ==

A. RESOLVE OPEN ITEM O-7 BEFORE TRUSTING ANY NUMBER.
   vLLM logs this on Neuron:
     "Chunked prefill is enabled on Neuron platform. Currently Neuron only supports chunking
      prefills with batch size of 1. Mixing prefill and decode in the same batch is not
      supported."
   Work out what that means for a concurrency sweep: does TTFT at concurrency > 1 reflect
   queueing rather than compute? Check the vLLM Neuron docs and the plugin source, and write
   your finding into PLAN.md under O-7. If it invalidates the sweep design, say so and propose
   a corrected one rather than producing numbers you cannot defend.

B. Start the server and confirm correctness before measuring anything.
   Use serve/run_qwen3_32b.sh inside the container, mounting the model read-only and the warm
   cache read-write. Then: /health, a known-answer completion ("The capital of France is" must
   mention Paris), and 16 concurrent requests all returning non-empty. Note that Qwen3 is a
   hybrid-reasoning model and emits <think> blocks, so match on content rather than exact text.

C. Run the SLO sweep with bench/run_vllm_bench.sh.
   Start narrow to prove the harness (SHAPES="4096:256" CONCURRENCIES="1 2 4"), then widen.
   --ignore-eos is mandatory: without it output length depends on the model's stopping
   behaviour and ITL stops being comparable across cells. Put RESULTS_DIR under
   /home/ubuntu/efs/lab/results/<timestamp>.

D. Derive the SLOs with bench/parse_results.py.
   SLO feasibility: --slo-ttft-p99 and --slo-itl-p99 for targets you want to test.
   An SLO is only reusable with its token shape, concurrency, replica count and dtype attached.
   State all five with every number.

E. Cross-validate with bench/locustfile.py.
   If median TTFT or throughput disagree with vllm bench serve by more than ~10%, treat that as
   a BLOCKER and find out which harness is wrong. Do not average them.

F. Multi-replica check.
   TP8 at LNC2 uses 8 of 64 logical cores, so 8 replicas fit on this instance. Re-run at 2 and
   4 replicas (ports 8000/8001/...) to confirm the per-replica SLO survives contention. A
   single-replica number does not predict EKS behaviour.

G. Commit as you go to branch trn2-qwen3-torchneuron-vllm and push.
   Commit after each working step, not in one batch at the end. Include the raw results JSON,
   the derived SLO tables, and your O-7 finding.

== REPORTING ==
Tell me, explicitly:
  - the latency floor at concurrency 1 (no SLO can beat it)
  - where the knee is, and which metric binds first
  - any configuration that is INFEASIBLE for a target, and whether replicas would help
    (they add throughput, not single-request speed, so they cannot fix a latency-floor miss)
  - anything that contradicts PLAN.md, which was largely written before hardware access

If a step fails, show me the actual error and your diagnosis before trying a fix. Do not
silently work around failures. Prefer saying "I do not know yet" over a confident guess.
```

---

## Notes for the human

- **Open items** O-2, O-4, O-6, O-7, O-8 are listed at the end of `PLAN.md`. O-7 and O-8 both
  came from runtime warnings on the first real run and are the two most likely to change
  conclusions. O-8 (the rank-less compile cache) matters for *future* sessions rather than this one.
- **Useful numbers measured on the first run:** HuggingFace download ~112 MB/s vs S3 to local NVMe
  ~538 MB/s; cold NEFF compile ~5 min producing 56 files / 371 MB.
- **If DCV logins fail**, the template leaves the `ubuntu` account locked. Set a password over
  SSM: `aws ssm start-session --target <instance-id> --region <region>`, then `sudo passwd ubuntu`.
