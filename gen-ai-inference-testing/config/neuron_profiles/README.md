# Neuron serving profiles

Optional per-model settings for `bench/serve_neuron.sh`, the scripted path to the same server the
notebooks start (`openai_server` + `vllm` on Neuron through `launch.sh`). A model needs a profile only
when the generic defaults are not what you want, e.g. to reproduce a published AWS model recipe or to
pin a benchmark configuration. The notebooks do not use profiles; they set the same values as
parameters.

A profile is a shell fragment. Use the `${VAR:-value}` form so values given on the command line win.

| Variable | Meaning |
|---|---|
| `MODEL_ID` or `MODEL_DIR` | Hugging Face id, or a checkpoint directory under `$HOME/snapshots` |
| `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS` | passed to the server through `compose/neuron.yaml` |
| `NEURON_VLLM_STACK` | `auto` (default), `native` or `legacy` |
| `BUCKET_PROFILE`, `PREFIX_CACHING`, `NUM_REPLICAS`, `MAX_NUM_BATCHED_TOKENS` | native stack only; see `openai_server/vllm/neuron_native.sh` |

Name profiles `<model>-<instance family>-<purpose>.env`. Settings that hold on trn2 are not
automatically right on trn3 (different HBM per device, so possibly a different TP), so a trn3 profile
is a separate file.
