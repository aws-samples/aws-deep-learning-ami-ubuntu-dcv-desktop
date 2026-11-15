# Megatron Parameter-Efficient Fine-Tuning (PEFT) with NeMo Megatron-Bridge

This example fine-tunes Large Language Models with LoRA/DoRA on Megatron, using NVIDIA
**NeMo Megatron-Bridge**. It provides a generalized data pipeline for HuggingFace datasets
and streamlined configuration for distributed training.

> **Requires a GPU of compute capability sm90 or newer** (Hopper H100/H200, or Blackwell).
> The NeMo 26.x containers are built for those architectures: Megatron Core computes a
> gradient norm every optimizer step through Transformer Engine's `multi_tensor_apply`,
> and that build ships no kernel for older GPUs such as the Ada-class L40S (sm89).
> `peft_megatron.py` checks this up front and exits with an explanation. For Ada or
> earlier, use the [accelerate](../accelerate/), [ptl](../ptl/) or
> [ray_train](../ray_train/) examples, which have no such requirement.

> **Migrated from NeMo 2.0.** NeMo 25.11 split the framework by domain: the `nemo`
> package became NeMo Speech and LLM training moved to Megatron-Bridge, so
> `nemo.collections.llm`, `nemo.lightning` and `nemo_run.Experiment` no longer exist and
> `nemo:25.09` was the last container carrying them. NVIDIA publishes no migration guide,
> so this example was rewritten rather than ported. If you are returning to it, the
> changes that matter are: launch with `torchrun`, pass a **local** model directory, and
> prepare datasets with `dataset_module.py`.

## Features

- **Generalized HuggingFace Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates and field mapping
- **Distributed Training**: Multi-node, multi-GPU training with tensor and pipeline parallelism
- **PEFT Methods**: Support for LoRA and other parameter-efficient fine-tuning schemes
- **Automatic Data Conversion**: Converts HuggingFace datasets to the prompt/completion JSONL layout Megatron-Bridge SFT reads
- **Customizable Training**: Extensive configuration options for hyperparameters, callbacks, and training strategies
- **Docker Support**: Containerized environment for reproducible training

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Different Models](#training-different-models)
- [Using Different Datasets](#using-different-datasets)
- [GPU Requirements](#gpu-requirements)
- [Configuration](#configuration)
- [Testing and Converting Checkpoints](#testing-and-converting-checkpoints)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Monitoring and Profiling](#monitoring-and-profiling)
## Prerequisites

Follow [Step by Step Tutorial](../../README.md) to launch a Deep Learning Desktop. On the desktop,

```bash
cd ~
git clone https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop.git
cd ~/aws-deep-learning-ami-ubuntu-dcv-desktop/gen-ai-training-testing/nemo2
```

## Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker buildx build -t nemo2:latest -f ../containers/Dockerfile.nemo2 .
```

2. Run the container with GPU support:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v $HOME/snapshots:/snapshots:ro \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=32g \
  nemo2:latest
```

The model directory is mounted because Megatron-Bridge loads the base weights from a
**local** path (see Quick Start). The `--ipc=host` and `--ulimit` flags are what NVIDIA
recommends for these images; without them PyTorch warns that the 64 MB default shared
memory limit may be insufficient.

## Quick Start

Megatron-Bridge loads base weights from a local directory, so download the model first:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3-8B', local_dir='$HOME/snapshots/huggingface/Qwen/Qwen3-8B')"
```

Then launch LoRA fine-tuning with `torchrun`, one process per GPU:

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
    --hf_model_id /snapshots/huggingface/Qwen/Qwen3-8B
```

This will:
1. Resolve the Megatron-Bridge recipe for the model (`qwen3_8b_peft_config`)
2. Load the base weights straight from the local HuggingFace snapshot -- no separate
   import or conversion step is needed
3. Fetch and prepare the recipe's default dataset (SQuAD)
4. Attach LoRA adapters and start fine-tuning across the GPUs it detects

The GPU count is detected rather than assumed; pass `--gpus_per_node` to override. Unlike
the previous version, the script does not re-launch itself -- `torchrun` owns process
startup.

## Training Different Models

### Supported Models

The framework supports any model available in NeMo's recipe collection. Common examples include:

- **Qwen Family**: `qwen3_600m`, `qwen3_1p7b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `qwen3_32b`, `qwen3_30b_a3b`, `qwen3_235b_a22b`
- **Llama Family**: `llama32_1b`, `llama3_8b`, `llama31_8b`, `llama31_70b`
- **Nemotron**, **DeepSeek**, **GLM**, **Phi**, **Gemma**, **GPT-OSS** and others

Coverage depends on the container, so list what yours actually provides rather than
guessing:

```bash
python peft_megatron.py --list_recipes
```

### Finding Recipe Names

Megatron-Bridge exposes a config factory per model and mode, named
`{family}_peft_config` for LoRA/DoRA and `{family}_sft_config` for a full fine-tune.
`--recipe_cls_name` takes the family part, so `--recipe_cls_name qwen3_8b` selects
`qwen3_8b_peft_config`, or `qwen3_8b_sft_config` with `--full_ft`. The factories live in
`megatron.bridge.recipes.*`.

## Using Different Datasets

### Dataset Configuration

The framework uses `HFDatasetConfig` to define dataset loading and formatting. Key parameters:

- `dataset_name`: HuggingFace dataset identifier
- `dataset_config`: Specific subset/configuration
- `input_template`: Format string for input prompts
- `output_template`: Format string for output completions
- `field_mapping`: Maps template variables to dataset columns

### Example 1: Alpaca Format Dataset

```python
HFDatasetConfig(
    dataset_name="tatsu-lab/alpaca",
    split="train",
    input_template="### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n",
    output_template="### Response:\n{output}",
    field_mapping={
        "instruction": "instruction",
        "input": "input", 
        "output": "output"
    }
)
```

### Example 2: OpenAssistant Conversations

```python
HFDatasetConfig(
    dataset_name="OpenAssistant/oasst1",
    split="train",
    input_template="<|prompter|>{prompt}<|endoftext|>\n",
    output_template="<|assistant|>{response}<|endoftext|>",
    field_mapping={
        "prompt": "text",  # Map to actual column name
        "response": "text"  # Adjust based on dataset structure
    }
)
```

### Example 3: ShareGPT Format

```python
HFDatasetConfig(
    dataset_name="anon8231489123/ShareGPT_Vicuna_unfiltered",
    split="train",
    input_template="{human}",
    output_template="{gpt}",
    field_mapping={
        "human": "conversations",  # Extract from conversations list
        "gpt": "conversations"
    },
    custom_converter=lambda x: {
        "input": x["conversations"][0]["value"],
        "output": x["conversations"][1]["value"]
    }
)
```

### Example 4: Custom Dataset with Complex Structure

For datasets with non-standard structures, use `custom_converter`:

```python
def custom_converter(sample):
    """Convert complex dataset format"""
    messages = sample["messages"]
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    
    input_text = f"System: {system_msg}\n\nUser: {user_msg}\n\n"
    output_text = f"Assistant: {assistant_msg}"
    
    return {"input": input_text, "output": output_text}

HFDatasetConfig(
    dataset_name="your-dataset/name",
    custom_converter=custom_converter
)
```

### Running with Custom Dataset

Update the configuration in `peft_megatron.py`:

```python
@dataclass
class Config:
    # ... other config ...
    
    hf_dataset_config: HFDatasetConfig = field(default_factory=lambda: HFDatasetConfig(
        dataset_name="your-org/your-dataset",
        dataset_config="subset-name",  # Optional
        split="train",
        train_split_ratio=0.9,
        val_test_split_ratio=0.5,
        input_template="Your input format: {field1}\n{field2}\n",
        output_template="Your output format: {field3}",
        field_mapping={
            "field1": "actual_column_1",
            "field2": "actual_column_2",
            "field3": "actual_column_3"
        }
    ))
```

## GPU Requirements

All configurations assume **8 GPUs per node** for optimal performance and training stability.

### Small Models (1B - 13B parameters)

**Examples**: Qwen3-8B, Llama3-8B, Mistral-7B, Phi-3-Medium

**Basic Configuration**:
- **Nodes**: 1 node
- **GPUs**: 8x A100 (40GB or 80GB)
- **Micro batch size**: 8-16
- **Gradient accumulation**: 4

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
  --hf_model_id "/snapshots/huggingface/Qwen/Qwen3-8B" \
  --recipe_cls_name "qwen3_8b" \
  --num_nodes 1 \
  --gpus_per_node 8 \
  --micro_batch_size 8 \
  --accumulate_grad_batches 4
```

---

### Medium Models (13B - 34B parameters)

**Examples**: Llama2-13B, Yi-34B, CodeLlama-34B

**Basic Configuration**:
- **Nodes**: 2 nodes
- **GPUs**: 16x A100 (80GB) total
- **Micro batch size**: 4-8
- **Gradient accumulation**: 4-8

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
  --hf_model_id "/snapshots/huggingface/Qwen/Qwen3-14B" \
  --recipe_cls_name "qwen3_14b" \
  --num_nodes 2 \
  --gpus_per_node 8 \
  --micro_batch_size 4 \
  --accumulate_grad_batches 8
```

---

### Large Models (34B - 100B parameters)

**Examples**: Llama3.1-70B, Mixtral-8x22B, Qwen2.5-72B

**Basic Configuration**:
- **Nodes**: 4-8 nodes
- **GPUs**: 32-64x A100 (80GB) or 32-64x H100 (80GB)
- **Micro batch size**: 2-4
- **Gradient accumulation**: 8-16

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
  --hf_model_id "/snapshots/huggingface/meta-llama/Meta-Llama-3.1-70B" \
  --recipe_cls_name "llama31_70b" \
  --num_nodes 4 \
  --gpus_per_node 8 \
  --micro_batch_size 2 \
  --accumulate_grad_batches 16
```

---

### Very Large Models (100B+ parameters)

**Examples**: Llama3.1-405B, Nemotron4-340B, Falcon-180B

**Basic Configuration**:
- **Nodes**: 16-32+ nodes
- **GPUs**: 128-256+ H100 (80GB)
- **Micro batch size**: 1-2
- **Gradient accumulation**: 16-32

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
  --hf_model_id "/snapshots/huggingface/meta-llama/Meta-Llama-3.1-405B" \
  --recipe_cls_name "llama31_405b" \
  --num_nodes 16 \
  --gpus_per_node 8 \
  --micro_batch_size 1 \
  --accumulate_grad_batches 32
```

---

### Hardware Requirements Summary

| Model Size | Parameters | Nodes | Total GPUs | GPU Type | Est. Training Time |
|------------|-----------|-------|------------|----------|-------------------|
| Small | 1B-13B | 1 | 8 | A100 40/80GB | 24-48 hours |
| Medium | 13B-34B | 2 | 16 | A100 80GB | 48-96 hours |
| Large | 34B-100B | 4-8 | 32-64 | A100/H100 80GB | 4-8 days |
| Very Large | 100B+ | 16-32 | 128-256 | H100 80GB | 1-2 weeks |

## Configuration

### Core Training Parameters

All configuration parameters are defined in the Config class in `peft_megatron.py`. Key parameters include:

#### Model and Recipe
- `hf_model_id`: HuggingFace model identifier (e.g., "Qwen/Qwen3-8B")
- `recipe_cls_name`: Megatron-Bridge recipe family (e.g., "qwen3_8b", "llama31_70b"); see [Finding Recipe Names](#finding-recipe-names) and `--list_recipes`
- `peft_scheme`: PEFT method to use (default: "lora")
- `full_ft`: Enable full fine-tuning instead of PEFT (default: False)

#### Paths
- `output_dir`: Base directory for checkpoints and TensorBoard logs (default: `outputs/{recipe_cls_name}`)
- `dataset_root`: Directory of prepared `training/validation/test.jsonl`. Omit it to use the recipe's own dataset (SQuAD); see [Using Different Datasets](#using-different-datasets).

There is no `nemo_ckpt_dir`: Megatron-Bridge reads the HuggingFace snapshot directly, so
there is no intermediate NeMo checkpoint to materialize.

#### Distributed Training
- `num_nodes`: Number of compute nodes (default: 1)
- `gpus_per_node`: GPUs per node (**default: detected from the machine**)
- `tensor_parallel_size`: Tensor parallelism size (**default: 1**)
- `pipeline_parallel_size`: Pipeline parallelism size (default: 1)
- `context_parallel_size`: Context parallelism size (default: 1)

Two notes on the defaults, both changed from the previous version:

**GPU count is detected.** It used to default to 8 regardless of the machine, which
oversubscribed smaller hosts. An explicit `--gpus_per_node` still wins.

**Tensor parallelism defaults to 1, not 8.** Tensor parallelism exists to make a model
*fit*, not to make training faster: it splits attention heads across ranks, costing
communication every layer and shrinking the data-parallel degree. Raise it only when a
model will not otherwise fit. An explicit value is validated -- it must divide both the
GPU count and the model's key/value head count, so for example Qwen3-8B has 32 attention
heads but only 8 KV heads, and a 6-GPU machine cannot use TP=6 even though 6 divides 6.
`node_rank` is gone; `torchrun` assigns ranks.

#### Training Hyperparameters
- `max_steps`: Maximum training steps (default: 10000)
- `val_check_interval`: Validation frequency in steps (default: 800)
- `log_every_n_steps`: Logging frequency (default: 10)
- `micro_batch_size`: Batch size per GPU (default: 8)
- `accumulate_grad_batches`: Gradient accumulation steps (default: 8)
- `global_batch_size`: Effective batch size (computed: micro_batch_size × accumulate_grad_batches × data_parallel_size)
- `limit_val_batches`: Maximum validation batches (default: 80)
- `clip_grad`: Gradient clipping threshold; 0 disables it (default: the recipe's value)
- `lora_dim` / `lora_alpha`: LoRA rank and alpha (default: the recipe's values)

Early stopping is not currently wired up: it was a NeMo `EarlyStopping` callback, and
Megatron-Bridge configures callbacks differently. Use `--max_steps` to bound a run.

#### Sequence Configuration
- `max_seq_length`: Maximum sequence length (default: 2048)

#### Dataset Configuration
- `dataset_root`: Directory of prepared JSONL files. Dataset loading and template
  formatting now live in `dataset_module.py`, which is run as a separate step rather than
  configured inline -- see [Using Different Datasets](#using-different-datasets).

### Advanced Configuration Options

#### Gradient Clipping and LoRA Shape

The recipe sets these; override them on the command line:

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
    --hf_model_id /snapshots/huggingface/Qwen/Qwen3-8B \
    --lora_dim 16 --lora_alpha 32 --clip_grad 1.0
```

Early stopping is not wired up in this version. It used NeMo's `EarlyStopping` callback,
which went away with `nemo.lightning`; Megatron-Bridge configures callbacks through its
own `CallbackManager`. Bound runs with `--max_steps` for now.

#### LoRA Parameters

Customize LoRA settings in the recipe:

```python
nemo_recipe.peft.lora_tuning.target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
nemo_recipe.peft.lora_tuning.adapter.dim = 16
nemo_recipe.peft.lora_tuning.adapter.alpha = 32
nemo_recipe.peft.lora_tuning.adapter.dropout = 0.05
```

#### Parallelism Strategy

Parallelism is set through Config parameters and applied to the recipe:
- `tensor_parallel_size`: Tensor parallelism size (**default: 1**, see [Distributed Training](#distributed-training))
- `pipeline_parallel_size`: Pipeline parallelism size (default: 1)
- `context_parallel_size`: Context parallelism size (default: 1)

```python
cfg.model.tensor_model_parallel_size = config.tensor_parallel_size
cfg.model.pipeline_model_parallel_size = config.pipeline_parallel_size
cfg.model.context_parallel_size = config.context_parallel_size
```

The data-parallel degree is whatever is left:
`(num_nodes x gpus_per_node) / (TP x PP x CP)`. If that reaches zero the script raises
rather than silently producing a zero global batch size, which is what the previous
version did when `--gpus_per_node` was lowered without also lowering TP.

### Monitoring Training

#### Using TensorBoard

```bash
# In a separate terminal
tensorboard --logdir tb_logs
```

TensorBoard logs are saved to `tb_logs/peft_megatron/` by default.

## CLI Usage Examples

### Basic Usage

Run the training script with default HFDatasetConfig:

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
  --hf_model_id "/snapshots/huggingface/Qwen/Qwen3-8B" \
  --recipe_cls_name "qwen3_8b" \
  --num_nodes 1 \
  --gpus_per_node 8
```

### Full Fine-Tuning vs PEFT

By default, the framework uses PEFT (LoRA). To enable full fine-tuning of all model parameters:

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
  --hf_model_id "/snapshots/huggingface/Qwen/Qwen3-8B" \
  --recipe_cls_name "qwen3_8b" \
  --num_nodes 1 \
  --gpus_per_node 8 \
  --full_ft
```

**Note**: Full fine-tuning requires significantly more GPU memory and compute resources than PEFT. Adjust `micro_batch_size` and `accumulate_grad_batches` accordingly.

### Preparing a Custom Dataset

Dataset loading and template formatting moved out of the training script into
`dataset_module.py`, which writes the prompt/completion JSONL that Megatron-Bridge SFT
reads. The `--hfdc_*` flags on the training script are gone; the same fields are now
options on the preparer, run once up front:

```bash
python dataset_module.py \
    --dataset_name databricks/databricks-dolly-15k \
    --split train \
    --dataset_root ./data/dolly \
    --field_mapping '{"instruction": "instruction", "input": "context", "output": "response"}' \
    --train_split_ratio 0.9 \
    --val_test_split_ratio 0.5
```

Then train against the prepared directory:

```bash
torchrun --nproc_per_node=8 peft_megatron.py \
    --hf_model_id /snapshots/huggingface/Qwen/Qwen3-8B \
    --dataset_root ./data/dolly
```

**Preparer options:** `--dataset_name`, `--dataset_config`, `--split`, `--dataset_root`,
`--train_split_ratio`, `--val_test_split_ratio`, `--input_template`,
`--output_template`, `--field_mapping`, `--num_proc`, `--overwrite`.

Templates and field mapping behave exactly as before: `{placeholder}` names in
`--input_template` / `--output_template` are looked up in the dataset columns, through
`--field_mapping` when the names differ. Each output row is
`{"input": ..., "output": ...}`, which training consumes via
`PromptCompletionSFTPreprocessingConfig(prompt_column="input", completion_column="output")`.

Omit `--dataset_root` entirely to use the recipe's own dataset (SQuAD), which needs no
preparation step.

### Complete Example with All Options

Two steps: prepare the dataset, then train against it.

```bash
# 1. prepare the dataset once
python dataset_module.py \
    --dataset_name timdettmers/openassistant-guanaco \
    --split train \
    --dataset_root ./data/guanaco \
    --train_split_ratio 0.95 \
    --val_test_split_ratio 0.5 \
    --input_template "### Human:\n{text}\n" \
    --output_template "### Assistant:\n{text}" \
    --num_proc 16

# 2. train
torchrun --nproc_per_node=8 peft_megatron.py \
    --hf_model_id /snapshots/huggingface/Qwen/Qwen3-8B \
    --recipe_cls_name qwen3_8b \
    --dataset_root ./data/guanaco \
    --output_dir outputs/qwen3_8b_guanaco \
    --max_steps 5000 \
    --micro_batch_size 4 \
    --accumulate_grad_batches 16 \
    --peft_scheme lora \
    --lora_dim 16 --lora_alpha 32 \
    --max_seq_length 4096
```

For multi-node, `torchrun` carries the topology (`--nnodes`, `--node_rank`,
`--master_addr`, `--master_port`) and `--num_nodes` tells the script the world size it
should expect when computing the global batch size.

### CLI Notes

- The `field_mapping` argument expects a JSON string
- Use single quotes around JSON to avoid shell interpretation issues
- Template strings can include `\n` for newlines
- Not all HFDatasetConfig fields are exposed via CLI (e.g., `custom_converter`, `load_kwargs`)
- For complex configurations, modify the Config dataclass in `peft_megatron.py` directly

## Testing and Converting Checkpoints

### Testing a Checkpoint

Evaluation is a two-step process: export the trained adapter to a HuggingFace model
directory, then score that directory with vLLM and BERTScore.

```bash
# 1. merge the LoRA adapter into the base weights
python convert_checkpoint_to_hf.py \
    --megatron_checkpoint ./outputs/qwen3_8b/checkpoints \
    --base_model /snapshots/huggingface/Qwen/Qwen3-8B

# 2. score the merged model
python test_checkpoint.py \
    --model_dir ./outputs/qwen3_8b/checkpoints.hf_model \
    --test_path ./data/alpaca/test.jsonl \
    --max_samples 1024
```

This replaces the previous NeMo dynamic-inference path, which depended on
`nemo.lightning` and `nemo.collections.llm.inference`. Scoring with vLLM is what the
accelerate, ptl and ray_train examples already do, so **all four frameworks now produce
comparable BERTScore numbers** -- previously nemo2's came from a different engine.

Evaluation has no sm90 requirement: vLLM runs on Ada-class GPUs, so you can score a
model here even where training needs Hopper.

**Parameters:**
- `--model_dir`: HuggingFace model directory to evaluate (required)
- `--test_path`: Test dataset JSONL with `input`/`output` rows (required; `dataset_module.py` writes `test.jsonl`)
- `--output_path`: Where predictions are written (default: `predictions.jsonl` beside the test file)
- `--max_samples`: Maximum test samples to evaluate (default: 1024)
- `--batch_size`: Generation batch size (default: 128)
- `--tensor_parallel_size`: vLLM tensor parallelism (default: auto-selected from the visible GPUs and the model's KV heads)
- `--gpu_memory_utilization`: Fraction of GPU memory vLLM may use (default: 0.9)
- `--max_model_len`: Maximum sequence length for serving (default: 4096)
- `--temperature`, `--top_k`, `--top_p`, `--max_tokens`: Sampling controls

### Converting to Hugging Face Format

Export a trained checkpoint to standard HuggingFace format for deployment:

```bash
python convert_checkpoint_to_hf.py \
    --megatron_checkpoint ./outputs/qwen3_8b/checkpoints \
    --base_model /snapshots/huggingface/Qwen/Qwen3-8B
```

The script:
- Selects the newest `iter_XXXXXXX` directory under the checkpoint path
- Delegates to the exporters shipped in the container rather than reimplementing the
  merge: `examples/peft/merge_lora.py` for merged weights, or
  `examples/conversion/adapter/export_adapter.py` with `--no_merge` for adapter-only
- Saves to `{checkpoint}.hf_model` (merged) or `{checkpoint}.hf_peft` (adapter only)

**Parameters:**
- `--megatron_checkpoint`: Directory holding the LoRA checkpoint from training (required)
- `--base_model`: HF model id or local snapshot supplying the architecture config (default: `Qwen/Qwen3-8B`)
- `--pretrained_checkpoint`: Base dense checkpoint; optional, resolved from `run_config.yaml` in the LoRA checkpoint when omitted
- `--output_dir`: Override the default `.hf_model`/`.hf_peft` output path
- `--no_merge`: Export the adapter alone instead of merging it (default: False)
- `--cpu`: Merge entirely on CPU -- slower, but needs no GPU
- `--tensor_parallel_size`, `--pipeline_parallel_size`, `--expert_parallel_size`: parallelism for a GPU export
- `--trust_remote_code`, `--debug`

**LoRA Merging:**
- By default LoRA weights are **merged** into the base model for maximum compatibility
- Merged models work with vLLM, TGI and all HuggingFace tools
- `--no_merge` saves a separate adapter (requires PEFT to load)

Gone from the previous version: `--target`, `--overwrite`, `--use_modelopt` and
`--nemo_logs_dir`, which belonged to `nemo.collections.llm.peft` and the NeMo log layout.

## Project Structure

```
.
├── peft_megatron.py                # Main training script (Megatron-Bridge)
├── dataset_module.py               # HuggingFace dataset -> prompt/completion JSONL
├── test_checkpoint.py              # Evaluate a HF model dir with vLLM + BERTScore
├── convert_checkpoint_to_hf.py     # Export/merge a LoRA checkpoint to HF format
├── device_preflight.py             # GPU detection and capacity checks
├── README.md                       # This file
├── data/                           # Prepared datasets (--dataset_root)
│   └── {name}/
│       ├── training.jsonl          # {"input": ..., "output": ...} per line
│       ├── validation.jsonl
│       ├── test.jsonl
│       └── .data_ready             # delete to force a rebuild
└── outputs/                        # Training outputs
    └── {recipe_cls_name}/
        ├── checkpoints/            # Megatron-Bridge checkpoints
        │   └── iter_XXXXXXX/
        ├── checkpoints.hf_model/   # merged HF model (convert_checkpoint_to_hf.py)
        ├── checkpoints.hf_peft/    # adapter-only export (--no_merge)
        └── tb_logs/                # TensorBoard
```

Gone from the previous layout: `imported_hf_ckpt/` (base weights load straight from the
HuggingFace snapshot) and `nemo_logs/` with its
`nemo_logs--val_loss={}-epoch={}-consumed_samples={}` checkpoint directories.

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce micro batch size**
```bash
--micro_batch_size 4  # or even 2 or 1 for very large models
```

**Solution 2: Increase gradient accumulation**
```bash
--accumulate_grad_batches 16  # or 32
```

**Solution 3: Reduce sequence length**
```bash
--max_seq_length 1024  # from default 2048
```

### Data Loading Issues

**Field mapping errors**

Check that template placeholders match dataset columns:
```python
from datasets import load_dataset

# Inspect dataset structure
ds = load_dataset("your-dataset")
print("Available columns:", ds['train'].column_names)
print("Sample data:", ds['train'][0])
```

**Encoding errors**

Ensure UTF-8 encoding in dataset conversion:
```python
# The framework handles this automatically, but verify:
with open("datasets/your-data/train.jsonl", "r", encoding="utf-8") as f:
    print(f.readline())
```

**Empty or invalid samples**

Add validation in custom converter:
```python
def custom_converter(sample):
    result = process_sample(sample)
    
    # Validate
    if not result or not result.get("input") or not result.get("output"):
        return None
    
    return result
```

### Training Instability

**Enable gradient clipping**
```python
nemo_recipe.trainer.gradient_clip_val = 1.0
nemo_recipe.trainer.gradient_clip_algorithm = "norm"
```

**Adjust learning rate**
```python
nemo_recipe.optim.lr = 1e-4  # Lower for stability
nemo_recipe.optim.sched.warmup_steps = 100
```
**Optimize data loading**
```python
# In HFDatasetConfig
config.hf_dataset_config.num_proc = 16  # Increase parallel processing
```

**Enable Flash Attention (if supported)**
```python
# For supported models
nemo_recipe.model.use_flash_attention = True
```

**Optimize checkpoint saving**
```python
# Save less frequently for large models
nemo_recipe.trainer.val_check_interval = 500
nemo_recipe.trainer.save_top_k = 3  # Keep only best 3 checkpoints
```
## Monitoring and Profiling

Megatron-Bridge configures profiling through its own `ProfilingConfig` and
`CallbackManager`, so the NeMo-era `--enable_*` flags are gone. What remains on the CLI:

| Configuration Option | Type | Default | Description | Output Location |
|---------------------|------|---------|-------------|----------------|
| `--use_wandb` | bool | False | Enables Weights & Biases logging for experiment tracking | W&B dashboard |

TensorBoard logging is always on, writing to `{output_dir}/tb_logs`.

The removed flags were `--enable_megatron_progress_bar`, `--enable_memory_monitor`,
`--enable_speed_monitor`, `--enable_runtime_estimator`, `--enable_memory_profile`,
`--enable_pytorch_profiler` and `--enable_nsys_callback`. Their equivalents live on
`ConfigContainer.profiling` (`ProfilingConfig`) and the callback list passed to
`finetune()`; wiring them back up is a reasonable extension to this example.

### Notes

- **Performance Impact**: Profiling callbacks (PyTorch Profiler, Nsys, Memory Profile) add overhead and should only be enabled for short diagnostic runs
- **Storage**: Profiler traces can be large; ensure adequate disk space in the output directory
- **Profiler Analysis**: Use `tensorboard --logdir outputs/{model}/trace` to visualize PyTorch profiler results
- **Nsys Analysis**: Analyze Nsys profiles with NVIDIA Nsight Systems GUI
---