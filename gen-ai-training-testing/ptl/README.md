# PyTorch Lightning with FSDP

This project provides a flexible framework for fine-tuning Large Language Models using PyTorch Lightning and FSDP (Fully Sharded Data Parallel). The framework provides a generalized data pipeline for HuggingFace datasets and streamlined configuration for distributed training with LoRA or full fine-tuning.

## Features

- **Generalized HuggingFace Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates and field mapping
- **Distributed Training**: Multi-node, multi-GPU training with FSDP for efficient memory usage
- **LoRA and Full Fine-Tuning**: Support for LoRA parameter-efficient fine-tuning or full fine-tuning
- **Automatic Data Conversion**: Converts HuggingFace datasets to JSONL format for efficient loading
- **Customizable Training**: Extensive configuration options for hyperparameters, callbacks, and training strategies
- **Flash Attention 2**: Optimized attention implementation for faster training
- **Activation Checkpointing**: Reduce memory usage for large models
- **Early Stopping**: Automatic training termination based on validation loss
- **vLLM Inference**: Test checkpoints with vLLM for efficient batch evaluation

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

## Prerequisites

Follow [Step by Step Tutorial](../../README.md) to launch a Deep Learning Desktop. On the desktop,

```bash
cd ~
git clone https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop.git
cd ~/aws-deep-learning-ami-ubuntu-dcv-desktop/gen-ai-training-testing/ptl
```
## Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker buildx build -t ptl:latest -f ../containers/Dockerfile.ptl .
```

2. Run the container with GPU support:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  --shm-size=32g \
  ptl:latest
```

## Quick Start

Train the default Qwen3-8B model on the Dolphin dataset with optimal settings:

```bash
python peft_hf.py
```

This will:
1. Download the Qwen/Qwen3-8B model from HuggingFace
2. Load and process the Dolphin dataset
3. Start LoRA fine-tuning with 8 GPUs using FSDP

## Training Different Models

### Supported Models

The framework supports any HuggingFace causal language model. Common examples include:

- **Qwen Family**: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-70B`
- **Llama Family**: `meta-llama/Llama-3-8B`, `meta-llama/Llama-3-70B`, `meta-llama/Meta-Llama-3.1-8B`, `meta-llama/Meta-Llama-3.1-70B`
- **Mistral**: `mistralai/Mistral-7B-v0.1`, `mistralai/Mixtral-8x7B-v0.1`
- **Phi**: `microsoft/Phi-3-medium-4k-instruct`

### Important: Decoder Layer Import for FSDP

**When training a different model family, you must update the decoder layer import in `peft_hf.py`** to match your model architecture. This is required for FSDP's auto-wrap policy and activation checkpointing.

The current implementation uses:
```python
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
```

For other models, change the import to the appropriate decoder layer class:

| Model Family | Import Statement |
|--------------|------------------|
| **Llama 3 / 3.1** | `from transformers.models.llama.modeling_llama import LlamaDecoderLayer` |
| **Mistral** | `from transformers.models.mistral.modeling_mistral import MistralDecoderLayer` |
| **Mixtral** | `from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer` |
| **Phi-3** | `from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer` |
| **Qwen3** | `from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer` |
| **Gemma** | `from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer` |

Then update the references in two places:

1. **In `configure_strategy()` function:**
```python
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},  # Change this
)
```

2. **In `HFCausalLMModule.configure_model()` method:**
```python
def check_fn(submodule):
    return isinstance(submodule, LlamaDecoderLayer)  # Change this
```

### Training a Different Model

```bash
python peft_hf.py \
  --hf_model_id "meta-llama/Llama-3-8B" \
  --micro_batch_size 2 \
  --accumulate_grad_batches 8
```

## Using Different Datasets

### Dataset Configuration

The framework uses `HFDatasetConfig` to define dataset loading and formatting. Key parameters:

- `dataset_name`: HuggingFace dataset identifier
- `dataset_config`: Specific subset/configuration
- `input_template`: Format string for input prompts
- `output_template`: Format string for output completions
- `field_mapping`: Maps template variables to dataset columns
- `num_proc`: Number of processes for dataset loading (default: 8)

### Example 1: Alpaca Format Dataset

```python
HFDatasetConfig(
    dataset_name="tatsu-lab/alpaca",
    split="train",
    input_template="### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n",
    output_template="### Response:\\n{output}",
    field_mapping={
        "instruction": "instruction",
        "input": "input", 
        "output": "output"
    },
    num_proc=8
)
```

### Example 2: Custom Dataset with Complex Structure

For datasets with non-standard structures, use `custom_converter`:

```python
def custom_converter(sample):
    """Convert complex dataset format"""
    messages = sample["messages"]
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    
    input_text = f"System: {system_msg}\\n\\nUser: {user_msg}\\n\\n"
    output_text = f"Assistant: {assistant_msg}"
    
    return {"input": input_text, "output": output_text}

HFDatasetConfig(
    dataset_name="your-dataset/name",
    custom_converter=custom_converter,
    num_proc=8
)
```

### Running with Custom Dataset

Update the configuration in `peft_hf.py`:

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
        },
        num_proc=8
    ))
```

## GPU Requirements

All configurations assume **8 GPUs per node** for optimal performance and training stability.

### Small Models (1B - 13B parameters)

**Examples**: Qwen3-8B, Llama3-8B, Mistral-7B, Phi-3-Medium

**Basic Configuration**:
- **Nodes**: 1 node
- **GPUs**: 8x A100 (40GB or 80GB)
- **Micro batch size**: 2-4
- **Gradient accumulation**: 4-8

```bash
python peft_hf.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --num_nodes 1 \
  --gpus_per_node 8 \
  --micro_batch_size 2 \
  --accumulate_grad_batches 4
```

---

### Medium Models (13B - 34B parameters)

**Examples**: Llama2-13B, Yi-34B, CodeLlama-34B

**Basic Configuration**:
- **Nodes**: 2 nodes
- **GPUs**: 16x A100 (80GB) total
- **Micro batch size**: 1-2
- **Gradient accumulation**: 8-16

```bash
python peft_hf.py \
  --hf_model_id "meta-llama/Llama-2-13b-hf" \
  --num_nodes 2 \
  --gpus_per_node 8 \
  --micro_batch_size 1 \
  --accumulate_grad_batches 16
```

---

### Large Models (34B - 100B parameters)

**Examples**: Llama3.1-70B, Mixtral-8x22B, Qwen2.5-72B

**Basic Configuration**:
- **Nodes**: 4-8 nodes
- **GPUs**: 32-64x A100 (80GB) or 32-64x H100 (80GB)
- **Micro batch size**: 1
- **Gradient accumulation**: 16-32
- **CPU Offload**: Consider enabling for memory constraints

```bash
python peft_hf.py \
  --hf_model_id "meta-llama/Meta-Llama-3.1-70B" \
  --num_nodes 4 \
  --gpus_per_node 8 \
  --micro_batch_size 1 \
  --accumulate_grad_batches 32 \
  --cpu_offload
```

---

### Hardware Requirements Summary

| Model Size | Parameters | Nodes | Total GPUs | GPU Type | Est. Training Time |
|------------|-----------|-------|------------|----------|-------------------|
| Small | 1B-13B | 1 | 8 | A100 40/80GB | 24-48 hours |
| Medium | 13B-34B | 2 | 16 | A100 80GB | 48-96 hours |
| Large | 34B-100B | 4-8 | 32-64 | A100/H100 80GB | 4-8 days |

## Configuration

### Core Training Parameters

All configuration parameters are defined in the Config class in `peft_hf.py`. Key parameters include:

- **Model**: `hf_model_id` - HuggingFace model identifier
- **Paths**: 
  - `data_dir`: Directory for processed datasets (default: auto-generated based on dataset configuration as `datasets/{dataset_name}/{dataset_config}/train={train_%}-val={val%}-test={test%}`)
  - `results_dir`: Base directory for training outputs (default: `results/{hf_model_id}`)
- **Distributed Training**: `num_nodes`, `gpus_per_node` - Multi-node/GPU setup

**Note on data_dir**: When not explicitly set, the framework automatically generates a self-documenting directory path based on your dataset configuration. For example, with the default Dolphin dataset (train_split_ratio=0.9, val_test_split_ratio=0.5), the path becomes: `datasets/cognitivecomputations_dolphin/flan1m-alpaca-uncensored/train=90%-val=5%-test=5%`. This ensures each dataset configuration has a unique directory and makes it easy to identify the split ratios used.
- **Training**: `max_steps`, `val_check_interval`, `micro_batch_size`, `accumulate_grad_batches`, `limit_val_batches`, `log_every_n_steps`
- **Optimizer**: `warmup_steps`, `weight_decay`, learning rates (computed properties)
- **LoRA**: `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules` - Comma-separated list of modules to apply LoRA
- **Fine-tuning Mode**: `full_ft` - Enable full fine-tuning instead of LoRA (default: False)
- **Sequence**: `max_seq_length` - Maximum sequence length
- **FSDP**: `cpu_offload` - Enable CPU offloading for large models
- **Logging**: `use_wandb` - Weights & Biases integration

### Full Fine-Tuning vs LoRA

By default, the framework uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. To perform full fine-tuning instead:

```bash
python peft_hf.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --full_ft
```

**Note:** Full fine-tuning requires significantly more memory. Learning rates are configured in the Config class (default: max_lr=1e-5, min_lr=1e-7).

### Advanced Configuration Options

#### Early Stopping

Modify the `configure_callbacks()` function:

```python
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,      # Minimum change to qualify as improvement
    patience=3,           # Number of checks with no improvement
    mode='min',           # Minimize validation loss
)
```

#### FSDP Strategy

The framework uses FSDP with the following optimizations:
- **Sharding Strategy**: FULL_SHARD for maximum memory efficiency
- **Mixed Precision**: BFloat16 for training stability
- **Backward Prefetch**: BACKWARD_PRE for improved performance
- **CPU Offload**: Optional for very large models

### Monitoring Training

#### Using TensorBoard

```bash
# In a separate terminal
tensorboard --logdir results/Qwen/Qwen3-8B/tb_logs
```

#### Using Weights & Biases

```bash
python peft_hf.py \
  --use_wandb
```

## CLI Usage Examples

### Basic Usage

Run the training script with default HFDatasetConfig:

```bash
python peft_hf.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --num_nodes 1 \
  --gpus_per_node 8
```

### Customizing HFDatasetConfig via CLI

Override HFDatasetConfig fields using the `--hfdc_<field_name>` pattern:

```bash
python peft_hf.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --num_nodes 1 \
  --gpus_per_node 8 \
  --hfdc_dataset_name "databricks/databricks-dolly-15k" \
  --hfdc_split "train" \
  --hfdc_train_split_ratio 0.95 \
  --hfdc_val_test_split_ratio 0.5 \
  --hfdc_input_template "### Instruction:\n{instruction}\n### Context:\n{context}\n" \
  --hfdc_output_template "### Response:\n{response}" \
  --hfdc_field_mapping '{"instruction": "instruction", "context": "context", "response": "response"}' \
  --hfdc_num_proc 8
```

### Available HFDatasetConfig CLI Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--hfdc_dataset_name` | str | HuggingFace dataset name | `"cognitivecomputations/dolphin"` |
| `--hfdc_dataset_config` | str | Dataset configuration/subset | `"flan1m-alpaca-uncensored"` |
| `--hfdc_split` | str | Initial split to load | `"train"` |
| `--hfdc_train_split_ratio` | float | Training data ratio | `0.9` |
| `--hfdc_val_test_split_ratio` | float | Val/test split ratio | `0.5` |
| `--hfdc_input_template` | str | Input formatting template | `"### Instruction:\n{instruction}\n"` |
| `--hfdc_output_template` | str | Output formatting template | `"### Response:\n{output}"` |
| `--hfdc_field_mapping` | str (JSON) | Field name mapping | `'{"instruction": "text"}'` |
| `--hfdc_num_proc` | int | Number of processes | `8` |

### Complete CLI Example with All Options

```bash
python peft_hf.py \
  --hf_model_id "meta-llama/Llama-3-8B" \
  --data_dir "datasets/custom" \
  --results_dir "results/llama3_custom" \
  --num_nodes 2 \
  --gpus_per_node 8 \
  --max_steps 5000 \
  --micro_batch_size 1 \
  --accumulate_grad_batches 16 \
  --max_seq_length 4096 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
  --hfdc_dataset_name "timdettmers/openassistant-guanaco" \
  --hfdc_split "train" \
  --hfdc_train_split_ratio 0.95 \
  --hfdc_val_test_split_ratio 0.5 \
  --hfdc_input_template "### Human:\n{text}\n" \
  --hfdc_output_template "### Assistant:\n{text}" \
  --hfdc_num_proc 8 \
  --use_wandb
```

### CLI Notes

- The `field_mapping` argument expects a JSON string
- Use single quotes around JSON to avoid shell interpretation issues
- Template strings can include `\n` for newlines
- Not all HFDatasetConfig fields are exposed via CLI (e.g., `custom_converter`, `load_kwargs`)
- For complex configurations, modify the Config dataclass in `peft_hf.py` directly

## Testing and Converting Checkpoints

### Testing a Checkpoint

After training, test your checkpoint using vLLM for efficient inference:

```bash
python test_checkpoint.py \
  --base_model "Qwen/Qwen3-8B" \
  --max_samples 1024 \
  --batch_size 128
```

The script automatically:
- Finds the latest checkpoint (by modification time) in `results/{base_model}/checkpoints/`
- Discovers the latest `test.jsonl` file (by modification time) under `datasets/`
- Detects checkpoint type (LoRA or full fine-tuned) automatically
- Loads the checkpoint and merges LoRA weights (if applicable)
- Uses vLLM for fast batched inference
- Evaluates predictions using BERTScore

**Parameters:**
- `--base_model`: Base model ID (must match the model used for training)
- `--checkpoints_dir`: Directory containing checkpoint files (optional, default: `results/{base_model}/checkpoints`)
- `--test_path`: Path to test dataset JSONL file (optional, auto-discovered from `datasets/`)
- `--max_samples`: Maximum number of test samples to evaluate (default: 1024)
- `--batch_size`: Batch size for generation (default: 128)
- `--temperature`: Sampling temperature (default: 0.1)
- `--top_k`: Top-k sampling parameter (default: -1, disabled)
- `--top_p`: Nucleus sampling parameter (default: 0.95)
- `--max_tokens`: Maximum tokens to generate (default: 512)
- `--tensor_parallel_size`: vLLM tensor parallelism (default: 8)
- `--gpu_memory_utilization`: GPU memory utilization (default: 0.9)
- `--max_model_len`: Maximum model sequence length (default: 8192)
- `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`: LoRA configuration (must match training)

**Output:** Predictions are saved to `{checkpoint_path}.jsonl` (replacing `.ckpt` extension) and evaluated using BERTScore.

### Converting to Hugging Face Format

Convert your PyTorch Lightning checkpoint to standard Hugging Face format for deployment:

```bash
python convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B"
```

The script automatically finds the latest checkpoint in `results/{base_model}/checkpoints/`. By default, it merges LoRA weights into the base model for maximum compatibility.

**Parameters:**
- `--base_model`: Base model ID (must match the model used for training)
- `--checkpoints_dir`: Directory containing checkpoint files (optional, default: `results/{base_model}/checkpoints`)
- `--no_merge`: If set, save as LoRA adapter; otherwise merge into base model (default: merge)
- `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`: LoRA configuration (must match training)

**Checkpoint Type Detection:**
- The script automatically detects whether the checkpoint is LoRA or full fine-tuned
- For LoRA checkpoints: By default, weights are **merged** into the base model for maximum compatibility
- Merged models work with vLLM, TGI, and all Hugging Face tools
- Use `--no_merge` to save as a separate LoRA adapter (smaller file size, requires PEFT library)
- Output is saved as `.hf_model` (merged) or `.hf_peft` (adapter) suffix on checkpoint filename

## Project Structure

```
.
├── peft_hf.py                      # Main training script
├── dataset_module.py               # Dataset processing module
├── test_checkpoint.py              # Test checkpoint with vLLM
├── convert_checkpoint_to_hf.py     # Convert to Hugging Face format
├── README.md                       # This file
├── datasets/                       # Downloaded and processed datasets
│   └── {dataset_name}/             # e.g., cognitivecomputations_dolphin
│       └── {dataset_config}/       # e.g., flan1m-alpaca-uncensored
│           └── train={train_%}-val={val%}-test={test%}/  # e.g., train=90%-val=5%-test=5%
│               ├── training.jsonl
│               ├── validation.jsonl
│               ├── test.jsonl
│               └── .data_ready
└── results/                        # Training outputs and logs
    └── Qwen/
        └── Qwen3-8B/
            ├── checkpoints/
            │   ├── model-peft-lora-{epoch:02d}-{step}.ckpt
            │   ├── model-ft-{epoch:02d}-{step}.ckpt
            │   ├── model-peft-lora-{epoch:02d}-{step}.jsonl  # Test predictions
            │   ├── model-peft-lora-{epoch:02d}-{step}.hf_model/  # Converted merged model
            │   └── model-peft-lora-{epoch:02d}-{step}.hf_peft/   # Converted LoRA adapter
            ├── tb_logs/
            └── wandb_logs/
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce micro batch size**
```bash
--micro_batch_size 1
```

**Solution 2: Increase gradient accumulation**
```bash
--accumulate_grad_batches 16  # or 32
```

**Solution 3: Enable CPU offload**
```bash
--cpu_offload
```

**Solution 4: Reduce sequence length**
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
with open("datasets/your-data/training.jsonl", "r", encoding="utf-8") as f:
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

**Gradient clipping**

The framework automatically handles gradient clipping in the `on_before_optimizer_step` method with `max_norm=1.0`. Gradient clipping is applied manually for FSDP + PEFT compatibility to ensure stable training with sharded parameters.

**Learning rates**
```python
# Learning rates are configured in the Config class
# Default: max_lr=1e-5, min_lr=1e-7
# Override by passing --max_learning_rate and --min_learning_rate arguments
```

**Optimizer configuration**
- Uses AdamW with fused implementation for better performance
- Separate weight decay for parameters (no decay for bias/norm layers)
- Cosine annealing scheduler with warmup

### vLLM Testing Issues

**Run directly with python**
```bash
python test_checkpoint.py --base_model "Qwen/Qwen3-8B"
```

**Tensor parallelism configuration**

The test script uses vLLM with tensor parallelism (default: 8 GPUs). Ensure all GPUs are available and visible. Adjust with `--tensor_parallel_size` if needed.

## Additional Notes

### Flash Attention
The training script uses `attn_implementation="flash_attention_2"` for improved stability and performance. Ensure flash-attn is installed.

### Checkpoint Format
- Training checkpoints are saved as `.ckpt` files with PyTorch Lightning format
- Checkpoint filenames indicate training type: `model-peft-lora-{epoch:02d}-{step}.ckpt` or `model-ft-{epoch:02d}-{step}.ckpt`
- Latest checkpoint is automatically selected by modification time
- State dict keys have `model.` prefix which is automatically removed during conversion

### Memory Optimization
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set for better memory management
- FSDP uses FULL_SHARD strategy for maximum memory efficiency
- Activation checkpointing applied to decoder layers when enabled
- CPU offload available for very large models
