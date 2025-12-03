# Hugging Face Accelerate Parameter-Efficient Fine-Tuning (PEFT) Framework

This project provides a flexible framework for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models using Hugging Face Trainer with FSDP (Fully Sharded Data Parallel). The framework provides a generalized data pipeline for HuggingFace datasets and streamlined configuration for distributed training with LoRA.

## Features

- **Generalized HuggingFace Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates and field mapping
- **Distributed Training**: Multi-node, multi-GPU training with FSDP for efficient memory usage
- **PEFT Methods**: Support for LoRA parameter-efficient fine-tuning via HuggingFace PEFT
- **Automatic Data Conversion**: Converts HuggingFace datasets to JSONL format for efficient loading
- **Customizable Training**: Extensive configuration options for hyperparameters and training strategies
- **Flash Attention 2**: Optimized attention implementation for faster training
- **Gradient Checkpointing**: Reduce memory usage for large models

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Different Models](#training-different-models)
- [Using Different Datasets](#using-different-datasets)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Follow [Step by Step Tutorial](../../README.md) to launch a Deep Learning Desktop. On the desktop:

```bash
cd ~
git clone https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop.git
cd ~/aws-deep-learning-ami-ubuntu-dcv-desktop/gen-ai-training-testing/accelerate
```

## Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker buildx build -t accelerate:latest -f ../containers/Dockerfile.accelerate .
```

2. Run the container with GPU support:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  --shm-size=32g \
  accelerate:latest
```

## Quick Start

Train the default Qwen3-8B model on the Dolphin dataset with optimal settings:

```bash
accelerate launch --config_file fsdp_config.yaml peft_accelerate.py
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

### Training a Different Model

```bash
accelerate launch --config_file fsdp_config.yaml peft_accelerate.py \
  --hf_model_id "meta-llama/Llama-3-8B" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8
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

### Example: Custom Dataset

Update the configuration in `peft_accelerate.py` or use CLI arguments:

```bash
accelerate launch --config_file fsdp_config.yaml peft_accelerate.py \
  --hfdc_dataset_name "databricks/databricks-dolly-15k" \
  --hfdc_split "train" \
  --hfdc_train_split_ratio 0.95 \
  --hfdc_val_test_split_ratio 0.5 \
  --hfdc_input_template "### Instruction:\n{instruction}\n### Context:\n{context}\n" \
  --hfdc_output_template "### Response:\n{response}" \
  --hfdc_field_mapping '{"instruction": "instruction", "context": "context", "response": "response"}' \
  --hfdc_num_proc 8
```

## Configuration

### Core Training Parameters

All configuration parameters are defined in the `TrainingConfig` class in `peft_accelerate.py`. Key parameters include:

- **Model**: `hf_model_id` - HuggingFace model identifier
- **Paths**: 
  - `data_dir`: Directory for processed datasets (default: auto-generated)
  - `output_dir`: Base directory for training outputs (default: `results/{hf_model_id}`)
- **Training**: `max_steps`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`
- **LoRA**: `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules`
- **Sequence**: `max_seq_length` - Maximum sequence length
- **Logging**: `logging_steps`, `eval_steps`
- **Early Stopping**: `early_stopping_patience`, `early_stopping_threshold`

### Accelerate Configuration

The `fsdp_config.yaml` file configures the distributed training setup:

- **FSDP Strategy**: FULL_SHARD for maximum memory efficiency
- **Mixed Precision**: BFloat16 for training stability
- **Backward Prefetch**: BACKWARD_PRE for improved performance
- **Number of Processes**: Set to match your GPU count

### Multi-Node Training

For multi-node training, update the `fsdp_config.yaml`:

```yaml
num_machines: 2
num_processes: 16  # 8 GPUs per node × 2 nodes
machine_rank: 0  # Set to 0 for main node, 1 for second node, etc.
main_process_ip: <main_node_ip>
main_process_port: 29500
```

Then launch on each node:

```bash
# On main node (machine_rank: 0)
accelerate launch --config_file fsdp_config.yaml peft_accelerate.py

# On worker nodes (machine_rank: 1, 2, ...)
accelerate launch --config_file fsdp_config.yaml peft_accelerate.py
```

## CLI Usage Examples

### Basic Usage

```bash
accelerate launch --config_file fsdp_config.yaml peft_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --max_steps 5000
```

### Advanced Configuration

```bash
accelerate launch --config_file fsdp_config.yaml peft_accelerate.py \
  --hf_model_id "meta-llama/Llama-3-8B" \
  --max_steps 10000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_seq_length 4096 \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
  --output_dir "results/llama3_custom"
```

### Available CLI Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--hf_model_id` | str | HuggingFace model name | `Qwen/Qwen3-8B` |
| `--full_ft` | flag | Disable LoRA (full fine-tuning) | `False` |
| `--lora_rank` | int | LoRA rank | `32` |
| `--lora_alpha` | int | LoRA alpha | `32` |
| `--lora_dropout` | float | LoRA dropout | `0.1` |
| `--lora_target_modules` | str | Comma-separated target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| `--max_steps` | int | Maximum training steps | `10000` |
| `--per_device_train_batch_size` | int | Batch size per device | `2` |
| `--per_device_eval_batch_size` | int | Eval batch size per device | `2` |
| `--gradient_accumulation_steps` | int | Gradient accumulation steps | `4` |
| `--learning_rate` | float | Learning rate | `5e-5` |
| `--weight_decay` | float | Weight decay | `0.01` |
| `--warmup_steps` | int | Warmup steps | `100` |
| `--max_grad_norm` | float | Max gradient norm | `1.0` |
| `--max_seq_length` | int | Maximum sequence length | `2048` |
| `--data_dir` | str | Data directory | Auto-generated |
| `--output_dir` | str | Output directory | `results/{hf_model_id}` |
| `--logging_steps` | int | Logging frequency | `10` |
| `--eval_steps` | int | Evaluation frequency | `100` |
| `--max_eval_samples` | int | Max eval samples | `None` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |
| `--early_stopping_threshold` | float | Early stopping threshold | `0.001` |
| `--use_wandb` | flag | Enable Weights & Biases logging | `False` |
| `--seed` | int | Random seed | `42` |
| `--num_workers` | int | Dataloader workers | `8` |

### Dataset Configuration CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--hfdc_dataset_name` | str | HuggingFace dataset name |
| `--hfdc_dataset_config` | str | Dataset configuration/subset |
| `--hfdc_split` | str | Initial split to load |
| `--hfdc_train_split_ratio` | float | Training data ratio |
| `--hfdc_val_test_split_ratio` | float | Val/test split ratio |
| `--hfdc_input_template` | str | Input formatting template |
| `--hfdc_output_template` | str | Output formatting template |
| `--hfdc_field_mapping` | str (JSON) | Field name mapping |
| `--hfdc_num_proc` | int | Number of processes |

## Testing Checkpoints

Test FSDP checkpoints using DeepSpeed tensor parallelism:

```bash
deepspeed --num_gpus=8 test_checkpoint.py \
  --base_model "Qwen/Qwen3-8B" \
  --max_samples 1024 \
  --max_batch_size 8
```

## Converting Checkpoints to HuggingFace Format

Convert FSDP checkpoints to standard HuggingFace format for deployment:

```bash
python convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B"
```

By default, this merges LoRA weights into the base model. To save as a LoRA adapter:

```bash
python convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --no_merge
```

## Project Structure

```
.
├── peft_accelerate.py           # Main training script
├── test_checkpoint.py           # Checkpoint testing script
├── convert_checkpoint_to_hf.py  # Checkpoint conversion script
├── dataset_module.py            # Dataset processing module
├── fsdp_config.yaml             # FSDP configuration
├── README.md                    # This file
├── datasets/                    # Downloaded and processed datasets
│   └── {dataset_name}/
│       └── {dataset_config}/
│           └── train={train_%}-val={val%}-test={test%}/
│               ├── training.jsonl
│               ├── validation.jsonl
│               ├── test.jsonl
│               └── .data_ready
└── results/                     # Training outputs and logs
    └── {hf_model_id}/
        ├── checkpoint-*/         # FSDP checkpoints
        ├── final/                # Final model
        └── logs/                 # TensorBoard logs
            ├── checkpoint-{step}/
            └── final/
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce batch size**
```bash
--per_device_train_batch_size 1
```

**Solution 2: Increase gradient accumulation**
```bash
--gradient_accumulation_steps 16
```

**Solution 3: Reduce sequence length**
```bash
--max_seq_length 1024
```

**Solution 4: Enable CPU offload in accelerate_config.yaml**
```yaml
fsdp_config:
  fsdp_offload_params: true
```

### Data Loading Issues

**Field mapping errors**

Check that template placeholders match dataset columns:
```python
from datasets import load_dataset

ds = load_dataset("your-dataset")
print("Available columns:", ds['train'].column_names)
print("Sample data:", ds['train'][0])
```

### Training Instability

The framework automatically handles:
- Gradient clipping with `max_norm=1.0`
- Cosine learning rate schedule with warmup
- Mixed precision training with BFloat16

## GPU Requirements

### Small Models (1B - 13B parameters)

**Examples**: Qwen3-8B, Llama3-8B, Mistral-7B

**Configuration**:
- **GPUs**: 8x A100 (40GB or 80GB)
- **Batch size**: 2-4 per device
- **Gradient accumulation**: 4-8

### Medium Models (13B - 34B parameters)

**Examples**: Llama2-13B, Yi-34B

**Configuration**:
- **GPUs**: 16x A100 (80GB) total (2 nodes)
- **Batch size**: 1-2 per device
- **Gradient accumulation**: 8-16

### Large Models (34B - 100B parameters)

**Examples**: Llama3.1-70B, Mixtral-8x22B

**Configuration**:
- **GPUs**: 32-64x A100 (80GB) or H100 (80GB)
- **Batch size**: 1 per device
- **Gradient accumulation**: 16-32
- **CPU Offload**: Consider enabling

## Additional Notes

### Flash Attention

The training script uses `attn_implementation="flash_attention_2"` for improved stability and performance. Ensure flash-attn is installed.

### Checkpoint Format

- Checkpoints are saved in HuggingFace format
- LoRA adapters can be loaded with PEFT library
- Checkpoints include both model weights and tokenizer

### Memory Optimization

- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set for better memory management
- FSDP uses FULL_SHARD strategy for maximum memory efficiency
- Gradient checkpointing is enabled by default
- CPU offload available via configuration
