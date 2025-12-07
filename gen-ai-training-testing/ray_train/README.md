# Ray Train Parameter-Efficient Fine-Tuning (PEFT) Framework

This project provides a framework for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models using Ray Train and FSDP (Fully Sharded Data Parallel). The framework provides a generalized data pipeline for HuggingFace datasets and streamlined configuration for distributed training with LoRA.

## Features

- **Ray Train Integration**: Distributed training with Ray's native orchestration
- **Generalized HuggingFace Dataset Support**: Easy integration with any HuggingFace dataset
- **FSDP Support**: Multi-GPU training with FSDP for efficient memory usage
- **PEFT Methods**: Support for LoRA via HuggingFace PEFT
- **Automatic Data Conversion**: Converts HuggingFace datasets to JSONL format
- **Flash Attention 2**: Optimized attention implementation
- **Gradient Checkpointing**: Reduce memory usage for large models

## Prerequisites

Follow [Step by Step Tutorial](../../README.md) to launch a Deep Learning Desktop. On the desktop:

```bash
cd ~
git clone https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop.git
cd ~/aws-deep-learning-ami-ubuntu-dcv-desktop/gen-ai-training-testing/ray_train
```

## Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker buildx build -t ray_train:latest -f ../containers/Dockerfile.ray_train .
```

2. Run the container with GPU support:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  --shm-size=32g \
  ray_train:latest
```

## Quick Start

Train the default Qwen3-8B model on the Dolphin dataset:

```bash
python ray_train_sft.py
```

This will:
1. Download the Qwen/Qwen3-8B model from HuggingFace
2. Load and process the Dolphin dataset
3. Start LoRA fine-tuning with available GPUs using FSDP

## Training Different Models

### Supported Models

The framework supports any HuggingFace causal language model:

- **Qwen Family**: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-70B`
- **Llama Family**: `meta-llama/Llama-3-8B`, `meta-llama/Meta-Llama-3.1-8B`, `meta-llama/Meta-Llama-3.1-70B`
- **Mistral**: `mistralai/Mistral-7B-v0.1`, `mistralai/Mixtral-8x7B-v0.1`
- **Phi**: `microsoft/Phi-3-medium-4k-instruct`

### Training a Different Model

```bash
python ray_train_sft.py \
  --hf_model_id "meta-llama/Llama-3-8B" \
  --max_steps 5000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8
```

## Using Different Datasets

The framework uses `HFDatasetConfig` to define dataset loading and formatting. Modify the configuration in `ray_train_sft.py`:

```python
@dataclass
class Config:
    hf_dataset_config: HFDatasetConfig = field(default_factory=lambda: HFDatasetConfig(
        dataset_name="your-org/your-dataset",
        dataset_config="subset-name",  # Optional
        split="train",
        train_split_ratio=0.9,
        val_test_split_ratio=0.5,
        input_template="Your input format: {field1}\\n{field2}\\n",
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

### Small Models (1B - 13B parameters)

**Examples**: Qwen3-8B, Llama3-8B, Mistral-7B

**Configuration**:
- **GPUs**: 8x A100 (40GB or 80GB)
- **Batch size**: 2-4
- **Gradient accumulation**: 4-8

```bash
python ray_train_sft.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --max_steps 10000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4
```

### Medium Models (13B - 34B parameters)

**Examples**: Llama2-13B, Yi-34B

**Configuration**:
- **GPUs**: 16x A100 (80GB)
- **Batch size**: 1-2
- **Gradient accumulation**: 8-16

### Large Models (34B - 100B parameters)

**Examples**: Llama3.1-70B, Mixtral-8x22B

**Configuration**:
- **GPUs**: 32-64x A100 (80GB) or H100 (80GB)
- **Batch size**: 1
- **Gradient accumulation**: 16-32

## Configuration

### Core Training Parameters

Key parameters in the Config class:

- **Model**: `hf_model_id` - HuggingFace model identifier
- **Paths**: 
  - `data_dir`: Directory for processed datasets (auto-generated)
  - `results_dir`: Base directory for training outputs
  - `logging_dir`: TensorBoard logging directory (auto-generated)
- **Training**: `max_steps`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`
- **Optimizer**: `weight_decay`, `warmup_ratio`, `lr_scheduler_type`, `max_grad_norm`
- **LoRA**: `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules`, `full_ft`
- **Sequence**: `max_seq_length` - Maximum sequence length
- **Logging**: `logging_steps`, `save_steps`, `eval_steps`, `save_total_limit`, `max_eval_samples`
- **Early Stopping**: `early_stopping_patience`, `early_stopping_threshold`
- **Other**: `seed`, `dataloader_num_workers`, `remove_unused_columns`

### Full Fine-Tuning vs LoRA

By default, the framework uses LoRA (`full_ft=False`). To perform full fine-tuning:

```bash
python ray_train_sft.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --full_ft
```

### Early Stopping and Best Model Saving

The framework includes:
- **SaveOnBestMetricCallback**: Saves checkpoints only when `eval_loss` improves
- **EarlyStoppingCallback**: Stops training if no improvement for `early_stopping_patience` evaluations
- Default settings: patience=3, threshold=0.001
- Evaluation frequency controlled by `eval_steps` (default: 100)

### Limiting Validation Samples

To speed up evaluation, limit validation samples:

```bash
python ray_train_sft.py \
  --max_eval_samples 1000
```

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
- Finds the latest checkpoint in `results/{base_model}/`
- Discovers the latest `test.jsonl` file under `datasets/`
- Loads the checkpoint and merges LoRA weights (if applicable)
- Uses vLLM for fast batched inference
- Evaluates predictions using BERTScore

### Converting to Hugging Face Format

Convert your checkpoint to standard Hugging Face format:

```bash
python convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B"
```

The script automatically finds the latest checkpoint in `results/{base_model}/`. By default, it merges LoRA weights into the base model for maximum compatibility.

**LoRA Merging:**
- By default, LoRA weights are **merged** into the base model
- Merged models work with vLLM, TGI, and all Hugging Face tools
- Use `--no_merge` to save as a separate LoRA adapter

## Project Structure

```
.
├── ray_train_sft.py              # Main training script
├── dataset_module.py             # Dataset processing module
├── test_checkpoint.py            # Test checkpoint script
├── convert_checkpoint_to_hf.py   # Convert to Hugging Face format
├── README.md                     # This file
├── datasets/                     # Downloaded and processed datasets
│   └── {dataset_name}/
│       └── {dataset_config}/
│           └── train={train_%}-val={val%}-test={test%}/
│               ├── training.jsonl
│               ├── validation.jsonl
│               ├── test.jsonl
│               └── .data_ready
└── results/                      # Training outputs
    └── {hf_model_id}/
        ├── checkpoint-*/          # Best model checkpoints
        │   ├── adapter_config.json
        │   ├── adapter_model.safetensors
        │   └── ...
        └── logs/                  # TensorBoard logs
```

## CLI Usage Examples

### Basic Usage

```bash
python ray_train_sft.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --max_steps 5000
```

### Advanced Configuration

```bash
python ray_train_sft.py \
  --hf_model_id "meta-llama/Llama-3-8B" \
  --max_steps 10000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_seq_length 4096 \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_eval_samples 1000 \
  --early_stopping_patience 5
```

### Available CLI Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--hf_model_id` | str | HuggingFace model name | `Qwen/Qwen3-8B` |
| `--full_ft` | bool | Enable full fine-tuning (disables LoRA) | `False` |
| `--lora_rank` | int | LoRA rank | `32` |
| `--lora_alpha` | int | LoRA alpha | `32` |
| `--lora_dropout` | float | LoRA dropout | `0.1` |
| `--lora_target_modules` | str | Comma-separated target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| `--max_steps` | int | Maximum training steps | `10000` |
| `--per_device_train_batch_size` | int | Batch size per device | `2` |
| `--per_device_eval_batch_size` | int | Eval batch size per device | `2` |
| `--gradient_accumulation_steps` | int | Gradient accumulation steps | `4` |
| `--learning_rate` | float | Learning rate | `5e-5` |
| `--min_learning_rate` | float | Minimum learning rate | `1e-6` |
| `--weight_decay` | float | Weight decay | `0.01` |
| `--warmup_ratio` | float | Warmup ratio | `0.03` |
| `--max_grad_norm` | float | Max gradient norm | `1.0` |
| `--lr_scheduler_type` | str | LR scheduler type | `cosine` |
| `--max_seq_length` | int | Maximum sequence length | `2048` |
| `--data_dir` | str | Data directory | Auto-generated |
| `--results_dir` | str | Results directory | `results` |
| `--logging_dir` | str | Logging directory | Auto-generated |
| `--logging_steps` | int | Logging frequency | `10` |
| `--save_steps` | int | Save frequency | `100` |
| `--eval_steps` | int | Evaluation frequency | `100` |
| `--save_total_limit` | int | Max checkpoints to keep | `2` |
| `--max_eval_samples` | int | Max eval samples | `None` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |
| `--early_stopping_threshold` | float | Early stopping threshold | `0.001` |
| `--seed` | int | Random seed | `16257` |
| `--dataloader_num_workers` | int | Dataloader workers | `4` |
| `--remove_unused_columns` | bool | Remove unused columns | `False` |

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

**Solution 4: Limit validation samples**
```bash
--max_eval_samples 500
```

### Data Loading Issues

Check dataset structure:
```python
from datasets import load_dataset

ds = load_dataset("your-dataset")
print("Available columns:", ds['train'].column_names)
print("Sample data:", ds['train'][0])
```

### Ray Initialization Issues

If Ray fails to initialize:
```bash
# Start Ray cluster manually
ray start --head

# Then run training
python ray_train_sft.py
```

## Additional Notes

### Flash Attention
The training script uses `attn_implementation="flash_attention_2"` for improved performance. Ensure flash-attn is installed.

### Memory Optimization
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set for better memory management
- FSDP uses FULL_SHARD strategy for maximum memory efficiency
- Gradient checkpointing applied to reduce memory usage

### Ray Train Benefits
- Automatic fault tolerance with configurable max failures
- Distributed checkpointing
- Native integration with Ray ecosystem
- Flexible resource management

### Training Features
- **FSDP**: Full sharding for memory efficiency
- **Flash Attention 2**: Optimized attention implementation
- **Gradient Checkpointing**: Reduces memory usage
- **Early Stopping**: Automatic stopping when validation loss plateaus
- **Best Model Saving**: Only saves checkpoints that improve validation loss
- **Cosine LR Schedule**: With warmup for stable training
- **Mixed Precision**: BFloat16 for faster training
