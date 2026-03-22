# Text-Only LLM Training with Ray Train

> **Note**: For installation instructions, see the [parent README](../README.md). For multi-modal training (vision-language models), see [../multimodal/](../multimodal/).

This directory provides a flexible framework for fine-tuning Large Language Models using Ray Train's `TorchTrainer` with FSDP (Fully Sharded Data Parallel), automatic fault tolerance, and checkpoint management.

## Features

- **Continual Pre-Training (CPT)**: Extend model knowledge with domain-specific corpora (full causal LM objective)
- **Supervised Fine-Tuning (SFT)**: LoRA or full fine-tuning for causal LMs
- **Generalized HuggingFace Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates and field mapping
- **Distributed Training**: Multi-GPU training with FSDP via Ray Train's `TorchTrainer`
- **LoRA and Full Fine-Tuning**: Support for LoRA parameter-efficient fine-tuning or full fine-tuning
- **Automatic Data Conversion**: Converts HuggingFace datasets to JSONL format for efficient loading
- **Flash Attention 2**: Optimized attention implementation for faster training
- **Gradient Checkpointing**: Reduce memory usage for large models
- **Early Stopping**: Automatic training termination based on validation loss
- **Fault Tolerance**: Automatic restart on failure via Ray Train

## Quick Start

After building and running the Docker container (see [parent README](../README.md)), navigate to the ray_train directory:

```bash
cd /app
```

### Supervised Fine-Tuning (SFT)

```bash
python text/ray_train_sft.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --hfdc_dataset_name "cognitivecomputations/dolphin" \
  --hfdc_dataset_config "flan1m-alpaca-uncensored" \
  --max_steps 5000
```

### Advanced Configuration

```bash
python text/ray_train_sft.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --max_steps 10000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_seq_length 4096 \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj"
```

## Continual Pre-Training (CPT)

Continual pre-training extends a pre-trained model's knowledge by training on domain-specific corpora using the standard causal language modeling objective. Unlike SFT, all tokens are training targets (no label masking), and training is epoch-based with regular checkpoint saving for resumability.

### Quick Start

```bash
python text/cpt_ray_train.py
```

This will train `Qwen/Qwen3-8B` on `wikimedia/wikipedia` (English, 20231101 snapshot) for 3 epochs with checkpoints saved every 1000 steps.

### CPT with Custom Domain Data

```bash
python text/cpt_ray_train.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --hfdc_dataset_name "your-org/domain-corpus" \
  --hfdc_output_template "{text}" \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --max_seq_length 4096 \
  --save_steps 500
```

### Resuming from Checkpoint

CPT supports resuming from a previously saved checkpoint:

```bash
python text/cpt_ray_train.py \
  --resume_from_checkpoint "results/checkpoint-2000"
```

### CPT vs SFT

| Aspect | CPT (`cpt_ray_train.py`) | SFT (`ray_train_sft.py`) |
|--------|--------------------------|--------------------------|
| Objective | Causal LM on all tokens | Causal LM on output tokens only |
| Label masking | None (all tokens are targets) | Input/instruction tokens masked |
| Fine-tuning | Full model weights | LoRA or full |
| Training schedule | Epoch-based | Step-based |
| Checkpointing | Regular interval (`save_steps`) | Best metric only |
| Typical data | Raw domain text | Instruction/response pairs |
| Learning rate | Lower (2e-5) | Higher (5e-5) |
| Warmup | Longer (1000 steps) | Shorter (warmup_ratio=0.03) |

### CPT CLI Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--hf_model_id` | str | HuggingFace model name | `Qwen/Qwen3-8B` |
| `--num_train_epochs` | int | Number of training epochs | `3` |
| `--per_device_train_batch_size` | int | Batch size per device | `1` |
| `--gradient_accumulation_steps` | int | Gradient accumulation steps | `8` |
| `--learning_rate` | float | Learning rate | `2e-5` |
| `--warmup_steps` | int | Warmup steps | `1000` |
| `--max_seq_length` | int | Maximum sequence length | `4096` |
| `--save_steps` | int | Checkpoint save frequency | `1000` |
| `--save_total_limit` | int | Max checkpoints to keep | `2` |
| `--eval_steps` | int | Evaluation frequency | `1000` |
| `--data_dir` | str | Data directory | Auto-generated |
| `--results_dir` | str | Results directory | `results` |
| `--resume_from_checkpoint` | str | Path to checkpoint to resume from | `None` |
| `--use_wandb` | flag | Enable Weights & Biases logging | `False` |

CPT also accepts `--hfdc_*` dataset arguments (see [Dataset Configuration CLI Arguments](#dataset-configuration-cli-arguments)).

## Training Configuration

### Supported Models

The framework supports any HuggingFace causal language model. Recommended:

- **Qwen3 Family**: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-70B`

### Using Different Datasets

The framework uses `HFDatasetConfig` to define dataset loading and formatting. Key parameters:

- `dataset_name`: HuggingFace dataset identifier
- `dataset_config`: Specific subset/configuration
- `input_template`: Format string for input prompts
- `output_template`: Format string for output completions
- `field_mapping`: Maps template variables to dataset columns
- `num_proc`: Number of processes for dataset loading (default: 8)

#### Example: Custom Dataset

```bash
python text/ray_train_sft.py \
  --hfdc_dataset_name "databricks/databricks-dolly-15k" \
  --hfdc_split "train" \
  --hfdc_train_split_ratio 0.95 \
  --hfdc_val_test_split_ratio 0.5 \
  --hfdc_input_template "### Instruction:\n{instruction}\n### Context:\n{context}\n" \
  --hfdc_output_template "### Response:\n{response}" \
  --hfdc_field_mapping '{"instruction": "instruction", "context": "context", "response": "response"}'
```

## CLI Arguments

### Core Training Arguments

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
| `--warmup_ratio` | float | Warmup ratio | `0.03` |
| `--max_grad_norm` | float | Max gradient norm | `1.0` |
| `--max_seq_length` | int | Maximum sequence length | `2048` |
| `--data_dir` | str | Data directory | Auto-generated |
| `--results_dir` | str | Results directory | `results` |
| `--logging_steps` | int | Logging frequency | `10` |
| `--save_steps` | int | Save frequency | `100` |
| `--eval_steps` | int | Evaluation frequency | `100` |
| `--max_eval_samples` | int | Max eval samples | `640` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |
| `--early_stopping_threshold` | float | Early stopping threshold | `0.001` |
| `--use_wandb` | flag | Enable Weights & Biases logging | `False` |
| `--seed` | int | Random seed | `16257` |

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

Test Ray Train checkpoints using vLLM for efficient inference:

```bash
python shared/test_checkpoint.py \
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

## Converting Checkpoints to HuggingFace Format

Convert Ray Train FSDP checkpoints to standard HuggingFace format for deployment:

```bash
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B"
```

The script automatically finds the latest checkpoint in `results/{base_model}/`. By default, it merges LoRA weights into the base model. To save as a LoRA adapter:

```bash
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --no_merge
```

## Project Structure

```
text/
├── ray_train_sft.py             # SFT training script
├── cpt_ray_train.py             # Continual pre-training script
├── dataset_module.py            # Dataset processing module (SFT + CPT)
└── README.md                    # This file

shared/
├── convert_checkpoint_to_hf.py  # Checkpoint conversion script
└── test_checkpoint.py           # Checkpoint testing script

datasets/                        # Downloaded and processed datasets
└── {dataset_name}/
    └── {dataset_config}/
        └── train={train_%}-val={val%}-test={test%}/
            ├── training.jsonl
            ├── validation.jsonl
            ├── test.jsonl
            └── .data_ready
results/                         # Training outputs and logs
└── {hf_model_id}/
    ├── TorchTrainer_*/      # Ray Train run directory
    │   └── checkpoint_*/    # Ray Train checkpoints
    └── logs/                # TensorBoard logs
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

### Data Loading Issues

**Field mapping errors**

Check that template placeholders match dataset columns:
```python
from datasets import load_dataset

ds = load_dataset("your-dataset")
print("Available columns:", ds['train'].column_names)
print("Sample data:", ds['train'][0])
```

## GPU Requirements

### Small Models (1B - 13B parameters)
- **GPUs**: 8x A100 (40GB or 80GB)
- **Batch size**: 2-4 per device
- **Gradient accumulation**: 4-8

### Medium Models (13B - 34B parameters)
- **GPUs**: 16x A100 (80GB) total (2 nodes)
- **Batch size**: 1-2 per device
- **Gradient accumulation**: 8-16

### Large Models (34B - 100B parameters)
- **GPUs**: 32-64x A100 (80GB) or H100 (80GB)
- **Batch size**: 1 per device
- **Gradient accumulation**: 16-32

## Additional Notes

### Flash Attention

The training script uses `attn_implementation="flash_attention_2"` for improved stability and performance. Ensure flash-attn is installed.

### Checkpoint Format

- Checkpoints are saved in Ray Train format under `TorchTrainer_*/checkpoint_*/checkpoint/`
- Use `shared/convert_checkpoint_to_hf.py` to convert to standard HuggingFace format
- LoRA adapters can be merged or saved separately
- Final model includes both model weights and tokenizer

### Memory Optimization

- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set for better memory management
- FSDP uses FULL_SHARD strategy for maximum memory efficiency
- Gradient checkpointing is enabled by default
