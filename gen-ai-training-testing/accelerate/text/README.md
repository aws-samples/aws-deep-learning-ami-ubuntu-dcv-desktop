# Text-Only LLM Training

> **Note**: For installation instructions, see the [parent README](../README.md). For multi-modal training (vision-language models), see [../multimodal/](../multimodal/).

This directory provides a flexible framework for fine-tuning Large Language Models, training Reward Models, and performing alignment using PPO-based RLHF or Direct Preference Optimization (DPO) with Hugging Face Trainer, Accelerate, and FSDP (Fully Sharded Data Parallel).

## Features

- **Continual Pre-Training**: Extend model knowledge with domain-specific corpora (full causal LM objective)
- **Complete RLHF Pipeline**: SFT → Reward Model → PPO policy optimization
- **DPO Pipeline**: SFT → DPO preference optimization (simpler, no reward model needed)
- **Generalized HuggingFace Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates and field mapping
- **Distributed Training**: Multi-node, multi-GPU training with FSDP for efficient memory usage
- **LoRA and Full Fine-Tuning**: Support for LoRA parameter-efficient fine-tuning or full fine-tuning
- **Automatic Data Conversion**: Converts HuggingFace datasets to JSONL format for efficient loading
- **Customizable Training**: Extensive configuration options for hyperparameters and training strategies
- **Flash Attention 2**: Optimized attention implementation for faster training
- **Gradient Checkpointing**: Reduce memory usage for large models
- **Early Stopping**: Automatic training termination based on validation loss

## Table of Contents

- [Quick Start](#quick-start)
- [Continual Pre-Training (CPT)](#continual-pre-training-cpt)
- [Alignment Methods](#alignment-methods)
- [Reward Model Training](#reward-model-training)
- [Training Configuration](#training-configuration)
- [Using Different Datasets](#using-different-datasets)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Quick Start

After building and running the Docker container (see [parent README](../README.md)), navigate to the accelerate directory:

```bash
cd /app
```

### DPO Pipeline (Recommended)

Run the DPO pipeline (SFT → DPO) with a single command:

```bash
bash text/run_dpo_pipeline.sh
```

This will:
1. Train SFT model on Dolphin dataset
2. Convert SFT checkpoint to HuggingFace format
3. Train DPO policy using preference data (Anthropic/hh-rlhf)

### PPO-RLHF Pipeline

Run the complete RLHF pipeline (SFT → Reward Model → PPO):

```bash
bash text/run_ppo_pipeline.sh
```

This will:
1. Train SFT model on Dolphin dataset
2. Convert SFT checkpoint to HuggingFace format
3. Train Reward Model on Anthropic/hh-rlhf dataset
4. Convert Reward Model checkpoint to HuggingFace format
5. Train PPO policy using the SFT and Reward models

### Supervised Fine-Tuning (SFT) Only

Train only the SFT model:

```bash
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py
```

## Continual Pre-Training (CPT)

Continual pre-training extends a pre-trained model's knowledge by training on domain-specific corpora using the standard causal language modeling objective. Unlike SFT, all tokens are training targets (no label masking), and training is epoch-based with regular checkpoint saving for resumability.

### Quick Start

```bash
accelerate launch --config_file cpt_accelerate_config.yaml text/cpt_accelerate.py
```

This will train `Qwen/Qwen3-8B` on `wikimedia/wikipedia` (English, 20231101 snapshot) for 3 epochs with checkpoints saved every 1000 steps.

### CPT with Custom Domain Data

```bash
accelerate launch --config_file cpt_accelerate_config.yaml text/cpt_accelerate.py \
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
accelerate launch --config_file cpt_accelerate_config.yaml text/cpt_accelerate.py \
  --resume_from_checkpoint "results/Qwen_Qwen3.5-2B-cpt/checkpoint-2000"
```

### CPT vs SFT

| Aspect | CPT (`cpt_accelerate.py`) | SFT (`peft_accelerate.py`) |
|--------|---------------------------|----------------------------|
| Objective | Causal LM on all tokens | Causal LM on output tokens only |
| Label masking | None (all tokens are targets) | Input/instruction tokens masked |
| Fine-tuning | Full model weights | LoRA or full |
| Training schedule | Epoch-based | Step-based |
| Checkpointing | Regular interval (`save_steps`) | Best metric only |
| Typical data | Raw domain text | Instruction/response pairs |
| Learning rate | Lower (2e-5) | Higher (5e-5) |
| Warmup | Longer (1000 steps) | Shorter (100 steps) |

### CPT CLI Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--hf_model_id` | str | HuggingFace model name | `Qwen/Qwen3-8B` |
| `--num_train_epochs` | int | Number of training epochs | `3` |
| `--per_device_train_batch_size` | int | Batch size per device | `2` |
| `--gradient_accumulation_steps` | int | Gradient accumulation steps | `4` |
| `--learning_rate` | float | Learning rate | `2e-5` |
| `--warmup_steps` | int | Warmup steps | `1000` |
| `--max_seq_length` | int | Maximum sequence length | `4096` |
| `--save_steps` | int | Checkpoint save frequency | `1000` |
| `--save_total_limit` | int | Max checkpoints to keep | `2` |
| `--eval_steps` | int | Evaluation frequency | `1000` |
| `--data_dir` | str | Data directory | Auto-generated |
| `--output_dir` | str | Output directory | `results/{model}-cpt` |
| `--resume_from_checkpoint` | str | Path to checkpoint to resume from | `None` |
| `--use_wandb` | flag | Enable Weights & Biases logging | `False` |

CPT also accepts `--hfdc_*` dataset arguments (see [Using Different Datasets](#using-different-datasets)).

## Alignment Methods

### DPO Pipeline (Recommended)

Direct Preference Optimization (DPO) is a simpler and more memory-efficient alternative to PPO-RLHF. It directly optimizes the policy using preference data without requiring a separate reward model.

**Advantages of DPO:**
- Simpler: No reward model training required
- More stable: Direct optimization without RL complexity
- Memory efficient: Only needs policy and reference models
- Faster: Fewer training steps and components

#### Using run_dpo_pipeline.sh

```bash
# Run full DPO pipeline with default settings
bash text/run_dpo_pipeline.sh

# Run with custom model
bash text/run_dpo_pipeline.sh --base-model "meta-llama/Llama-3-8B"

# Skip specific steps
bash text/run_dpo_pipeline.sh --skip-sft  # Use existing SFT checkpoint

# Pass additional arguments to training scripts
bash text/run_dpo_pipeline.sh --max_steps 5000 --beta 0.2
```

#### Manual DPO Execution

```bash
# Step 1: Supervised Fine-Tuning
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B"

# Step 2: Convert SFT checkpoint
python shared/convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B"

# Step 3: DPO Training
accelerate launch --config_file peft_accelerate_config.yaml text/dpo_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --beta 0.1 \
  --learning_rate 5e-7
```

#### DPO Configuration

Key DPO hyperparameters:

- `sft_model_path`: Path to SFT checkpoint (used as policy initialization)
- `beta`: KL divergence penalty coefficient (default: 0.1, typical range: 0.1-0.5)
- `learning_rate`: Learning rate (default: 5e-7, lower than SFT)
- `max_steps`: Training steps (default: 10000)
- `rmdc_dataset_name`: Preference dataset (default: "Anthropic/hh-rlhf")

**Dataset Format**: DPO uses preference datasets with `chosen` and `rejected` responses (same format as reward model training).

### PPO-RLHF Pipeline

**⚠️ Note**: PPO is provided as a reference implementation but may encounter OOM (Out of Memory) errors on systems with limited GPU memory. PPO requires running multiple models simultaneously:
- Policy model (trainable, on GPU)
- Reference model (frozen, on CPU)
- Reward model (frozen, on CPU)
- vLLM inference engine (distributed across GPUs)

This multi-model setup requires significant memory. **For most use cases, DPO is recommended** as it's simpler and more memory-efficient.

#### Using run_ppo_pipeline.sh

The easiest way to run the complete pipeline:

```bash
# Run full pipeline with default settings
bash text/run_ppo_pipeline.sh

# Run with custom model
bash text/run_ppo_pipeline.sh --base-model "meta-llama/Llama-3-8B"

# Skip specific steps
bash text/run_ppo_pipeline.sh --skip-sft  # Use existing SFT checkpoint
bash text/run_ppo_pipeline.sh --skip-reward  # Use existing reward model

# Pass additional arguments to training scripts
bash text/run_ppo_pipeline.sh --max_steps 5000 --learning_rate 1e-4
```

### Manual Step-by-Step Execution

Alternatively, run each step manually:

```bash
# Step 1: Supervised Fine-Tuning
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B"

# Step 2: Convert SFT checkpoint
python shared/convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B"

# Step 3: Train Reward Model
accelerate launch --config_file peft_accelerate_config.yaml text/reward_model_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B"

# Step 4: Convert Reward Model checkpoint
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --checkpoints_dir "results/reward_Qwen/Qwen3-8B"

# Step 5: PPO Policy Optimization
accelerate launch --config_file peft_accelerate_config.yaml text/ppo_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B"
```

## Reward Model Training

Train a reward model for RLHF. The reward model automatically uses the latest converted SFT checkpoint:

```bash
# Train from latest SFT checkpoint (automatic)
accelerate launch --config_file peft_accelerate_config.yaml text/reward_model_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B"

# Train from specific SFT checkpoint
accelerate launch --config_file peft_accelerate_config.yaml text/reward_model_accelerate.py \
  --sft_model_path "results/Qwen/Qwen3-8B/checkpoint-1000.hf_model"

# Use different reward dataset
accelerate launch --config_file peft_accelerate_config.yaml text/reward_model_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --rmdc_dataset_name "OpenAssistant/oasst1"
```

### Reward Dataset Format

Reward datasets support two formats:

**Format 1: Separate input field**
```json
{"input": "prompt", "chosen": "good response", "rejected": "bad response"}
```

**Format 2: Combined format (no separate input)**
```json
{"chosen": "prompt + good response", "rejected": "prompt + bad response"}
```

Use custom converters for different dataset structures:

```python
# Example: Anthropic HH-RLHF converter
def hh_rlhf_converter(sample):
    return {
        "chosen": sample["chosen"],
        "rejected": sample["rejected"]
    }

# Pass to RMDatasetConfig
config = RMDatasetConfig(
    dataset_name="Anthropic/hh-rlhf",
    custom_converter=hh_rlhf_converter
)
```

See `rm_dataset_module.py` RewardModelDataset docstring for more converter examples.

## DPO vs PPO: Which to Choose?

| Aspect | DPO | PPO-RLHF |
|--------|-----|----------|
| **Complexity** | Simple (2 models) | Complex (4 models) |
| **Memory** | Lower | Higher (may OOM) |
| **Training Speed** | Faster | Slower |
| **Stability** | More stable | Can be unstable |
| **Dataset** | Preference pairs | Prompts + Reward model |
| **Recommended For** | Most use cases | Research/experimentation |

**Recommendation**: Use DPO unless you specifically need PPO for research purposes or have a well-tuned reward model.

## Training Configuration

### Supported Models

The framework supports any HuggingFace causal language model. Recommended:

- **Qwen3 Family**: `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-70B`

### Training Example

```bash
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B" \
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
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py \
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

The `peft_accelerate_config.yaml` and `cpt_accelerate_config.yaml` files configure the distributed training setup:

- **FSDP Strategy**: FULL_SHARD for maximum memory efficiency
- **Mixed Precision**: BFloat16 for training stability
- **Backward Prefetch**: BACKWARD_PRE for improved performance
- **Number of Processes**: Set to match your GPU count (default: 8)

### Multi-Node Training

For multi-node training, update the `peft_accelerate_config.yaml` or `cpt_accelerate_config.yaml`:

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
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py

# On worker nodes (machine_rank: 1, 2, ...)
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py
```

## CLI Usage Examples

### Basic Usage

```bash
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --max_steps 5000
```

### Advanced Configuration

```bash
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py \
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
| `--max_eval_samples` | int | Max eval samples | `640` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |
| `--early_stopping_threshold` | float | Early stopping threshold | `0.001` |
| `--use_wandb` | flag | Enable Weights & Biases logging | `False` |
| `--seed` | int | Random seed | `42` |
| `--num_workers` | int | Dataloader workers | `8` |

### SFT Dataset Configuration CLI Arguments

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

### Reward Model Dataset Configuration CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--rmdc_dataset_name` | str | HuggingFace dataset name |
| `--rmdc_dataset_config` | str | Dataset configuration/subset |
| `--rmdc_split` | str | Initial split to load |
| `--rmdc_train_split_ratio` | float | Training data ratio |
| `--rmdc_val_test_split_ratio` | float | Val/test split ratio |
| `--rmdc_num_proc` | int | Number of processes |

## Testing Checkpoints

Test FSDP checkpoints using vLLM for efficient inference:

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

Convert FSDP checkpoints to standard HuggingFace format for deployment:

```bash
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B"
```

The script automatically finds the latest checkpoint in `results/{base_model}/`. By default, it merges LoRA weights into the base model for maximum compatibility. To save as a LoRA adapter:

```bash
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --no_merge
```

## Project Structure

```
text/
├── peft_accelerate.py           # SFT training script
├── cpt_accelerate.py            # Continual pre-training script
├── reward_model_accelerate.py   # Reward model training script
├── dpo_accelerate.py            # DPO training script
├── ppo_accelerate.py            # PPO policy training script
├── dataset_module.py            # SFT/CPT dataset processing module
├── rm_dataset_module.py         # Reward Model dataset processing module
├── run_dpo_pipeline.sh          # DPO pipeline script (recommended)
├── run_ppo_pipeline.sh          # PPO-RLHF pipeline script
└── README.md                    # This file

../shared/
├── callbacks.py                 # Training callbacks
├── convert_checkpoint_to_hf.py  # Checkpoint conversion script
└── test_checkpoint.py           # Checkpoint testing script

../
├── peft_accelerate_config.yaml  # Accelerate FSDP configuration for PEFT/SFT
├── cpt_accelerate_config.yaml   # Accelerate FSDP configuration for CPT
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
        ├── checkpoint-*/         # Training checkpoints
        │   └── pytorch_model_fsdp_0/  # FSDP checkpoint files
        ├── final/                # Final model
        └── logs/                 # TensorBoard logs
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

**Solution 4: Enable CPU offload in peft_accelerate_config.yaml or cpt_accelerate_config.yaml**
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

**Examples**: Qwen3-8B

**Configuration**:
- **GPUs**: 8x A100 (40GB or 80GB)
- **Batch size**: 2-4 per device
- **Gradient accumulation**: 4-8

### Medium Models (13B - 34B parameters)

**Examples**: Qwen3-14B

**Configuration**:
- **GPUs**: 16x A100 (80GB) total (2 nodes)
- **Batch size**: 1-2 per device
- **Gradient accumulation**: 8-16

### Large Models (34B - 100B parameters)

**Examples**: Qwen3-70B

**Configuration**:
- **GPUs**: 32-64x A100 (80GB) or H100 (80GB)
- **Batch size**: 1 per device
- **Gradient accumulation**: 16-32

## Additional Notes

### Flash Attention

The training script uses `attn_implementation="flash_attention_2"` for improved stability and performance. Ensure flash-attn is installed.

### Checkpoint Format

- Checkpoints are saved in FSDP format under `pytorch_model_fsdp_0/` directory
- Use `shared/convert_checkpoint_to_hf.py` to convert to standard HuggingFace format
- LoRA adapters can be merged or saved separately
- Final model includes both model weights and tokenizer

### Memory Optimization

- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set for better memory management
- FSDP uses FULL_SHARD strategy for maximum memory efficiency
- Gradient checkpointing is enabled by default
- CPU offload available via peft_accelerate_config.yaml or cpt_accelerate_config.yaml configuration

### Training Features

- **Early Stopping**: Automatically stops training when validation loss stops improving
- **Best Model Saving**: Only saves checkpoints when validation loss improves
- **Data Collator**: Uses DataCollatorForSeq2Seq for robust padding with label masking
