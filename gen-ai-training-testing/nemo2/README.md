# NeMo 2.0  Parameter-Efficient Fine-Tuning (PEFT) Flexible Framework

This project provides a flexible framework for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models using NVIDIA NeMo 2.0 and Megatron-LM. The framework provides a generalized data pipeline for HuggingFace datasets and streamlined configuration for distributed training with LoRA and other PEFT methods.

## Features

- **Generalized HuggingFace Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates and field mapping
- **Distributed Training**: Multi-node, multi-GPU training with tensor and pipeline parallelism
- **PEFT Methods**: Support for LoRA and other parameter-efficient fine-tuning schemes
- **Automatic Data Conversion**: Converts HuggingFace datasets to NeMo-compatible JSONL format
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
  --shm-size=32g \
  nemo2:latest
```

## Quick Start

Train the default Qwen3-8B model on the Dolphin dataset with optimal settings:

```bash
python peft_megatron.py
```

This will:
1. Download the Qwen/Qwen3-8B model from HuggingFace
2. Convert it to NeMo checkpoint format
3. Load and process the Dolphin dataset
4. Start LoRA fine-tuning with 8 GPUs

## Training Different Models

### Supported Models

The framework supports any model available in NeMo's recipe collection. Common examples include:

- **Qwen Family**: `qwen3_8b`, `qwen3_14b`, `qwen3_70b`
- **Llama Family**: `llama3_8b`, `llama3_70b`, `llama3_1_8b`, `llama3_1_70b`, `llama3_1_405b`
- **Mistral**: `mistral_7b`, `mixtral_8x7b`, `mixtral_8x22b`
- **Nemotron**: `nemotron3_8b`, `nemotron4_340b`

### Finding Recipe Names

NeMo recipe names typically follow the pattern: `{model_family}_{size}`. To find available recipes, check:
```python
from nemo.collections.llm import recipes
# Available recipes are in: nemo.collections.llm.recipes.*
```

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
python peft_megatron.py \
  --hf_model_id "Qwen/Qwen3-8B" \
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
python peft_megatron.py \
  --hf_model_id "meta-llama/Llama-2-13b-hf" \
  --recipe_cls_name "llama2_13b" \
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
python peft_megatron.py \
  --hf_model_id "meta-llama/Meta-Llama-3.1-70B" \
  --recipe_cls_name "llama3_1_70b" \
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
python peft_megatron.py \
  --hf_model_id "meta-llama/Meta-Llama-3.1-405B" \
  --recipe_cls_name "llama3_1_405b" \
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
- `recipe_cls_name`: Nemo 2.0 recipe class name (e.g., "qwen3_8b", "llama3_1_70b")
- `peft_scheme`: PEFT method to use (default: "lora")
- `full_ft`: Enable full fine-tuning instead of PEFT (default: False)

#### Paths
- `data_dir`: Directory for processed datasets (default: auto-generated as `datasets/{dataset_name}/{dataset_config}/train={train_%}-val={val%}-test={test%}`)
- `output_dir`: Base directory for training outputs (default: `outputs/{hf_model_id}`)
- `nemo_ckpt_dir`: Directory for imported NeMo checkpoints (default: `{output_dir}/imported_hf_ckpt`)

**Note on data_dir**: When not explicitly set, the framework automatically generates a self-documenting directory path. For example, with the default Dolphin dataset (train_split_ratio=0.9, val_test_split_ratio=0.5), the path becomes: `datasets/cognitivecomputations_dolphin/flan1m-alpaca-uncensored/train=90%-val=5%-test=5%`

#### Distributed Training
- `num_nodes`: Number of compute nodes (default: 1)
- `gpus_per_node`: GPUs per node (default: 8)
- `node_rank`: Node rank for multi-node training (default: 0)
- `tensor_parallel_size`: Tensor parallelism size (default: 8)
- `pipeline_parallel_size`: Pipeline parallelism size (default: 1)
- `context_parallel_size`: Context parallelism size (default: 1)

#### Training Hyperparameters
- `max_steps`: Maximum training steps (default: 10000)
- `val_check_interval`: Validation frequency in steps (default: 800)
- `log_every_n_steps`: Logging frequency (default: 10)
- `micro_batch_size`: Batch size per GPU (default: 8)
- `accumulate_grad_batches`: Gradient accumulation steps (default: 8)
- `global_batch_size`: Effective batch size (computed: micro_batch_size × accumulate_grad_batches × data_parallel_size)
- `limit_val_batches`: Maximum validation batches (default: 80)
- `early_stopping_patience`: Early stopping patience (default: 3)
- `early_stopping_threshold`: Early stopping threshold (default: 0.001)

#### Sequence Configuration
- `max_seq_length`: Maximum sequence length (default: 2048)

#### Dataset Configuration
- `hf_dataset_config`: HFDatasetConfig instance with dataset loading and formatting settings

### Advanced Configuration Options

#### Early Stopping

Modify the `configure_callbacks()` function:

```python
early_stopping_callback = run.Config(
    EarlyStopping,
    monitor='val_loss',
    min_delta=0.001,      # Minimum change to qualify as improvement
    patience=5,           # Number of checks with no improvement
    mode='min',           # Minimize validation loss
)
```

#### LoRA Parameters

Customize LoRA settings in the recipe:

```python
nemo_recipe.peft.lora_tuning.target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
nemo_recipe.peft.lora_tuning.adapter.dim = 16
nemo_recipe.peft.lora_tuning.adapter.alpha = 32
nemo_recipe.peft.lora_tuning.adapter.dropout = 0.05
```

#### Parallelism Strategy

The framework configures parallelism via Config parameters:
- `tensor_parallel_size`: Tensor parallelism size (default: 8)
- `pipeline_parallel_size`: Pipeline parallelism size (default: 1)
- `context_parallel_size`: Context parallelism size (default: 1)

These are applied to the recipe:
```python
nemo_recipe.trainer.strategy.tensor_model_parallel_size = config.tensor_parallel_size
nemo_recipe.trainer.strategy.pipeline_model_parallel_size = config.pipeline_parallel_size
nemo_recipe.trainer.strategy.context_parallel_size = config.context_parallel_size
```

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
python peft_megatron.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --recipe_cls_name "qwen3_8b" \
  --num_nodes 1 \
  --gpus_per_node 8
```

### Full Fine-Tuning vs PEFT

By default, the framework uses PEFT (LoRA). To enable full fine-tuning of all model parameters:

```bash
python peft_megatron.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --recipe_cls_name "qwen3_8b" \
  --num_nodes 1 \
  --gpus_per_node 8 \
  --full_ft
```

**Note**: Full fine-tuning requires significantly more GPU memory and compute resources than PEFT. Adjust `micro_batch_size` and `accumulate_grad_batches` accordingly.

### Customizing HFDatasetConfig via CLI

Override HFDatasetConfig fields using the `--hfdc_<field_name>` pattern:

```bash
python peft_megatron.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --recipe_cls_name "qwen3_8b" \
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
python peft_megatron.py \
  --hf_model_id "meta-llama/Llama-2-7b-hf" \
  --recipe_cls_name "llama2_7b" \
  --data_dir "datasets/custom" \
  --output_dir "outputs/llama2_custom" \
  --num_nodes 2 \
  --gpus_per_node 8 \
  --max_steps 5000 \
  --micro_batch_size 4 \
  --accumulate_grad_batches 16 \
  --peft_scheme "lora" \
  --max_seq_length 4096 \
  --hfdc_dataset_name "timdettmers/openassistant-guanaco" \
  --hfdc_split "train" \
  --hfdc_train_split_ratio 0.95 \
  --hfdc_val_test_split_ratio 0.5 \
  --hfdc_input_template "### Human:\n{text}\n" \
  --hfdc_output_template "### Assistant:\n{text}" \
  --hfdc_num_proc 16
```

### CLI Notes

- The `field_mapping` argument expects a JSON string
- Use single quotes around JSON to avoid shell interpretation issues
- Template strings can include `\n` for newlines
- Not all HFDatasetConfig fields are exposed via CLI (e.g., `custom_converter`, `load_kwargs`)
- For complex configurations, modify the Config dataclass in `peft_megatron.py` directly

## Testing and Converting Checkpoints

### Testing a Checkpoint

After training, test your checkpoint with batch evaluation using NeMo's dynamic inference engine:

```bash
torchrun --standalone \
  --nproc-per-node=8 \
  test_checkpoint.py \
  --max_samples 1024 \
  --max_batch_size 16
```

The script automatically:
- Finds the latest `nemo_logs` directory under `outputs/`
- Discovers the latest checkpoint within that directory (format: `nemo_logs--val_loss={}-epoch={}-consumed_samples={}`)
- Discovers the latest `test.jsonl` file under `datasets/`
- Uses Megatron-Core's DynamicInferenceEngine for efficient batched generation
- Evaluates predictions using BERTScore

**Parameters:**
- `--nemo_logs_dir`: Directory containing NeMo logs (optional, auto-discovered from `outputs/`)
- `--test_path`: Path to test dataset JSONL file (optional, auto-discovered from `datasets/`)
- `--max_samples`: Maximum number of test samples to evaluate (default: 1024)
- `--max_batch_size`: Maximum concurrent requests for dynamic batching (default: 16)
- `--gpus_per_node`: Number of GPUs to use (default: 8)
- `--num_nodes`: Number of nodes (default: 1)
- `--tensor_parallel_size`: Tensor parallelism size (default: 8)
- `--pipeline_parallel_size`: Pipeline parallelism size (default: 1)
- `--context_parallel_size`: Context parallelism size (default: 1)
- `--temperature`: Sampling temperature (default: 0.1)
- `--top_k`: Top-k sampling (default: 0)
- `--top_p`: Nucleus sampling parameter (default: 0.95)
- `--num_tokens_to_generate`: Maximum tokens to generate (default: 512)
- `--inference_max_seq_length`: Maximum sequence length for inference (default: 8192)
- `--buffer_size_gb`: KV cache buffer size in GB (default: 20.0)
- `--block_size_tokens`: KV cache block size (default: 256)
- `--max_tokens`: Max tokens per batch (default: 65536)

**Output:**
- Predictions saved to: `{checkpoint_path}.jsonl`
- Evaluation metrics: BERTScore F1

### Converting to Hugging Face Format

Convert your NeMo checkpoint to standard Hugging Face format for deployment:

```bash
python convert_checkpoint_to_hf.py
```

The script automatically:
- Finds the latest `nemo_logs` directory under `outputs/`
- Discovers the latest checkpoint within that directory (format: `nemo_logs--val_loss={}-epoch={}-consumed_samples={}`)
- Merges LoRA weights with base model (if using LoRA)
- Exports to HuggingFace format
- Saves to `{checkpoint_path}.hf_model` (merged) or `{checkpoint_path}.hf_peft` (adapter only)

**Parameters:**
- `--nemo_logs_dir`: Directory containing NeMo logs (optional, auto-discovered from `outputs/`)
- `--target`: Target format (default: "hf")
- `--no_merge`: Save as LoRA adapter instead of merging (default: False)
- `--overwrite`: Whether to overwrite existing files (default: False)
- `--use_modelopt`: Enable ModelOpt for quantized models (default: False)

**LoRA Merging:**
- By default, LoRA weights are **merged** into the base model for maximum compatibility
- Merged models work with vLLM, TGI, and all Hugging Face tools
- Use `--no_merge` to save as a separate LoRA adapter (requires PEFT library to load)

**Export Process:**
1. If using LoRA: Merges adapter weights with base model (creates `.merged` checkpoint)
2. Exports merged/full checkpoint to HuggingFace format using `api.export_ckpt`
3. Saves model files including config.json, tokenizer files, and model weights

## Project Structure

```
.
├── peft_megatron.py                # Main training script
├── dataset_module.py               # Dataset processing module
├── test_checkpoint.py              # Test checkpoint with batch evaluation
├── convert_checkpoint_to_hf.py     # Convert NeMo checkpoint to HuggingFace
├── README.md                       # This file
├── datasets/                       # Downloaded and processed datasets
│   └── {dataset_name}/             # e.g., cognitivecomputations_dolphin
│       └── {dataset_config}/       # e.g., flan1m-alpaca-uncensored
│           └── train={train_%}-val={val%}-test={test%}/  # e.g., train=90%-val=5%-test=5%
│               ├── training.jsonl
│               ├── validation.jsonl
│               ├── test.jsonl
│               └── .data_ready
└── outputs/                        # Training outputs and logs
    └── {hf_model_id}/
        ├── imported_hf_ckpt/       # Imported HF checkpoint to NeMo format
        │   ├── context/
        │   └── weights/
        ├── nemo_logs/              # Training logs and checkpoints
        │   └── {timestamp}/        # e.g., 2024-01-15_10-30-00
        │       └── checkpoints/
        │           ├── nemo_logs--val_loss={}-epoch={}-consumed_samples={}/
        │           │   ├── context/
        │           │   └── weights/
        │           ├── nemo_logs--val_loss={}-epoch={}-consumed_samples={}.jsonl      # Test predictions
        │           ├── nemo_logs--val_loss={}-epoch={}-consumed_samples={}.merged/    # Merged checkpoint (LoRA)
        │           └── nemo_logs--val_loss={}-epoch={}-consumed_samples={}.hf_model/  # Exported HF model
        ├── tb_logs/                # TensorBoard logs
        ├── wandb_logs/             # Weights & Biases logs (if enabled)
        ├── mem_profile/            # Memory profiling data (if enabled)
        └── trace/                  # PyTorch profiler traces (if enabled)
```

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

The training script supports various monitoring and profiling callbacks to track performance, memory usage, and execution traces. Configure these options via command-line arguments:

| Configuration Option | Type | Default | Description | Output Location |
|---------------------|------|---------|-------------|----------------|
| `--enable_megatron_progress_bar` | bool | False | Displays Megatron-style progress bar with training metrics during execution | Console |
| `--enable_memory_monitor` | bool | False | Monitors GPU memory usage throughout training | Logs |
| `--enable_speed_monitor` | bool | False | Monitors training speed metrics (samples/sec, tokens/sec) | Logs |
| `--enable_runtime_estimator` | bool | False | Estimates remaining training time based on current progress | Logs |
| `--enable_memory_profile` | bool | False | Profiles detailed memory usage patterns | `{output_dir}/mem_profile/` |
| `--enable_pytorch_profiler` | bool | False | Enables PyTorch profiler for performance analysis | `{output_dir}/trace/` |
| `--enable_nsys_callback` | bool | False | Enables Nsys profiling for NVIDIA tools | System profiler |
| `--use_wandb` | bool | False | Enables Weights & Biases logging for experiment tracking | W&B dashboard |

**Note**: PyTorch profiler and Nsys callback cannot be enabled simultaneously.

### Notes

- **Performance Impact**: Profiling callbacks (PyTorch Profiler, Nsys, Memory Profile) add overhead and should only be enabled for short diagnostic runs
- **Storage**: Profiler traces can be large; ensure adequate disk space in the output directory
- **Profiler Analysis**: Use `tensorboard --logdir outputs/{model}/trace` to visualize PyTorch profiler results
- **Nsys Analysis**: Analyze Nsys profiles with NVIDIA Nsight Systems GUI
---