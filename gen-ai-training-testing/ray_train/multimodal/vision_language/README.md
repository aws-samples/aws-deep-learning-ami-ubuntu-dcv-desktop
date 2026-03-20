# Vision-Language Model Training with Ray Train

> **Note**: For installation instructions, see the [parent README](../../README.md).

Train and continually pre-train Qwen3-VL vision-language models using Ray Train's `TorchTrainer` with FSDP and the adapter pattern for extensibility.

## Supported Models

**Qwen3-VL (Latest):**
- **Qwen/Qwen3-VL-8B-Instruct** ⭐ Recommended

## Quick Start

After building and running the Docker container (see [parent README](../../README.md)), navigate to the vision_language directory:

```bash
cd /app/ray_train/multimodal/vision_language
```

### List Supported Models

```bash
python ray_train_vlm.py --list_models
```

### Train with HuggingFace Dataset (Recommended)

The simplest way to get started - automatically downloads and prepares the dataset:

```bash
python ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data"
```

The `data_dir` and `results_dir` are automatically derived:
- `data_dir`: `datasets/lmms-lab_LLaVA-NeXT-Data/default/train=90%-val=5%-test=5%`
- `results_dir`: `results/`

### Train with Custom Dataset

If you have pre-prepared JSONL files:

```bash
python ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --data_dir "datasets/my_custom_dataset"
```

### Continual Pre-Training (Image+Text)

```bash
python cpt_ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --num_train_epochs 3 \
  --save_steps 1000
```

### Test Adapters

```bash
python test_adapters.py
```

## Using HuggingFace Datasets

The training script can automatically download and prepare HuggingFace vision-language datasets.

### Supported Datasets

- **lmms-lab/LLaVA-NeXT-Data** (recommended) - 779K high-quality vision-language instruction samples
- **HuggingFaceM4/VQAv2** - Visual question answering
- Custom datasets in JSONL format

### Basic Usage

```bash
python ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data"
```

Images are automatically saved to disk during dataset preparation.

### Advanced Options

```bash
python ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --hf_train_split_ratio 0.95 \
  --hf_val_test_split_ratio 0.5
```

### Custom Data Directory

Override the auto-generated data directory:

```bash
python ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --data_dir "datasets/my_llava_data"
```

## Dataset Format

The dataset format is model-agnostic. The same dataset works for all vision-language models because adapters handle model-specific formatting automatically.

### Directory Structure

```
datasets/your_dataset/
├── training.jsonl       # Training data
├── validation.jsonl     # Validation data
└── test.jsonl          # Test data (optional)
```

### JSONL Format

Each line is a JSON object representing one sample.

**Single-Turn Example:**

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is in this image?"},
    {"from": "gpt", "value": "This image shows a cat sitting on a couch."}
  ]
}
```

**Multi-Turn Example:**

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat animal is this?"},
    {"from": "gpt", "value": "This is a cat."},
    {"from": "human", "value": "What color is it?"},
    {"from": "gpt", "value": "The cat is orange and white."}
  ]
}
```

### Field Specifications

**`image` (required)** - Supports three formats:

1. Local path: `"images/cat.jpg"`
2. URL: `"https://example.com/images/cat.jpg"`
3. Base64: `"data:image/jpeg;base64,/9j/4AAQSkZJRg..."`

**`conversations` (required)** - Array of conversation turns with:
- `from`: `"human"` or `"gpt"`
- `value`: Text content with `<image>` token

### Image Token Placement

Place `<image>` in the first human turn:

```json
{"from": "human", "value": "<image>\nWhat is this?"}
```

### How Adapters Work

The adapter converts the universal format to Qwen-VL's specific format automatically:

**Universal Format:**
```json
{
  "image": "cat.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat animal is this?"},
    {"from": "gpt", "value": "This is a cat."}
  ]
}
```

**Qwen-VL Output (automatic):**
```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What animal is this?<|im_end|>
<|im_start|>assistant
This is a cat.<|im_end|>
```

### Creating Custom Datasets

**Step 1: Prepare Images**

```
datasets/my_dataset/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
```

**Step 2: Create JSONL Files**

Create `training.jsonl`:
```json
{"image": "images/img_001.jpg", "conversations": [{"from": "human", "value": "<image>\nWhat is this?"}, {"from": "gpt", "value": "A cat."}]}
{"image": "images/img_002.jpg", "conversations": [{"from": "human", "value": "<image>\nDescribe this."}, {"from": "gpt", "value": "A dog playing."}]}
```

Create `validation.jsonl`:
```json
{"image": "images/img_003.jpg", "conversations": [{"from": "human", "value": "<image>\nWhat do you see?"}, {"from": "gpt", "value": "A bird."}]}
```

**Step 3: Train**

```bash
python ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --data_dir "datasets/my_dataset"
```

### Best Practices

1. Use relative paths from dataset directory
2. Start with human, alternate speakers
3. Place `<image>` in first human turn
4. Be specific and descriptive in responses
5. Recommended sizes:
   - Training: 10K - 1M samples
   - Validation: 500 - 10K samples (1-10% of training)

## Continual Pre-Training (CPT)

VLM CPT extends a vision-language model's knowledge by training on domain-specific image+text data using the full causal LM objective (all tokens are training targets, no label masking). This is useful for adapting a VLM to a new visual domain (e.g., medical imaging, satellite imagery, technical diagrams) before doing SFT.

For text-only CPT on a VLM backbone (no images), use `text/cpt_accelerate.py` in the accelerate framework instead.

### CPT Quick Start

```bash
python cpt_ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --num_train_epochs 3 \
  --save_steps 1000
```

### CPT with Custom Domain Data

```bash
python cpt_ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --data_dir "datasets/medical_images" \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --save_steps 500
```

### Resuming from Checkpoint

```bash
python cpt_ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --data_dir "datasets/medical_images" \
  --resume_from_checkpoint "results/checkpoint-2000"
```

### CPT vs SFT for VLMs

| Aspect | CPT (`cpt_ray_train_vlm.py`) | SFT (`ray_train_vlm.py`) |
|--------|------------------------------|--------------------------|
| Objective | Causal LM on all tokens | Causal LM on assistant tokens only |
| Label masking | None | Human/instruction tokens masked |
| Fine-tuning | Full model weights | LoRA or full |
| Vision encoder | Typically unfrozen | Typically frozen |
| Training schedule | Epoch-based | Step-based |
| Checkpointing | Regular interval (`save_steps`) | Best metric only |
| Typical data | Domain image+text corpora | Instruction/response pairs |
| Dataset class | `VLMCPTDataset` | `VLMDataset` |

### CPT Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | Qwen/Qwen3-VL-8B-Instruct | Model to train |
| `--num_train_epochs` | 3 | Number of training epochs |
| `--freeze_vision_encoder` | False | Freeze vision encoder |
| `--learning_rate` | 2e-5 | Learning rate |
| `--warmup_steps` | 1000 | Warmup steps (longer for CPT) |
| `--save_steps` | 1000 | Checkpoint save frequency |
| `--save_total_limit` | 2 | Max checkpoints to keep |
| `--resume_from_checkpoint` | None | Path to resume from |
| `--hf_dataset_name` | None | HuggingFace dataset name |
| `--data_dir` | Auto-derived | Dataset directory |
| `--results_dir` | results | Results directory |

## SFT Training Configuration

### Basic Training

```bash
python ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data"
```

### SFT Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | Qwen/Qwen3-VL-8B-Instruct | Model to train |
| `--hf_dataset_name` | None | HuggingFace dataset name |
| `--data_dir` | Auto-derived | Dataset directory |
| `--results_dir` | results | Results directory |
| `--freeze_vision_encoder` | True | Freeze vision encoder (saves memory) |
| `--lora_rank` | 64 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--max_steps` | 10000 | Training steps |
| `--per_device_train_batch_size` | 1 | Batch size per GPU |
| `--gradient_accumulation_steps` | 16 | Gradient accumulation |
| `--learning_rate` | 2e-5 | Learning rate |
| `--early_stopping_patience` | 3 | Early stopping patience |

### Advanced Options

```bash
# With LoRA on vision encoder
--lora_on_vision

# Full fine-tuning (not recommended)
--full_ft

# Custom LoRA configuration
--lora_rank 128 --lora_alpha 32

# Adjust for memory
--per_device_train_batch_size 1 --gradient_accumulation_steps 16
```

## GPU Requirements

### Qwen3-VL-8B (Recommended)
- **GPUs**: 8x A100 80GB
- **Batch size**: 1 per device (required for dynamic resolution)
- **Gradient accumulation**: 16 steps (effective batch size = 128)
- **Memory**: ~50GB per GPU with frozen vision encoder

## Key Features

### Dynamic Resolution
Qwen-VL supports any image resolution and aspect ratio (no padding/cropping required). Images that produce sequences exceeding `max_seq_length` are progressively downscaled (up to 3 attempts) to fit.

### Adapter Pattern
Extensible architecture - easy to add other model families later:
- Base adapter interface defines the contract
- Qwen adapter implements Qwen-specific logic
- Registry auto-detects model from ID

### LoRA Support
Parameter-efficient fine-tuning:
- Language model LoRA (default)
- Optional vision encoder LoRA
- Significant memory savings

### Automatic Dataset Preparation
- Downloads HuggingFace datasets automatically
- Converts to universal JSONL format
- Handles images (PIL, URLs, local paths)
- Pre-built converters for common datasets

### File-Based Rank Synchronization
Dataset preparation runs on rank 0 only. Other ranks poll for a `.data_ready` marker file every 30 seconds, avoiding NCCL timeout issues with large datasets (700K+ samples).

### Ray Train Benefits
- Automatic fault tolerance with configurable max failures
- Built-in checkpoint management via `CheckpointConfig`
- Automatic GPU discovery and scaling
- SPREAD placement strategy for optimal GPU utilization

## Architecture

### Directory Structure

```
vision_language/
├── base/
│   ├── base_adapter.py          # Abstract adapter interface
│   └── base_dataset.py          # VLMDataset (SFT) and VLMCPTDataset (CPT)
│
├── adapters/
│   ├── qwen_vl_adapter.py       # Qwen-VL implementation
│   └── registry.py              # Adapter registry & auto-detection
│
├── dataset_module.py             # VLM dataset preparation & converters
├── ray_train_vlm.py              # VLM SFT with Ray Train
├── cpt_ray_train_vlm.py          # VLM CPT with Ray Train
└── test_adapters.py              # Adapter tests
```

### Core Components

**1. Base Adapter Interface** - Defines the contract that all model-specific adapters must implement (model loading, image processing, conversation formatting, LoRA targets).

**2. Qwen-VL Adapter** - Implements the interface for Qwen-VL models with dynamic resolution, structured message format, and model-specific special tokens.

**3. Adapter Registry** - Automatically detects and returns the appropriate adapter based on model ID.

**4. Model-Agnostic Training Script** - Single training script works for all models via the adapter pattern.

### Adding New Model Families

To add support for a new model family:

1. Create new adapter file in `adapters/`
2. Inherit from `BaseVLMAdapter`
3. Implement all abstract methods
4. Register in `adapters/registry.py`
5. Training script works automatically

Example:
```python
# adapters/new_model_adapter.py
class NewModelAdapter(BaseVLMAdapter):
    @property
    def model_family(self) -> str:
        return "new-model"
    
    # Implement other methods...

# adapters/registry.py
ADAPTER_REGISTRY = {
    "qwen-vl": QwenVLAdapter,
    "new-model": NewModelAdapter,  # Add this line
}
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 16

# Ensure vision encoder is frozen
--freeze_vision_encoder
```

### Slow Training
- Check dataset loading (images should be accessible)
- Reduce `--max_seq_length` if sequences are very long

### Image Loading Errors
- Verify image paths are relative to `data_dir`
- Check images are accessible (local paths or URLs)
- Ensure sufficient disk space

### Image Token/Feature Mismatch
If you see `ValueError: Image features and image tokens do not match`, the image produces too many vision tokens. The `_fit_to_seq_length` method handles this automatically by progressively downscaling images. Increase `--max_seq_length` if needed.

## Resources

- [Qwen-VL GitHub](https://github.com/QwenLM/Qwen-VL)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [HuggingFace Models](https://huggingface.co/Qwen)
