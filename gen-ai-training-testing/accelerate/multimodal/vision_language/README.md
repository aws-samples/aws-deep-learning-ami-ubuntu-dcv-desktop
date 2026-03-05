# Vision-Language Model Training

> **Note**: For installation instructions, see the [parent README](../../README.md).

Train Qwen3-VL vision-language models using Hugging Face Trainer and Accelerate with the adapter pattern for extensibility.

## Supported Models

**Qwen3-VL (Latest):**
- **Qwen/Qwen3-VL-8B-Instruct** ⭐ Recommended

## Quick Start

After building and running the Docker container (see [parent README](../../README.md)), navigate to the vision_language directory:

```bash
cd /app/accelerate/multimodal/vision_language
```

### List Supported Models

```bash
python peft_accelerate.py --list_models
```

### Train with HuggingFace Dataset (Recommended)

The simplest way to get started - automatically downloads and prepares the dataset:

```bash
accelerate launch --config_file ../../accelerate_config.yaml peft_accelerate.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data"
```

The `data_dir` and `output_dir` are automatically derived:
- `data_dir`: `datasets/lmms-lab_LLaVA-NeXT-Data/default/train=90%-val=5%-test=5%`
- `output_dir`: `results/Qwen/Qwen3-VL-8B-Instruct`

### Train with Custom Dataset

If you have pre-prepared JSONL files:

```bash
accelerate launch --config_file ../../accelerate_config.yaml peft_accelerate.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --data_dir "datasets/my_custom_dataset"
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
accelerate launch --config_file ../../accelerate_config.yaml peft_accelerate.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data"
```

Images are automatically saved to disk during dataset preparation.

### Advanced Options

```bash
accelerate launch --config_file ../../accelerate_config.yaml peft_accelerate.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --hf_train_split_ratio 0.95 \
  --hf_val_test_split_ratio 0.5
```

Note: Images are automatically saved to disk by default. The dataset contains PIL Image objects that are converted to JPEG files during preparation.

### Custom Data Directory

Override the auto-generated data directory:

```bash
accelerate launch --config_file ../../accelerate_config.yaml peft_accelerate.py \
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
accelerate launch --config_file ../../accelerate_config.yaml peft_accelerate.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --data_dir "datasets/my_dataset" \
  --output_dir "results/my_model"
```

### Best Practices

1. Use relative paths from dataset directory
2. Start with human, alternate speakers
3. Place `<image>` in first human turn
4. Be specific and descriptive in responses
5. Recommended sizes:
   - Training: 10K - 1M samples
   - Validation: 500 - 10K samples (1-10% of training)

### Dataset Troubleshooting

**"Image not found"** - Use relative paths from dataset directory:
```json
"image": "images/cat.jpg"  // Relative to data_dir
```

**"Invalid conversation format"** - Ensure each sample has required fields:
```json
{
  "image": "...",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

## Training Configuration

### Basic Training

```bash
accelerate launch --config_file ../../accelerate_config.yaml peft_accelerate.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data"
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | Qwen/Qwen3-VL-8B-Instruct | Model to train |
| `--hf_dataset_name` | None | HuggingFace dataset name |
| `--data_dir` | Auto-derived | Dataset directory |
| `--output_dir` | Auto-derived | Output directory |
| `--freeze_vision_encoder` | True | Freeze vision encoder (saves memory) |
| `--lora_rank` | 64 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--max_steps` | 10000 | Training steps |
| `--per_device_train_batch_size` | 2 | Batch size per GPU |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation |
| `--learning_rate` | 2e-5 | Learning rate |

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
Qwen-VL supports any image resolution and aspect ratio (no padding/cropping required).

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

## Architecture

The adapter pattern allows supporting multiple VLM families with a single training codebase by abstracting model-specific differences.

### Directory Structure

```
vision_language/
├── base/
│   ├── base_adapter.py      # Abstract adapter interface
│   └── base_dataset.py      # Dataset using adapters
│
├── adapters/
│   ├── qwen_vl_adapter.py   # Qwen-VL implementation
│   └── registry.py          # Adapter registry & auto-detection
│
└── peft_accelerate.py       # Model-agnostic training script
```

### Core Components

**1. Base Adapter Interface**

Defines the contract that all model-specific adapters must implement:

```python
class BaseVLMAdapter(ABC):
    """Abstract base class for vision-language model adapters."""
    
    @abstractmethod
    def model_family(self) -> str:
        """Return model family name (e.g., 'qwen-vl')."""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str, **kwargs):
        """Load the vision-language model."""
        pass
    
    @abstractmethod
    def load_processor(self, model_id: str, **kwargs):
        """Load the processor (handles both image and text)."""
        pass
    
    @abstractmethod
    def format_conversation(self, conversations, processor, tokenizer) -> str:
        """Format conversation into model-specific format."""
        pass
    
    @abstractmethod
    def get_lora_target_modules(self, include_vision: bool = False) -> List[str]:
        """Get LoRA target modules for this model."""
        pass
```

**2. Qwen-VL Adapter**

Implements the interface for Qwen-VL models:

```python
class QwenVLAdapter(BaseVLMAdapter):
    """Adapter for Qwen-VL model family."""
    
    @property
    def model_family(self) -> str:
        return "qwen-vl"
    
    @property
    def supported_models(self) -> List[str]:
        return [
            "Qwen/Qwen3-VL-8B-Instruct",
        ]
    
    def load_model(self, model_id: str, **kwargs):
        """Load Qwen-VL model."""
        return AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            **kwargs
        )
    
    def format_conversation(self, conversations, processor, tokenizer) -> str:
        """Format conversation for Qwen-VL."""
        # Converts universal format to Qwen-VL's structured message format
        # Handles <image> tokens and special formatting
        ...
```

**3. Adapter Registry**

Automatically detects and returns the appropriate adapter:

```python
ADAPTER_REGISTRY = {
    "qwen-vl": QwenVLAdapter,
}

def get_adapter_for_model(model_id: str) -> BaseVLMAdapter:
    """Automatically detect and return the appropriate adapter."""
    for adapter_class in ADAPTER_REGISTRY.values():
        adapter = adapter_class()
        if adapter.validate_model_id(model_id):
            return adapter
    raise ValueError(f"No adapter found for model: {model_id}")
```

**4. Model-Agnostic Training Script**

Single training script works for all models:

```python
def train(args):
    # Get appropriate adapter automatically
    adapter = get_adapter_for_model(args.model_id)
    
    # Load model using adapter
    model = adapter.load_model(args.model_id)
    processor = adapter.load_processor(args.model_id)
    
    # Apply LoRA using adapter-specific targets
    lora_target_modules = adapter.get_lora_target_modules()
    
    # Create dataset using adapter
    train_dataset = VLMDataset(
        adapter=adapter,
        processor=processor,
        ...
    )
    
    # Train (same for all models)
    trainer.train()
```

### Architecture Benefits

1. Single training script for all models
2. Easy to add new model families
3. Model-specific optimizations handled automatically
4. Testable and maintainable
5. No code duplication

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

## Resources

- [Qwen-VL GitHub](https://github.com/QwenLM/Qwen-VL)
- [Qwen-VL Paper](https://arxiv.org/abs/2308.12966)
- [HuggingFace Models](https://huggingface.co/Qwen)
