# Shared Utilities

Common utilities used across text and multi-modal training implementations.

## Files

### `callbacks.py`

Training callbacks for Hugging Face Trainer.

**Classes:**
- `SaveOnBestMetricCallback`: Saves checkpoints only when the evaluation metric improves

**Usage:**
```python
from shared.callbacks import SaveOnBestMetricCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[SaveOnBestMetricCallback()],
)
```

### `convert_checkpoint_to_hf.py`

Converts FSDP training checkpoints to standard HuggingFace format.

**Features:**
- Automatically finds the latest checkpoint
- Merges LoRA weights into base model (default)
- Can save as LoRA adapter (with `--no_merge`)
- Works with both text and multi-modal models

**Usage:**
```bash
# Convert latest checkpoint (merge LoRA)
python shared/convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B"

# Convert specific checkpoint
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --checkpoints_dir "results/Qwen/Qwen3-8B"

# Save as LoRA adapter (don't merge)
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --no_merge
```

**Arguments:**
- `--base_model`: Base model ID (required)
- `--checkpoints_dir`: Directory containing checkpoints (optional, auto-detected)
- `--no_merge`: Save as LoRA adapter instead of merging (optional)

### `test_checkpoint.py`

Tests converted checkpoints using vLLM for efficient inference.

**Features:**
- Automatically finds latest checkpoint and test dataset
- Uses vLLM for fast batched inference
- Evaluates predictions using BERTScore
- Supports both text and multi-modal models

**Usage:**
```bash
# Test latest checkpoint
python shared/test_checkpoint.py --base_model "Qwen/Qwen3-8B"

# Test with custom settings
python shared/test_checkpoint.py \
  --base_model "Qwen/Qwen3-8B" \
  --max_samples 1024 \
  --batch_size 128
```

**Arguments:**
- `--base_model`: Base model ID (required)
- `--max_samples`: Maximum number of test samples (default: 1024)
- `--batch_size`: Batch size for inference (default: 128)

## Design Principles

### Modality Agnostic

These utilities are designed to work with both text-only and multi-modal models:
- Automatic model type detection
- Flexible checkpoint loading
- Generic evaluation metrics

### Minimal Dependencies

Shared utilities have minimal dependencies to avoid conflicts:
- Core: `transformers`, `torch`, `accelerate`
- Optional: `vllm` (for testing), `bert_score` (for evaluation)

### Consistent Interface

All utilities follow consistent patterns:
- CLI-based with argparse
- Automatic path discovery
- Clear error messages
- Verbose logging

## Adding New Utilities

When adding new shared utilities:

1. **Keep it generic**: Should work for both text and multi-modal
2. **Document thoroughly**: Add docstrings and usage examples
3. **Handle errors gracefully**: Provide clear error messages
4. **Test across modalities**: Verify with both text and multi-modal models
5. **Update this README**: Document the new utility

## Example: Using Shared Utilities in Training Scripts

```python
# In text/peft_accelerate.py or multimodal/peft_accelerate.py
import sys
from pathlib import Path

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from callbacks import SaveOnBestMetricCallback

# Use in training
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[SaveOnBestMetricCallback()],
)
```

## Future Utilities

Planned additions:
- `metrics.py`: Common evaluation metrics
- `utils.py`: General helper functions
- `checkpoint_manager.py`: Advanced checkpoint management
- `distributed_utils.py`: Distributed training helpers
