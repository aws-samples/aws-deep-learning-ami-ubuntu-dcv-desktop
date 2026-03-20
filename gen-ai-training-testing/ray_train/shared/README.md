# Shared Utilities

Common utilities used across text and multi-modal training implementations with Ray Train.

## Files

### `convert_checkpoint_to_hf.py`

Converts Ray Train FSDP checkpoints to standard HuggingFace format.

**Features:**
- Automatically finds the latest checkpoint (including nested `TorchTrainer_*` directories)
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
  --checkpoints_dir "results/Qwen-Qwen3-8B"

# Save as LoRA adapter (don't merge)
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --no_merge

# Full fine-tuning checkpoint (no LoRA)
python shared/convert_checkpoint_to_hf.py \
  --base_model "Qwen/Qwen3-8B" \
  --full_ft
```

**Arguments:**

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model ID | `Qwen/Qwen3-8B` |
| `--checkpoints_dir` | str | Directory containing checkpoints | Auto-detected |
| `--full_ft` | flag | Full fine-tuning mode (no LoRA) | `False` |
| `--no_merge` | flag | Save as LoRA adapter instead of merging | `False` |
| `--lora_rank` | int | LoRA rank (for adapter info) | `32` |
| `--lora_alpha` | int | LoRA alpha | `32` |

### `test_checkpoint.py`

Tests converted checkpoints using vLLM for efficient inference and evaluates with BERTScore.

**Features:**
- Automatically finds latest checkpoint and test dataset
- Uses vLLM for fast batched inference
- Evaluates predictions using BERTScore
- Supports both LoRA and full fine-tuning checkpoints
- Handles Ray Train's nested checkpoint directory structure

**Usage:**
```bash
# Test latest checkpoint
python shared/test_checkpoint.py --base_model "Qwen/Qwen3-8B"

# Test with custom settings
python shared/test_checkpoint.py \
  --base_model "Qwen/Qwen3-8B" \
  --max_samples 1024 \
  --batch_size 128

# Test full fine-tuning checkpoint
python shared/test_checkpoint.py \
  --base_model "Qwen/Qwen3-8B" \
  --full_ft
```

**Arguments:**

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model ID | `Qwen/Qwen3-8B` |
| `--checkpoints_dir` | str | Directory containing checkpoints | Auto-detected |
| `--full_ft` | flag | Full fine-tuning mode | `False` |
| `--test_path` | str | Path to test.jsonl | Auto-detected |
| `--max_samples` | int | Maximum test samples | `1024` |
| `--batch_size` | int | Inference batch size | `128` |
| `--temperature` | float | Generation temperature | `0.1` |
| `--max_tokens` | int | Max tokens to generate | `512` |
| `--tensor_parallel_size` | int | vLLM tensor parallel size | `8` |
| `--gpu_memory_utilization` | float | vLLM GPU memory utilization | `0.9` |
| `--max_model_len` | int | vLLM max model length | `8192` |

## Design Principles

### Modality Agnostic

These utilities are designed to work with both text-only and multi-modal models:
- Automatic model type detection
- Flexible checkpoint loading
- Generic evaluation metrics

### Ray Train Aware

Handles Ray Train's specific checkpoint structure:
- Nested `TorchTrainer_*/checkpoint_*/checkpoint/` directories
- Automatic discovery of latest checkpoint by modification time
- Support for Ray's checkpoint management

### Minimal Dependencies

Shared utilities have minimal dependencies to avoid conflicts:
- Core: `transformers`, `torch`, `ray`
- Optional: `vllm` (for testing), `evaluate` (for BERTScore)

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
# After training, convert and test the checkpoint
import subprocess

# Convert checkpoint
subprocess.run([
    "python", "../shared/convert_checkpoint_to_hf.py",
    "--base_model", "Qwen/Qwen3-8B"
])

# Test checkpoint
subprocess.run([
    "python", "../shared/test_checkpoint.py",
    "--base_model", "Qwen/Qwen3-8B",
    "--max_samples", "512"
])
```

## Future Utilities

Planned additions:
- `metrics.py`: Common evaluation metrics
- `utils.py`: General helper functions
- `checkpoint_manager.py`: Advanced checkpoint management
- `distributed_utils.py`: Distributed training helpers
