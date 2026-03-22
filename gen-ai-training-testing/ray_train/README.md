# Ray Train: Distributed Training Framework

This directory provides flexible frameworks for fine-tuning models using Ray Train's `TorchTrainer` with FSDP (Fully Sharded Data Parallel), automatic fault tolerance, and checkpoint management.

## Installation

### Using Docker

Build the Docker image:
```bash
cd ~/aws-deep-learning-ami-ubuntu-dcv-desktop/gen-ai-training-testing/ray_train
docker buildx build -t ray_train:latest -f ../containers/Dockerfile.ray_train .
```

Run the container with GPU support:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  --shm-size=32g \
  ray_train:latest
```

Inside the container, navigate to the ray_train directory:
```bash
cd /app
```

## Directory Structure

```
ray_train/
├── text/                              # Text-only LLM training
│   ├── ray_train_sft.py              # SFT with LoRA/full fine-tuning
│   ├── dataset_module.py             # Text dataset processing
│   └── README.md                     # Detailed text training documentation
│
├── multimodal/                        # Multi-modal model training
│   ├── vision_language/              # Text + Image (VLMs)
│   ├── shared/                       # Multi-modal utilities
│   └── README.md                     # Multi-modal overview
│
├── shared/                            # Common utilities
│   ├── convert_checkpoint_to_hf.py   # Ray Train → HuggingFace conversion
│   └── test_checkpoint.py            # vLLM inference + BERTScore evaluation
│
└── README.md                          # This file
```

## Quick Start

After building and running the Docker container (see Installation above):

### Text-Only Training

```bash
cd /app

python text/ray_train_sft.py \
  --hf_model_id "Qwen/Qwen3-8B" \
  --hfdc_dataset_name "cognitivecomputations/dolphin" \
  --hfdc_dataset_config "flan1m-alpaca-uncensored" \
  --max_steps 5000
```

See [text/README.md](./text/README.md) for comprehensive documentation.

### Vision-Language Training

```bash
cd /app

python multimodal/vision_language/ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --freeze_vision_encoder \
  --lora_rank 64
```

See [multimodal/vision_language/README.md](./multimodal/vision_language/README.md) for comprehensive documentation.

## Features

### Text Training
- Supervised Fine-Tuning (SFT) with LoRA or full fine-tuning
- Generalized HuggingFace Dataset Support
- Distributed Training with FSDP via Ray Train
- Flash Attention 2
- Early Stopping and Best Model Saving
- Automatic fault tolerance and restart

### Multi-Modal Training
- Vision-language model SFT (Qwen3-VL)
- Vision-language model CPT (continual pre-training)
- Image + text dataset processing
- Multi-modal LoRA support
- Vision encoder freezing and fine-tuning options
- Adapter pattern for extensibility

## Common Utilities

### Checkpoint Conversion

Convert Ray Train FSDP checkpoints to HuggingFace format:

```bash
python shared/convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B"
```

### Checkpoint Testing

Test checkpoints with vLLM:

```bash
python shared/test_checkpoint.py --base_model "Qwen/Qwen3-8B"
```

## Ray Train vs Accelerate

| Aspect | Ray Train | Accelerate |
|--------|-----------|------------|
| Orchestration | Ray `TorchTrainer` | HF Accelerate launcher |
| Fault tolerance | Built-in (auto-restart) | Manual |
| Checkpoint management | Ray checkpoint system | HF Trainer checkpoints |
| Scaling | Ray `ScalingConfig` | `accelerate_config.yaml` |
| GPU discovery | Automatic via Ray | Manual process count |
| Multi-node | Ray cluster | SSH + accelerate config |

## GPU Requirements

### Small Models (1B - 13B parameters)
- GPUs: 8x A100 (40GB or 80GB)
- Batch size: 2-4 per device
- Gradient accumulation: 4-8

### Medium Models (13B - 34B parameters)
- GPUs: 16x A100 (80GB) total (2 nodes)
- Batch size: 1-2 per device
- Gradient accumulation: 8-16

### Large Models (34B - 100B parameters)
- GPUs: 32-64x A100 (80GB) or H100 (80GB)
- Batch size: 1 per device
- Gradient accumulation: 16-32

## Documentation

- [Text Training Documentation](./text/README.md) - Comprehensive guide for text-only LLM training
- [Multi-Modal Training Documentation](./multimodal/README.md) - Guide for vision-language model training
- [Shared Utilities](./shared/README.md) - Common tools for checkpoint management and testing

## Support

For detailed information on specific training types:
- Text-only training: See [text/README.md](./text/README.md)
- Multi-modal training: See [multimodal/README.md](./multimodal/README.md)

For general questions about the framework, refer to the parent [README.md](../README.md).
