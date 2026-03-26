# Hugging Face Trainer with Accelerate

This directory provides flexible frameworks for fine-tuning models using Hugging Face Trainer, Accelerate, and FSDP (Fully Sharded Data Parallel).

## Installation

### Using Docker

Build the Docker image:
```bash
cd ~/aws-deep-learning-ami-ubuntu-dcv-desktop/gen-ai-training-testing/accelerate
docker buildx build -t accelerate:latest -f ../containers/Dockerfile.accelerate .
```

Run the container with GPU support:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  --shm-size=32g \
  accelerate:latest
```

Inside the container, navigate to the accelerate directory:
```bash
cd /app
```

## Directory Structure

```
accelerate/
├── text/                          # Text-only LLM training
│   ├── peft_accelerate.py         # SFT training
│   ├── cpt_accelerate.py          # Continual pre-training
│   ├── dpo_accelerate.py          # DPO alignment
│   ├── ppo_accelerate.py          # PPO-RLHF alignment
│   ├── reward_model_accelerate.py # Reward model training
│   ├── dataset_module.py          # Text dataset processing
│   ├── rm_dataset_module.py       # Reward model dataset processing
│   ├── run_dpo_pipeline.sh        # DPO pipeline script
│   ├── run_ppo_pipeline.sh        # PPO-RLHF pipeline script
│   └── README.md                  # Detailed text training documentation
│
├── multimodal/                    # Multi-modal model training
│   ├── vision_language/           # Text + Image (VLMs)
│   ├── shared/                    # Multi-modal utilities
│   └── README.md                  # Multi-modal overview
│
├── shared/                        # Common utilities
│   ├── callbacks.py               # Training callbacks
│   ├── convert_checkpoint_to_hf.py # Checkpoint conversion
│   └── test_checkpoint.py         # Checkpoint testing
│
└── peft_accelerate_config.yaml   # FSDP configuration for PEFT/SFT
└── cpt_accelerate_config.yaml    # FSDP configuration for CPT
```

## Quick Start

After building and running the Docker container (see Installation above):

### Text-Only Training

```bash
cd /app

# Run DPO pipeline (recommended)
bash text/run_dpo_pipeline.sh

# Or run SFT only
accelerate launch --config_file peft_accelerate_config.yaml text/peft_accelerate.py
```

See [text/README.md](./text/README.md) for comprehensive documentation.

### Vision-Language Training

```bash
cd /app

# Train Qwen3-VL-8B with HuggingFace dataset (auto data_dir)
accelerate launch --config_file peft_accelerate_config.yaml multimodal/vision_language/peft_accelerate.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "MMInstruction/M3IT"
```

See [multimodal/vision_language/README.md](./multimodal/vision_language/README.md) for comprehensive documentation.

## Features

### Text Training
- Continual Pre-Training: Extend model knowledge with domain-specific corpora
- Complete RLHF Pipeline: SFT → Reward Model → PPO policy optimization
- DPO Pipeline: SFT → DPO preference optimization (simpler, no reward model needed)
- Generalized HuggingFace Dataset Support
- Distributed Training with FSDP
- LoRA and Full Fine-Tuning
- Flash Attention 2
- Early Stopping and Best Model Saving

### Multi-Modal Training
- Vision-language model fine-tuning (Qwen3-VL)
- Image + text dataset processing
- Multi-modal LoRA support
- Vision encoder freezing and fine-tuning options

## Common Utilities

### Checkpoint Conversion

Convert FSDP checkpoints to HuggingFace format:

```bash
python shared/convert_checkpoint_to_hf.py --base_model "Qwen/Qwen3-8B"
```

### Checkpoint Testing

Test checkpoints with vLLM:

```bash
python shared/test_checkpoint.py --base_model "Qwen/Qwen3-8B"
```

## Configuration

### Accelerate Configuration

The `peft_accelerate_config.yaml` and `cpt_accelerate_config.yaml` files configure distributed training:

- FSDP Strategy: FULL_SHARD for maximum memory efficiency
- Mixed Precision: BFloat16
- Backward Prefetch: BACKWARD_PRE
- Number of Processes: Set to match your GPU count

### Multi-Node Training

Update `peft_accelerate_config.yaml` or `cpt_accelerate_config.yaml` for multi-node:

```yaml
num_machines: 2
num_processes: 16  # 8 GPUs per node × 2 nodes
machine_rank: 0    # Set per node
main_process_ip: <main_node_ip>
main_process_port: 29500
```

## Documentation

- [Text Training Documentation](./text/README.md) - Comprehensive guide for text-only LLM training
- [Multi-Modal Training Documentation](./multimodal/README.md) - Guide for vision-language model training
- [Shared Utilities](./shared/) - Common tools for checkpoint management and testing

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

## Support

For detailed information on specific training types:
- Text-only training: See [text/README.md](./text/README.md)
- Multi-modal training: See [multimodal/README.md](./multimodal/README.md)

For general questions about the framework, refer to the parent [README.md](../README.md).
