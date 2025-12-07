# Gen AI Training Testing

Comprehensive fine-tuning frameworks for Large Language Models using multiple distributed training approaches on [AWS Deep Learning Desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop).

## Overview

This directory provides four frameworks for fine-tuning LLMs with Parameter-Efficient Fine-Tuning (PEFT) or full fine-tuning. Each framework offers unique advantages for different use cases and infrastructure setups.

**Frameworks:**
* [NeMo 2.0](./nemo2/) - NVIDIA's enterprise-grade framework with Megatron-LM
* [PyTorch Lightning (PTL)](./ptl/) - Flexible framework with FSDP and custom training loops
* [Hugging Face Accelerate](./accelerate/) - Simplified distributed training with Trainer API
* [Ray Train](./ray_train/) - Scalable training with Ray's distributed orchestration

**Common Features:**
* Parameter-Efficient Fine-Tuning (LoRA) and full fine-tuning support
* Generalized HuggingFace dataset pipeline with flexible templates
* Multi-node, multi-GPU distributed training
* Automatic checkpoint conversion to HuggingFace format
* Comprehensive testing and evaluation scripts
* Docker containerization for reproducibility

**Supported Hardware:**
* NVIDIA GPUs (A100, H100, etc.)
* AWS Trainium/Inferentia (NeMo 2.0 only)

## Quick Comparison

| Framework | Best For | Key Advantage | Complexity |
|-----------|----------|---------------|------------|
| **NeMo 2.0** | Very large models (70B+), production deployments | Tensor/pipeline parallelism, Megatron optimizations | High |
| **PyTorch Lightning** | Custom training logic, research | Full control, flexible callbacks | Medium |
| **Accelerate** | Quick experiments, standard workflows | Simple API, minimal code | Low |
| **Ray Train** | Multi-cluster, fault tolerance | Distributed orchestration, auto-recovery | Medium |

## Framework Details

### [NeMo 2.0](./nemo2/)

NVIDIA's enterprise framework built on Megatron-LM for training very large models.

**Key Features:**
* Tensor and pipeline parallelism for 100B+ parameter models
* Megatron-LM optimizations for maximum performance
* Native support for AWS Trainium/Inferentia
* Advanced memory optimizations (activation checkpointing, CPU offload)
* Comprehensive monitoring and profiling tools

**Best For:**
* Models 70B+ parameters
* Production deployments requiring maximum efficiency
* Multi-node training with complex parallelism strategies

**Documentation:** [nemo2/README.md](./nemo2/README.md)

---

### [PyTorch Lightning (PTL)](./ptl/)

Flexible framework with FSDP for custom training workflows and research.

**Key Features:**
* Full control over training loop with Lightning modules
* FSDP (Fully Sharded Data Parallel) for memory efficiency
* Flash Attention 2 for faster training
* Extensive callback system for customization
* vLLM integration for fast checkpoint testing

**Best For:**
* Research projects requiring custom training logic
* Models 8B-70B parameters
* Teams familiar with PyTorch Lightning

**Documentation:** [ptl/README.md](./ptl/README.md)

---

### [Hugging Face Accelerate](./accelerate/)

Simplified distributed training using Hugging Face Trainer API.

**Key Features:**
* Minimal code with Trainer API
* FSDP configuration via YAML
* Seamless HuggingFace ecosystem integration
* Easy multi-node setup
* Gradient checkpointing and mixed precision

**Best For:**
* Quick experiments and prototyping
* Standard fine-tuning workflows
* Teams familiar with HuggingFace ecosystem
* Models 8B-70B parameters

**Documentation:** [accelerate/README.md](./accelerate/README.md)

---

### [Ray Train](./ray_train/)

Distributed training with Ray's orchestration for fault tolerance and scalability.

**Key Features:**
* Automatic fault tolerance with configurable retries
* Ray ecosystem integration (Tune, Serve, etc.)
* Distributed checkpointing
* Flexible resource management
* FSDP support with HuggingFace Trainer

**Best For:**
* Multi-cluster deployments
* Long-running training jobs requiring fault tolerance
* Integration with Ray ecosystem (hyperparameter tuning, serving)
* Models 8B-70B parameters

**Documentation:** [ray_train/README.md](./ray_train/README.md)

## Common Workflow

All frameworks follow a similar workflow:

### 1. Dataset Preparation

Each framework uses `HFDatasetConfig` for flexible dataset loading.

Datasets are automatically:
* Downloaded from HuggingFace
* Converted to JSONL format
* Split into train/validation/test sets
* Cached in `datasets/{dataset_name}/{config}/train={%}-val={%}-test={%}/`

### 2. Training

Each framework provides CLI arguments for configuration.

### 3. Testing Checkpoints

All frameworks include testing scripts.

Tests automatically:
* Find latest checkpoint
* Load and merge LoRA weights (if applicable)
* Run inference on test set
* Evaluate with BERTScore

### 4. Converting to HuggingFace Format

All frameworks support conversion to standard HuggingFace format:

By default, LoRA weights are merged into the base model for maximum compatibility with deployment tools (vLLM, TGI, etc.).

## Docker Containers

All frameworks provide Docker containers for reproducibility.

## LoRA vs Full Fine-Tuning

All frameworks support both LoRA (Parameter-Efficient Fine-Tuning) and full fine-tuning:

### LoRA (Default)

**Advantages:**
* Significantly lower memory requirements
* Faster training
* Smaller checkpoint sizes
* Can train larger models on same hardware

### Full Fine-Tuning

**Advantages:**
* Potentially better performance
* No adapter overhead at inference
* Full model customization


**Note:** Full fine-tuning requires significantly more GPU memory and uses different learning rates.

## Checkpoint Management

### Checkpoint Formats

| Framework | Training Format | Converted Format |
|-----------|----------------|------------------|
| NeMo 2.0 | NeMo checkpoint (context/ + weights/) | HuggingFace |
| PyTorch Lightning | .ckpt (PyTorch Lightning) | HuggingFace |
| Accelerate | HuggingFace checkpoint-* | HuggingFace |
| Ray Train | HuggingFace checkpoint-* | HuggingFace |

### LoRA Merging

By default, all frameworks merge LoRA weights into the base model during conversion.

## Monitoring and Logging

All frameworks support Tensorboard and Wandb logging backends.

## Requirements

* AWS Deep Learning Desktop with GPU or Neuron instances
* Docker and Docker Compose
* HuggingFace account and access token (for gated models)
* EFS mounted at `/home/ubuntu/efs` for model/dataset caching
* Sufficient GPU memory for target model size

## Additional Resources

* [NeMo 2.0 Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html)
* [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
* [HuggingFace Accelerate Documentation](https://huggingface.co/docs/accelerate/index)
* [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
* [PEFT Documentation](https://huggingface.co/docs/peft/index)
* [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

## Support

For framework-specific questions, refer to the individual README files:
* [NeMo 2.0 README](./nemo2/README.md)
* [PyTorch Lightning README](./ptl/README.md)
* [Accelerate README](./accelerate/README.md)
* [Ray Train README](./ray_train/README.md)
