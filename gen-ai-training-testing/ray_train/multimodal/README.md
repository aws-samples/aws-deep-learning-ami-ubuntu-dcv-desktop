# Multi-Modal Training with Ray Train

This directory contains implementations for training multi-modal models (vision-language, video-language, etc.) using Ray Train's `TorchTrainer` with FSDP.

## Directory Structure

```
multimodal/
├── vision_language/           # Text + Image models (VLMs)
│   └── README.md
│
├── shared/                    # Multi-modal specific utilities
│   └── README.md
│
└── README.md                  # This file
```

## Status

✅ **Vision-Language Training** - Fully implemented with Qwen3-VL support

## Modality Types

### Vision-Language (Text + Image)

Train models that understand both text and images:
- **Models**: Qwen3-VL (supported)
- **Use Cases**: Visual question answering, image captioning, visual instruction following
- **Documentation**: [vision_language/README.md](./vision_language/README.md)

### Future Modalities

Planned support for:
- **Video-Language**: Video understanding, video captioning, temporal reasoning
- **Audio-Language**: Speech understanding, audio captioning
- **Any-to-Any**: Models handling multiple modalities simultaneously

## Quick Start

> **Note**: For installation instructions, see the [parent README](../README.md).

### Vision-Language SFT

```bash
cd /app

python multimodal/vision_language/ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --freeze_vision_encoder
```

### Vision-Language CPT

```bash
cd /app

python multimodal/vision_language/cpt_ray_train_vlm.py \
  --model_id "Qwen/Qwen3-VL-8B-Instruct" \
  --hf_dataset_name "lmms-lab/LLaVA-NeXT-Data" \
  --num_train_epochs 3
```

## Key Differences from Text-Only Training

### Model Architecture
- Vision/Video encoder + Language model
- Requires both image/video processor and tokenizer
- Additional projection layers between modalities

### Dataset Processing
- Loads and preprocesses images/videos
- Combines visual embeddings with text tokens
- Handles various input formats (URLs, local paths, base64)

### LoRA Configuration
- Can apply LoRA to language model only (most common)
- Can apply LoRA to vision/video encoder (optional)
- Can apply LoRA to projection layers

### Training Considerations
- Higher memory requirements due to visual inputs
- Longer training times (processing images/videos)
- May need to freeze vision encoder for efficiency
- Batch size must be 1 for dynamic resolution models

## Shared Utilities

The `shared/` directory contains utilities used across different multi-modal implementations:

- **Image Processing**: Loading, resizing, normalization, augmentation
- **Video Processing**: Frame sampling, temporal ordering, video decoding (planned)
- **Base Dataset Classes**: Common multi-modal dataset patterns
- **Data Collators**: Batching visual and text inputs together

See [shared/README.md](./shared/README.md) for details.

## Common Features (Planned)

### Training Capabilities
- Supervised Fine-Tuning (SFT) for multi-modal models ✅
- Continual Pre-Training (CPT) for multi-modal models ✅
- DPO for multi-modal alignment (planned)
- Distributed training with FSDP via Ray Train ✅
- LoRA and full fine-tuning support ✅
- Flash Attention 2 support ✅
- Gradient checkpointing for memory efficiency ✅

### Dataset Support
- Image-text pair datasets ✅
- Visual instruction tuning datasets ✅
- Multi-turn visual conversations ✅
- Flexible preprocessing and augmentation ✅

### Vision Encoder Options
- Frozen (most memory efficient) ✅
- LoRA fine-tuning (balanced) ✅
- Full fine-tuning (most flexible, highest memory) ✅

## GPU Requirements

### Vision-Language Models (7B-13B)
- **GPUs**: 8x A100 (80GB) recommended
- **Batch size**: 1 per device (images are memory intensive)
- **Gradient accumulation**: 8-16
- **Notes**: Freezing vision encoder significantly reduces memory

## Implementation Roadmap

### Phase 1: Vision-Language ✅ Complete
- ✅ VLM SFT with Ray Train (Qwen3-VL)
- ✅ VLM CPT with Ray Train
- ✅ Image dataset processing
- ✅ Vision encoder freezing/LoRA options
- ✅ HuggingFace dataset integration
- ✅ Adapter pattern for extensibility
- ✅ Comprehensive documentation

### Phase 2: Advanced Features (Planned)
- [ ] Video-language model training
- [ ] DPO for multi-modal alignment
- [ ] Advanced augmentation strategies
- [ ] Multi-modal evaluation metrics

## Related Resources

- [HuggingFace Vision-Language Models](https://huggingface.co/models?pipeline_tag=image-text-to-text)
- [Qwen-VL Documentation](https://github.com/QwenLM/Qwen-VL)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [Text Training Documentation](../text/README.md) - Reference implementation

## Support

For questions or contributions:
- Vision-language: See [vision_language/README.md](./vision_language/README.md)
- Shared utilities: See [shared/README.md](./shared/README.md)
- Text training reference: See [../text/README.md](../text/README.md)
