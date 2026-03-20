# Multi-Modal Shared Utilities

Common utilities for multi-modal training (vision-language, video-language, etc.) with Ray Train.

## Status

🚧 **Under Development** - Utilities will be implemented alongside multi-modal training support.

## Overview

This directory contains utilities shared across different multi-modal implementations:
- Image processing and preprocessing
- Video processing and frame sampling (planned)
- Base dataset classes
- Data collators
- Common helper functions

## Planned Structure

```
shared/
├── __init__.py                    # Package initialization
├── image_processors.py            # Image loading and preprocessing
├── video_processors.py            # Video loading and frame sampling
├── base_dataset.py                # Base multi-modal dataset classes
├── collators.py                   # Data collators for batching
├── utils.py                       # Helper functions
└── README.md                      # This file
```

## Planned Utilities

### image_processors.py

Image loading, preprocessing, and augmentation utilities.

**Functions:**
```python
def load_image(source: str) -> Image:
    """Load image from path, URL, or base64."""
    
def preprocess_image(image: Image, size: int, processor) -> torch.Tensor:
    """Preprocess image for model input."""
    
def augment_image(image: Image, augmentation_config: dict) -> Image:
    """Apply data augmentation to image."""
```

### video_processors.py

Video loading, frame sampling, and preprocessing utilities.

**Functions:**
```python
def load_video(path: str, num_frames: int = 8) -> np.ndarray:
    """Load video and sample frames."""
    
def uniform_sample_frames(video: np.ndarray, num_frames: int) -> np.ndarray:
    """Sample frames uniformly across video."""
    
def preprocess_video(frames: np.ndarray, processor) -> torch.Tensor:
    """Preprocess video frames for model input."""
```

### base_dataset.py

Base dataset classes for multi-modal training.

**Classes:**
- `BaseMultiModalDataset`: Base class for multi-modal datasets
- `ImageTextDataset`: Dataset for image-text pairs
- `VideoTextDataset`: Dataset for video-text pairs (planned)

### collators.py

Data collators for batching multi-modal inputs.

**Classes:**
- `ImageTextCollator`: Collator for image-text data
- `VideoTextCollator`: Collator for video-text data (planned)

### utils.py

General helper functions for multi-modal training.

**Functions:**
```python
def find_vision_encoder(model):
    """Find vision encoder module in multi-modal model."""
    
def freeze_vision_encoder(model):
    """Freeze vision encoder parameters."""
    
def count_trainable_parameters(model):
    """Count trainable parameters in model."""
```

## Design Principles

### 1. Reusability
Utilities should be reusable across vision-language and video-language implementations.

### 2. Extensibility
Easy to extend for new modalities (audio, video, etc.).

### 3. Consistency
Consistent interfaces across all utilities.

### 4. Performance
Optimized for training efficiency with parallel loading and direct tensor conversion.

## Dependencies

### Core
- `torch`: PyTorch
- `transformers`: HuggingFace Transformers
- `PIL`: Image processing
- `numpy`: Array operations

### Optional
- `torchvision`: Image transformations
- `opencv-python`: Advanced image processing
- `decord`: Fast video loading (planned)

## Contributing

When adding new utilities:

1. Follow existing patterns and interfaces
2. Add comprehensive docstrings
3. Include usage examples
4. Write unit tests
5. Update this README
6. Consider performance implications

## Related Resources

- [PIL Documentation](https://pillow.readthedocs.io/)
- [torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)
- [HuggingFace Processors](https://huggingface.co/docs/transformers/main_classes/processors)
