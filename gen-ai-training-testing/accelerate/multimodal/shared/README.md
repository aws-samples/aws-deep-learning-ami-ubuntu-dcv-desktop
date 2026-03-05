# Multi-Modal Shared Utilities

Common utilities for multi-modal training (vision-language, video-language, etc.).

## Status

🚧 **Under Development** - Utilities will be implemented alongside multi-modal training support.

## Overview

This directory contains utilities shared across different multi-modal implementations:
- Image processing and preprocessing
- Video processing and frame sampling
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

**Classes:**
- `ImageLoader`: Load images from various sources (path, URL, base64)
- `ImagePreprocessor`: Resize, normalize, augment images
- `ImageAugmenter`: Data augmentation for training

**Functions:**
```python
def load_image(source: str) -> Image:
    """Load image from path, URL, or base64."""
    
def preprocess_image(image: Image, size: int, processor) -> torch.Tensor:
    """Preprocess image for model input."""
    
def augment_image(image: Image, augmentation_config: dict) -> Image:
    """Apply data augmentation to image."""
```

**Example Usage:**
```python
from multimodal.shared.image_processors import load_image, preprocess_image

# Load image
image = load_image("path/to/image.jpg")

# Preprocess
pixel_values = preprocess_image(image, size=336, processor=processor)
```

---

### video_processors.py

Video loading, frame sampling, and preprocessing utilities.

**Classes:**
- `VideoLoader`: Load videos and extract frames
- `FrameSampler`: Various frame sampling strategies
- `VideoPreprocessor`: Process video frames

**Functions:**
```python
def load_video(path: str, num_frames: int = 8) -> np.ndarray:
    """Load video and sample frames."""
    
def uniform_sample_frames(video: np.ndarray, num_frames: int) -> np.ndarray:
    """Sample frames uniformly across video."""
    
def random_sample_frames(video: np.ndarray, num_frames: int) -> np.ndarray:
    """Randomly sample frames from video."""
    
def preprocess_video(frames: np.ndarray, processor) -> torch.Tensor:
    """Preprocess video frames for model input."""
```

**Example Usage:**
```python
from multimodal.shared.video_processors import load_video, preprocess_video

# Load and sample video
frames = load_video("path/to/video.mp4", num_frames=8)

# Preprocess
pixel_values = preprocess_video(frames, processor=processor)
```

---

### base_dataset.py

Base dataset classes for multi-modal training.

**Classes:**

```python
class BaseMultiModalDataset(Dataset):
    """Base class for multi-modal datasets."""
    
    def __init__(self, data_path, processor, tokenizer, max_seq_length):
        self.data_path = data_path
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = self.load_samples()
    
    def load_samples(self):
        """Load dataset samples."""
        raise NotImplementedError
    
    def process_visual_input(self, sample):
        """Process visual input (image or video)."""
        raise NotImplementedError
    
    def process_text_input(self, sample):
        """Process text input."""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """Get a single sample."""
        raise NotImplementedError


class ImageTextDataset(BaseMultiModalDataset):
    """Dataset for image-text pairs."""
    
    def process_visual_input(self, sample):
        """Process image input."""
        image = load_image(sample['image'])
        return preprocess_image(image, self.processor)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process image
        pixel_values = self.process_visual_input(sample)
        
        # Process text
        text_inputs = self.process_text_input(sample)
        
        return {
            'pixel_values': pixel_values,
            **text_inputs
        }


class VideoTextDataset(BaseMultiModalDataset):
    """Dataset for video-text pairs."""
    
    def __init__(self, data_path, processor, tokenizer, max_seq_length, num_frames=8):
        self.num_frames = num_frames
        super().__init__(data_path, processor, tokenizer, max_seq_length)
    
    def process_visual_input(self, sample):
        """Process video input."""
        frames = load_video(sample['video'], num_frames=self.num_frames)
        return preprocess_video(frames, self.processor)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process video
        pixel_values = self.process_visual_input(sample)
        
        # Process text
        text_inputs = self.process_text_input(sample)
        
        return {
            'pixel_values': pixel_values,
            **text_inputs
        }
```

---

### collators.py

Data collators for batching multi-modal inputs.

**Classes:**

```python
@dataclass
class ImageTextCollator:
    """Collator for image-text data."""
    
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    
    def __call__(self, features):
        # Batch images
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        
        # Batch text with padding
        input_ids = pad_sequence(
            [f['input_ids'] for f in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = pad_sequence(
            [f['labels'] for f in features],
            batch_first=True,
            padding_value=-100
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


@dataclass
class VideoTextCollator:
    """Collator for video-text data."""
    
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    
    def __call__(self, features):
        # Batch videos (with num_frames dimension)
        pixel_values = torch.stack([f['pixel_values'] for f in features])
        # Shape: (batch_size, num_frames, C, H, W)
        
        # Batch text with padding
        input_ids = pad_sequence(
            [f['input_ids'] for f in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = pad_sequence(
            [f['labels'] for f in features],
            batch_first=True,
            padding_value=-100
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
```

---

### utils.py

General helper functions for multi-modal training.

**Functions:**

```python
def find_vision_encoder(model):
    """Find vision encoder module in multi-modal model."""
    
def freeze_vision_encoder(model):
    """Freeze vision encoder parameters."""
    
def get_vision_lora_target_modules(model):
    """Get target modules for LoRA on vision encoder."""
    
def get_language_lora_target_modules(model):
    """Get target modules for LoRA on language model."""
    
def count_trainable_parameters(model):
    """Count trainable parameters in model."""
    
def print_model_structure(model):
    """Print model structure for debugging."""
```

**Example Usage:**
```python
from multimodal.shared.utils import freeze_vision_encoder, count_trainable_parameters

# Freeze vision encoder
freeze_vision_encoder(model)

# Check trainable parameters
trainable_params = count_trainable_parameters(model)
print(f"Trainable parameters: {trainable_params:,}")
```

## Design Principles

### 1. Reusability

Utilities should be reusable across vision-language and video-language:

```python
# Used by both vision_language and video_language
from multimodal.shared.image_processors import preprocess_image

# Vision-language: process single image
pixel_values = preprocess_image(image, processor)

# Video-language: process multiple frames
pixel_values = [preprocess_image(frame, processor) for frame in frames]
```

### 2. Extensibility

Easy to extend for new modalities:

```python
class AudioProcessor:
    """Extend for audio-language models."""
    
    def load_audio(self, path):
        """Load audio file."""
        pass
    
    def preprocess_audio(self, audio):
        """Preprocess audio for model."""
        pass
```

### 3. Consistency

Consistent interfaces across utilities:

```python
# All loaders follow same pattern
image = load_image(source)
video = load_video(source, num_frames=8)
audio = load_audio(source, sample_rate=16000)

# All preprocessors follow same pattern
pixel_values = preprocess_image(image, processor)
pixel_values = preprocess_video(frames, processor)
audio_values = preprocess_audio(audio, processor)
```

### 4. Performance

Optimized for training efficiency:

```python
# Efficient video loading with decord
import decord
decord.bridge.set_bridge('torch')  # Direct to torch tensors

# Parallel image loading
from concurrent.futures import ThreadPoolExecutor

def load_images_parallel(paths, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_image, paths))
    return images
```

## Dependencies

### Core
- `torch`: PyTorch
- `transformers`: HuggingFace Transformers
- `PIL`: Image processing
- `numpy`: Array operations

### Image Processing
- `torchvision`: Image transformations
- `opencv-python`: Advanced image processing

### Video Processing
- `decord`: Fast video loading (recommended)
- `opencv-python`: Alternative video loading
- `av`: PyAV for video processing

### Optional
- `albumentations`: Advanced augmentation
- `imagecorruptions`: Robustness testing

## Installation

```bash
# Core dependencies
pip install torch transformers pillow numpy

# Image processing
pip install torchvision opencv-python

# Video processing (choose one)
pip install decord  # Recommended
# or
pip install av

# Optional
pip install albumentations imagecorruptions
```

## Testing

Each utility should have unit tests:

```python
# test_image_processors.py
def test_load_image_from_path():
    image = load_image("test_image.jpg")
    assert isinstance(image, Image.Image)

def test_load_image_from_url():
    image = load_image("https://example.com/image.jpg")
    assert isinstance(image, Image.Image)

def test_preprocess_image():
    image = Image.new('RGB', (100, 100))
    pixel_values = preprocess_image(image, size=224, processor=processor)
    assert pixel_values.shape == (3, 224, 224)
```

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
- [decord Documentation](https://github.com/dmlc/decord)
- [HuggingFace Processors](https://huggingface.co/docs/transformers/main_classes/processors)
