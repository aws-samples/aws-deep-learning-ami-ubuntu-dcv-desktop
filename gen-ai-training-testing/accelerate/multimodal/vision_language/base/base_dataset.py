"""Base dataset class for vision-language model training using adapters."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import requests
from io import BytesIO
import base64

from .base_adapter import BaseVLMAdapter


class VLMDataset(Dataset):
    """
    Vision-language dataset using adapter pattern.
    
    This dataset works with any VLM family by using adapters to handle
    model-specific image processing and conversation formatting.
    """
    
    def __init__(
        self,
        data_path: Path,
        adapter: BaseVLMAdapter,
        processor,
        tokenizer,
        max_seq_length: int = 2048,
        is_test: bool = False,
    ):
        """
        Initialize VLM dataset.
        
        Args:
            data_path: Path to JSONL dataset file
            adapter: Model-specific adapter
            processor: Model processor
            tokenizer: Model tokenizer
            max_seq_length: Maximum sequence length
            is_test: Whether this is test dataset (affects label masking)
        """
        self.data_path = data_path
        self.adapter = adapter
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        
        # Load samples
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dict containing:
                - pixel_values: Processed image tensor
                - image_grid_thw: Image grid metadata (for Qwen3-VL)
                - input_ids: Tokenized input
                - labels: Labels for training (with instruction masked)
                - attention_mask: Attention mask
        """
        sample = self.samples[idx]
        
        try:
            # Validate sample has required fields
            if 'image' not in sample or 'conversations' not in sample:
                raise ValueError(f"Sample missing required fields: {sample.keys()}")
            
            if not sample['conversations'] or len(sample['conversations']) == 0:
                raise ValueError("Sample has empty conversations")
            
            # Load image
            image = self._load_image(sample['image'])
            
            # Format conversation using adapter
            formatted_text = self.adapter.format_conversation(
                sample['conversations'],
                self.processor,
                self.tokenizer
            )
            
            # Validate formatted text
            if formatted_text is None or not formatted_text.strip():
                raise ValueError("Formatted text is None or empty")
            
            # Process image and text together (required for Qwen3-VL)
            # The processor needs both to generate proper image_grid_thw
            inputs = self.processor(
                text=[formatted_text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            
            # Extract components
            input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
            pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
            image_grid_thw = inputs.get('image_grid_thw')
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.squeeze(0)  # Remove batch dimension
            
            # Truncate if needed
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
            
            # Create labels (copy of input_ids)
            labels = input_ids.clone()
            
            # Mask instruction part (only train on assistant responses)
            if not self.is_test:
                labels = self._mask_instruction_tokens(
                    labels, 
                    sample['conversations'],
                    formatted_text
                )
            
            # Get attention mask
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(0)  # Remove batch dimension
                if len(attention_mask) > self.max_seq_length:
                    attention_mask = attention_mask[:self.max_seq_length]
            else:
                attention_mask = torch.ones_like(input_ids)
            
            result = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            }
            
            # Add image_grid_thw if present (required for Qwen3-VL)
            if image_grid_thw is not None:
                result['image_grid_thw'] = image_grid_thw
            
            return result
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            print(f"Sample keys: {sample.keys()}")
            if 'conversations' in sample:
                print(f"Conversations: {sample['conversations'][:2] if len(sample['conversations']) > 2 else sample['conversations']}")
            # Return a dummy sample to avoid crashing the dataloader
            # The trainer will skip it
            dummy_input_ids = torch.zeros(10, dtype=torch.long)
            return {
                'pixel_values': torch.zeros(3, 224, 224),
                'input_ids': dummy_input_ids,
                'labels': torch.full_like(dummy_input_ids, -100),
                'attention_mask': torch.zeros_like(dummy_input_ids)
            }
    
    def _load_image(self, image_source: str) -> Image.Image:
        """
        Load image from path, URL, or base64.
        
        Args:
            image_source: Image path, URL, or base64 string
            
        Returns:
            PIL Image in RGB format
        """
        try:
            if image_source.startswith('http://') or image_source.startswith('https://'):
                # Load from URL
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif image_source.startswith('data:image'):
                # Load from base64
                image_data = image_source.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_data)))
            else:
                # Load from local path
                image_path = Path(image_source)
                if not image_path.is_absolute():
                    # Try relative to dataset directory
                    image_path = self.data_path.parent / image_source
                image = Image.open(image_path)
            
            return image.convert('RGB')
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_source}: {e}")
    
    def _mask_instruction_tokens(
        self,
        labels: torch.Tensor,
        conversations: list,
        formatted_text: str
    ) -> torch.Tensor:
        """
        Mask instruction tokens (only train on assistant responses).
        
        This is a simplified version that masks everything except the last
        assistant response. For more sophisticated masking, this can be
        overridden or made adapter-specific.
        
        Args:
            labels: Label tensor
            conversations: Original conversations
            formatted_text: Formatted text
            
        Returns:
            Masked labels tensor
        """
        # Simple approach: Find assistant responses and only keep those
        # For now, we'll use a heuristic based on the conversation structure
        
        # Count human turns to determine where to start unmasking
        human_turns = sum(1 for conv in conversations if conv['from'] == 'human')
        
        # If there are multiple turns, we want to mask the instructions
        # This is a simplified approach - can be made more sophisticated
        if human_turns > 0:
            # Mask first ~60% of tokens (rough heuristic for instruction part)
            mask_until = int(len(labels) * 0.6)
            if isinstance(labels, torch.Tensor):
                labels[:mask_until] = -100
            else:
                labels = [-100] * mask_until + labels[mask_until:]
        
        return labels
