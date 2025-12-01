import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


@dataclass
class HFDatasetConfig:
    """Configuration for loading and converting HuggingFace datasets."""
    
    # Dataset loading parameters
    dataset_name: str
    """HuggingFace dataset name (e.g., 'cognitivecomputations/dolphin')"""
    
    dataset_config: Optional[str] = None
    """Dataset configuration/subset name (e.g., 'flan1m-alpaca-uncensored')"""
    
    split: str = "train"
    """Initial split to load from HuggingFace"""
    
    # Train/val/test split configuration
    train_split_ratio: float = 0.9
    """Ratio of data to use for training (remaining is split between val and test)"""
    
    val_test_split_ratio: float = 0.5
    """Ratio to split remaining data between validation and test"""
    
    # Data conversion parameters
    input_template: str = "### Instruction:\n{instruction}\n ### Input:\n{input}\n"
    """Template for formatting input. Use {field_name} for dataset field placeholders"""
    
    output_template: str = "### Response:\n{output}"
    """Template for formatting output. Use {field_name} for dataset field placeholders"""
    
    field_mapping: Optional[Dict[str, str]] = None
    """Mapping from template placeholders to actual dataset column names.
    Example: {'instruction': 'text', 'input': 'context', 'output': 'answer'}
    If None, assumes template placeholders match dataset column names exactly."""
    
    # Additional loading parameters
    num_proc: int = 8
    """Number of processes for dataset loading"""
    
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to load_dataset"""
    
    # Custom conversion function (advanced usage)
    custom_converter: Optional[Callable] = None
    """Optional custom function to convert a dataset sample to input/output dict.
    Should have signature: func(sample: Dict) -> Dict[str, str] with keys 'input' and 'output'"""


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning with tokenization."""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        is_test: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        
        # Load JSONL data
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Concatenate input and output for causal LM training
        full_text = sample['input'] + sample['output']
        
        # Tokenize the full text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        
        input_ids = tokenized['input_ids']
        
        # Create labels (for causal LM, labels = input_ids)
        labels = input_ids.copy()
        
        # Tokenize only the input to find where output starts
        input_tokenized = self.tokenizer(
            sample['input'],
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        input_length = len(input_tokenized['input_ids'])
        
        # Mask the input portion in labels (set to -100 to ignore in loss)
        # We want the model to only learn to predict the output, not the input
        labels[:input_length] = [-100] * input_length
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': tokenized.get('attention_mask', [1] * len(input_ids))
        }


class GeneralizedHFDataModule(pl.LightningDataModule):
    """Pure PyTorch Lightning data module for HuggingFace datasets."""
    
    def __init__(
        self,
        config: HFDatasetConfig,
        dataset_root: str,
        tokenizer_name: str,
        max_seq_length: int = 2048,
        micro_batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        """
        Initialize the data module.
        
        Args:
            config: HFDatasetConfig instance with dataset configuration
            dataset_root: Root directory to store converted datasets
            tokenizer_name: HuggingFace tokenizer name/path
            max_seq_length: Maximum sequence length
            micro_batch_size: Batch size per device
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            persistent_workers: Whether to keep workers persistent
        """
        super().__init__()
        self.config = config
        self.dataset_root = Path(dataset_root)
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # Create dataset directory
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = None
    
    def _convert_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert a single dataset sample to input/output format.
        
        Args:
            sample: A dictionary containing the dataset sample
            
        Returns:
            Dictionary with 'input' and 'output' keys
        """
        # Use custom converter if provided
        if self.config.custom_converter is not None:
            return self.config.custom_converter(sample)
        
        # Apply field mapping if provided
        if self.config.field_mapping is not None:
            mapped_sample = {
                placeholder: sample.get(self.config.field_mapping.get(placeholder, placeholder), "")
                for placeholder in self._extract_template_fields()
            }
        else:
            mapped_sample = sample
        
        # Format input and output using templates
        try:
            input_text = self.config.input_template.format(**mapped_sample)
            output_text = self.config.output_template.format(**mapped_sample)
        except KeyError as e:
            raise KeyError(
                f"Missing field {e} in dataset sample. "
                f"Available fields: {list(sample.keys())}. "
                f"Consider using field_mapping to map template placeholders to dataset columns."
            )
        
        return {"input": input_text, "output": output_text}
    
    def _extract_template_fields(self) -> set:
        """Extract field names from templates."""
        pattern = r'\{(\w+)\}'
        
        input_fields = set(re.findall(pattern, self.config.input_template))
        output_fields = set(re.findall(pattern, self.config.output_template))
        
        return input_fields | output_fields
    
    def _convert_hf_dataset_to_jsonl(self, dataset, path: Path):
        """
        Convert HuggingFace dataset to JSONL format.
        
        Args:
            dataset: HuggingFace dataset or dataset split
            path: Output path for JSONL file
        """
        with open(path, "w", encoding='utf-8') as f:
            for sample in dataset:
                converted = self._convert_sample(sample)
                json_string = json.dumps(converted, ensure_ascii=False) + "\n"
                f.write(json_string)
    
    def _load_and_split_dataset(self) -> DatasetDict:
        """
        Load dataset from HuggingFace and split into train/val/test.
        
        Returns:
            DatasetDict with 'train', 'val', and 'test' splits
        """
        # Load dataset
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            num_proc=self.config.num_proc,
            **self.config.load_kwargs
        )
        
        # Get the initial split
        if self.config.split not in dataset:
            raise ValueError(
                f"Split '{self.config.split}' not found in dataset. "
                f"Available splits: {list(dataset.keys())}"
            )
        
        initial_data = dataset[self.config.split]
        
        # Split into train and (val+test)
        train_testval = initial_data.train_test_split(
            test_size=1.0 - self.config.train_split_ratio
        )
        
        # Split (val+test) into val and test
        test_val = train_testval['test'].train_test_split(
            test_size=self.config.val_test_split_ratio
        )
        
        # Create final DatasetDict
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': test_val['train'],
            'test': test_val['test']
        })
        
        return split_dataset
    
    @property
    def train_path(self) -> Path:
        """Path to training dataset file"""
        return self.dataset_root / "training.jsonl"
    
    @property
    def validation_path(self) -> Path:
        """Path to validation dataset file"""
        return self.dataset_root / "validation.jsonl"
    
    @property
    def test_path(self) -> Path:
        """Path to test dataset file"""
        return self.dataset_root / "test.jsonl"
    
    def prepare_data(self):
        """Prepare data by converting HuggingFace dataset to JSONL format if needed."""
        marker_file = os.path.join(os.path.dirname(self.train_path), ".data_ready")

        if self.trainer is None or self.trainer.is_global_zero:
            if not os.path.exists(marker_file):
                print(f"Loading dataset '{self.config.dataset_name}'...")
                hf_dataset = self._load_and_split_dataset()
                
                print(f"Converting to JSONL format...")
                print(f"  Train samples: {len(hf_dataset['train'])}")
                print(f"  Val samples: {len(hf_dataset['val'])}")
                print(f"  Test samples: {len(hf_dataset['test'])}")
                
                self._convert_hf_dataset_to_jsonl(hf_dataset['train'], self.train_path)
                self._convert_hf_dataset_to_jsonl(hf_dataset['val'], self.validation_path)
                self._convert_hf_dataset_to_jsonl(hf_dataset['test'], self.test_path)
                
                with open(marker_file, 'w') as f:
                    f.write('ready')
            print("Dataset preparation complete!")
        else:
            print(f"Global rank: {self.trainer.global_rank} waiting for data preparation to complete...")
            while not os.path.exists(marker_file):
                time.sleep(10)
        
        super().prepare_data()
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        # Initialize tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
                trust_remote_code=True,
            )
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = SFTDataset(
                self.train_path,
                self.tokenizer,
                self.max_seq_length,
                is_test=False,
            )
            self.val_dataset = SFTDataset(
                self.validation_path,
                self.tokenizer,
                self.max_seq_length,
                is_test=True,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = SFTDataset(
                self.test_path,
                self.tokenizer,
                self.max_seq_length,
                is_test=True,
            )
    
    def collate_fn(self, batch):
        """Collate function to pad batch."""
        # Extract fields
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        
        # Pad sequences
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded,
            'attention_mask': attention_mask_padded,
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=self.collate_fn,
        )
