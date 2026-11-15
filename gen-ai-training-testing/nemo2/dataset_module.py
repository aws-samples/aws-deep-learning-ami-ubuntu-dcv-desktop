"""Turn a Hugging Face dataset into the JSONL layout Megatron-Bridge SFT expects.

Previously this subclassed NeMo 2.0's `FineTuningDataModule`. That class went away when
NeMo 25.11 split the framework and LLM training moved to Megatron-Bridge, so the data
module is now a plain preparer: it does the same Hugging Face load, split and template
conversion, and writes the files Megatron-Bridge reads directly.

Megatron-Bridge's `GPTSFTDatasetConfig(dataset_root=...)` expects a directory holding
`training.jsonl` and optionally `validation.jsonl` / `test.jsonl`, one JSON object per
line. With `PromptCompletionSFTPreprocessingConfig(prompt_column="input",
completion_column="output")` the rows are `{"input": ..., "output": ...}`, which is
exactly what the old conversion already produced -- so the template and field-mapping
behaviour is unchanged.

Usable as a library or directly:

    python dataset_module.py --dataset_name tatsu-lab/alpaca --dataset_root ./data/alpaca

then point training at it:

    torchrun --nproc_per_node=8 peft_megatron.py --dataset_root ./data/alpaca ...
"""
import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from datasets import DatasetDict, load_dataset

# Filenames Megatron-Bridge looks for inside dataset_root
TRAIN_FILE = "training.jsonl"
VALIDATION_FILE = "validation.jsonl"
TEST_FILE = "test.jsonl"
READY_MARKER = ".data_ready"


@dataclass
class HFDatasetConfig:
    """Configuration for loading and converting HuggingFace datasets."""

    dataset_name: str
    """HuggingFace dataset name (e.g., 'tatsu-lab/alpaca')"""

    dataset_config: Optional[str] = None
    """Dataset configuration/subset name (e.g., 'flan1m-alpaca-uncensored')"""

    split: str = "train"
    """Initial split to load from HuggingFace"""

    train_split_ratio: float = 0.9
    """Ratio of data to use for training (remaining is split between val and test)"""

    val_test_split_ratio: float = 0.5
    """Ratio to split remaining data between validation and test"""

    input_template: str = "### Instruction:\n{instruction}\n ### Input:\n{input}\n"
    """Template for formatting input. Use {field_name} for dataset field placeholders"""

    output_template: str = "### Response:\n{output}"
    """Template for formatting output. Use {field_name} for dataset field placeholders"""

    field_mapping: Optional[Dict[str, str]] = None
    """Mapping from template placeholders to actual dataset column names.
    Example: {'instruction': 'text', 'input': 'context', 'output': 'answer'}
    If None, assumes template placeholders match dataset column names exactly."""

    num_proc: int = 8
    """Number of processes for dataset loading"""

    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to pass to load_dataset"""

    custom_converter: Optional[Callable] = None
    """Optional custom function to convert a dataset sample to input/output dict.
    Should have signature: func(sample: Dict) -> Dict[str, str] with keys 'input' and 'output'"""


class HFDatasetPreparer:
    """Materialise a Hugging Face dataset as prompt/completion JSONL files."""

    def __init__(self, config: HFDatasetConfig, dataset_root: str):
        self.config = config
        self.dataset_root = dataset_root
        self.train_path = os.path.join(dataset_root, TRAIN_FILE)
        self.validation_path = os.path.join(dataset_root, VALIDATION_FILE)
        self.test_path = os.path.join(dataset_root, TEST_FILE)

    def _convert_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Render one dataset row into {"input": ..., "output": ...}."""
        if self.config.custom_converter is not None:
            return self.config.custom_converter(sample)

        mapping = self.config.field_mapping or {}
        values = {}
        for field_name in self._extract_template_fields():
            column = mapping.get(field_name, field_name)
            if column not in sample:
                raise KeyError(
                    f"Template field '{field_name}' maps to column '{column}', which is "
                    f"not in the dataset. Available columns: {sorted(sample.keys())}. "
                    f"Use field_mapping to map template fields onto column names."
                )
            values[field_name] = sample[column]

        return {
            "input": self.config.input_template.format(**values),
            "output": self.config.output_template.format(**values),
        }

    def _extract_template_fields(self) -> set:
        """Extract field names from templates."""
        pattern = r'\{(\w+)\}'
        input_fields = set(re.findall(pattern, self.config.input_template))
        output_fields = set(re.findall(pattern, self.config.output_template))
        return input_fields | output_fields

    def _convert_hf_dataset_to_jsonl(self, dataset, path: str) -> None:
        with open(path, "w", encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(self._convert_sample(sample), ensure_ascii=False) + "\n")

    def _load_and_split_dataset(self) -> DatasetDict:
        """Load from the Hub and split into train/val/test."""
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            num_proc=self.config.num_proc,
            **self.config.load_kwargs
        )

        if self.config.split not in dataset:
            raise ValueError(
                f"Split '{self.config.split}' not found in dataset. "
                f"Available splits: {list(dataset.keys())}"
            )

        initial_data = dataset[self.config.split]
        train_testval = initial_data.train_test_split(
            test_size=1.0 - self.config.train_split_ratio
        )
        test_val = train_testval['test'].train_test_split(
            test_size=self.config.val_test_split_ratio
        )
        return DatasetDict({
            'train': train_testval['train'],
            'val': test_val['train'],
            'test': test_val['test'],
        })

    def prepare(self, overwrite: bool = False) -> str:
        """Write the JSONL files, skipping the work if they are already present.

        Returns the dataset_root to hand to GPTSFTDatasetConfig.
        """
        os.makedirs(self.dataset_root, exist_ok=True)
        marker = os.path.join(self.dataset_root, READY_MARKER)

        if os.path.exists(marker) and not overwrite:
            print(f"Dataset already prepared in {self.dataset_root} "
                  f"(delete {READY_MARKER} to rebuild)")
            return self.dataset_root

        print(f"Loading dataset '{self.config.dataset_name}'...")
        hf_dataset = self._load_and_split_dataset()

        print("Converting to JSONL format...")
        print(f"  Train samples: {len(hf_dataset['train'])}")
        print(f"  Val samples: {len(hf_dataset['val'])}")
        print(f"  Test samples: {len(hf_dataset['test'])}")

        self._convert_hf_dataset_to_jsonl(hf_dataset['train'], self.train_path)
        self._convert_hf_dataset_to_jsonl(hf_dataset['val'], self.validation_path)
        self._convert_hf_dataset_to_jsonl(hf_dataset['test'], self.test_path)

        with open(marker, 'w') as f:
            f.write('ready')

        print(f"Dataset preparation complete: {self.dataset_root}")
        return self.dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_root", required=True,
                        help="output directory, passed to training as --dataset_root")
    parser.add_argument("--train_split_ratio", type=float, default=0.9)
    parser.add_argument("--val_test_split_ratio", type=float, default=0.5)
    parser.add_argument("--input_template", default=HFDatasetConfig.input_template)
    parser.add_argument("--output_template", default=HFDatasetConfig.output_template)
    parser.add_argument("--field_mapping", default=None,
                        help='JSON object mapping template fields to columns, e.g. '
                             '\'{"instruction": "text"}\'')
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = HFDatasetConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        train_split_ratio=args.train_split_ratio,
        val_test_split_ratio=args.val_test_split_ratio,
        input_template=args.input_template,
        output_template=args.output_template,
        field_mapping=json.loads(args.field_mapping) if args.field_mapping else None,
        num_proc=args.num_proc,
    )
    HFDatasetPreparer(config, args.dataset_root).prepare(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
