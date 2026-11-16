import os
from datasets import load_dataset

class MultiModalPromptGenerator:

    def __init__(self) -> None:
        ds_name = os.getenv("MM_DATASET_NAME", "visual-mrc")
        # Loaded from the Hub's auto-generated parquet branch, since datasets>=4 dropped
        # script-based loading. That branch exposes no dataset card, so the config has to
        # be passed as data_dir; as name= it raises BuilderConfig not found.
        self.dataset = load_dataset("MMInstruction/M3IT", data_dir=ds_name, split="test",
                                    revision="refs/convert/parquet")

    def __call__(self) -> list:
        for example in self.dataset:
            instruction = example["instruction"]
            inputs = example["inputs"] 
            base64_image = example["image_base64_str"][0]
            yield [ f"{instruction} {inputs}", f"data:image/jpeg;base64,{base64_image}" ]