import datasets

class TextInputGenerator:

    def __init__(self) -> None:
        # The repo moved to zai-org and still ships only a loading script plus data.zip,
        # with no parquet branch, so the per-config jsonl is read out of the zip directly.
        self.dataset = datasets.load_dataset(
            'json',
            data_files="zip://data/narrativeqa.jsonl::hf://datasets/zai-org/LongBench@main/data.zip",
            split='train')
    
    def __call__(self) -> str:
        for example in self.dataset:
            context = example["context"]
            yield [context]
            
            
    