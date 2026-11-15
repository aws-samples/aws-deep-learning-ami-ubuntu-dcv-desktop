"""Evaluate a fine-tuned model with vLLM and BERTScore.

Rewritten for NeMo Framework 26.x. The previous version drove NeMo 2.0's dynamic
inference engine through `nemo.lightning`, `nemo.collections.llm.inference` and
`nemo.lightning.ckpt_utils`, none of which exist after the 25.11 split.

Rather than port that to Megatron-Bridge's inference stack, this now evaluates a
plain Hugging Face model directory with vLLM, which is what the accelerate, ptl and
ray_train examples in this repository already do -- so all four frameworks are scored
the same way and the numbers are comparable. Produce the HF directory first with
convert_checkpoint_to_hf.py, which merges the LoRA adapter into the base weights.

    python convert_checkpoint_to_hf.py --megatron_checkpoint ./outputs/.../checkpoints \
        --base_model /snapshots/huggingface/Qwen/Qwen3-8B --output_dir ./merged
    python test_checkpoint.py --model_dir ./merged --test_path ./data/alpaca/test.jsonl

Unlike training, this path has no sm90 requirement: vLLM runs on Ada-class GPUs.
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, fields, MISSING

import evaluate
from vllm import LLM, SamplingParams

from device_preflight import detected_gpu_count, select_tensor_parallel_size


@dataclass
class Config:
    """Evaluation configuration."""

    # Hugging Face model directory to evaluate: either a merged fine-tuned model from
    # convert_checkpoint_to_hf.py, or any base model for a reference score.
    model_dir: str = None

    test_path: str = None
    output_path: str = None

    max_samples: int = 1024
    batch_size: int = 128

    # Sampling
    temperature: float = 0.1
    top_k: int = -1            # vLLM uses -1 for disabled
    top_p: float = 0.95
    max_tokens: int = 512      # max new tokens

    # Serving. 0 means auto-select from the visible GPUs and the model's KV heads.
    tensor_parallel_size: int = 0
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items()
                  if k in config_fields and v is not None}
        return cls(**kwargs)


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(dataclass_type):
        default = None if f.default is MISSING else f.default
        if f.type is bool or isinstance(default, bool):
            parser.add_argument(f"--{f.name}", action="store_true", default=None)
        else:
            arg_type = int if f.type in (int, 'int') else (float if f.type in (float, 'float') else str)
            parser.add_argument(f"--{f.name}", type=arg_type, default=None)
    return parser


def load_model_with_vllm(config: Config) -> LLM:
    print(f"Loading {config.model_dir} with vLLM "
          f"(tensor_parallel_size={config.tensor_parallel_size})...")
    return LLM(
        model=config.model_dir,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
    )


def generate_and_save_predictions(llm: LLM, config: Config) -> None:
    """Generate predictions for test samples using vLLM batched inference."""
    sampling_params = SamplingParams(
        temperature=config.temperature if config.temperature > 0 else 0.0,
        top_k=config.top_k if config.top_k > 0 else -1,
        top_p=config.top_p if config.temperature > 0 else 1.0,
        max_tokens=config.max_tokens,
    )

    print(f"\nGenerating predictions (batch_size={config.batch_size})...")
    start_time = time.time()
    total_samples = 0
    total_tokens_generated = 0

    def flush(f_out, inputs, labels):
        nonlocal total_samples, total_tokens_generated
        if not inputs:
            return
        outputs = llm.generate(inputs, sampling_params)
        for input_text, label, output in zip(inputs, labels, outputs):
            f_out.write(json.dumps({
                'input': input_text,
                'label': label,
                'prediction': output.outputs[0].text,
            }) + '\n')
            total_tokens_generated += len(output.outputs[0].token_ids)
        total_samples += len(inputs)
        print(f"  Processed {total_samples} samples...")

    with open(config.output_path, 'w') as f_out:
        chunk_inputs, chunk_labels = [], []
        with open(config.test_path, 'r') as f_in:
            for idx, line in enumerate(f_in):
                if idx >= config.max_samples:
                    break
                sample = json.loads(line)
                chunk_inputs.append(sample['input'])
                chunk_labels.append(sample['output'])
                if len(chunk_inputs) >= config.batch_size:
                    flush(f_out, chunk_inputs, chunk_labels)
                    chunk_inputs, chunk_labels = [], []
        flush(f_out, chunk_inputs, chunk_labels)

    generation_time = time.time() - start_time
    print(f"\n✓ Predictions saved to: {config.output_path}")
    print(f"\nTiming Statistics:")
    print(f"  Total time:             {generation_time:.2f}s")
    print(f"  Samples/sec:            {total_samples/generation_time:.2f}")
    print(f"  Total tokens generated: {total_tokens_generated}")
    print(f"  Tokens/sec:             {total_tokens_generated/generation_time:.0f}")
    print(f"  Avg tokens/sample:      {total_tokens_generated/max(total_samples,1):.1f}")


def evaluate_predictions(output_path: str) -> dict:
    """Evaluate predictions using BERTScore."""
    bertscore = evaluate.load('bertscore')

    predictions, references = [], []
    with open(output_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            predictions.append(sample['prediction'])
            references.append(sample['label'])

    print("\nComputing metrics...")
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang='en')

    f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nBERTScore F1: {f1:.4f}")
    print("=" * 80)
    return {'bertscore': f1}


def run_testing(config: Config) -> dict:
    llm = load_model_with_vllm(config)
    generate_and_save_predictions(llm, config)
    return evaluate_predictions(config.output_path)


def main() -> None:
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()
    config = Config.from_args(args)

    if not config.model_dir or not os.path.isdir(config.model_dir):
        raise SystemExit(
            f"--model_dir must be a Hugging Face model directory; got "
            f"'{config.model_dir}'. Create one from a Megatron checkpoint with "
            f"convert_checkpoint_to_hf.py."
        )
    if not config.test_path or not os.path.isfile(config.test_path):
        raise SystemExit(
            f"--test_path must point at a test .jsonl with 'input'/'output' rows; got "
            f"'{config.test_path}'. dataset_module.py writes one as test.jsonl."
        )

    if config.tensor_parallel_size:
        config.tensor_parallel_size = select_tensor_parallel_size(
            detected_gpu_count(), config.model_dir, requested=config.tensor_parallel_size)
    else:
        # vLLM shards attention heads, so the size must divide both the GPU count and
        # the model's key/value head count. Unlike training, serving benefits from
        # using the GPUs available, so take the largest valid size.
        config.tensor_parallel_size = select_tensor_parallel_size(
            detected_gpu_count(), config.model_dir)

    if not config.output_path:
        config.output_path = os.path.join(
            os.path.dirname(os.path.abspath(config.test_path)), "predictions.jsonl")

    run_testing(config)


if __name__ == "__main__":
    main()
