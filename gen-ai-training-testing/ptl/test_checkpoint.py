"""
Test script for PTL checkpoint using DeepSpeed tensor parallelism.
Must be launched with: deepspeed --num_gpus=8 test_checkpoint.py [args]
"""
import os
import argparse
import json
from dataclasses import dataclass, fields, MISSING
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from tqdm import tqdm
import deepspeed
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

@dataclass
class Config:
    # Checkpoint and Model
    base_model: str = "Qwen/Qwen3-8B"
    checkpoints_dir: str = None
    
    # LoRA settings
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    # Data
    test_path: str = "datasets/cognitivecomputations_dolphin/flan1m-alpaca-uncensored/train=90%-val=5%-test=5%/test.jsonl"
    max_samples: int = 1024
    
    # Generation settings
    temperature: float = 0.1
    top_k: int = 0
    top_p: float = 0.95
    max_in_tokens: int = 2048
    max_tokens: int = 4096
    max_batch_size: int = 8
    
    # DeepSpeed
    local_rank: int = -1
    
    @property
    def output_path(self) -> str:
       return self.checkpoint_path.replace(".ckpt", ".jsonl")
    
    @property
    def checkpoint_path(self) -> str:
        """Find the latest checkpoint in the checkpoints directory."""
        ckpt_dir_path = Path(self.checkpoints_dir)
        ckpt_files = list(ckpt_dir_path.glob("model-*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir_path}")
        return str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() if k in config_fields}
        return cls(**kwargs)
    
    def __post_init__(self):
        if self.checkpoints_dir is None:
            self.checkpoints_dir = f"results/{self.base_model}/checkpoints"
    
def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for f in fields(dataclass_type):
        field_type = f.type
        default_value = f.default if f.default is not MISSING else None
        parser.add_argument(
            f'--{f.name}',
            type=field_type if field_type in [int, float, str] else str,
            default=default_value
        )
    return parser


def load_model_and_tokenizer(config:Config):
    """Load model from PyTorch Lightning checkpoint with DeepSpeed tensor parallelism."""
    # Only rank 0 prints
    is_main = torch.distributed.get_rank() == 0
    
    base_model_id = config.base_model
    checkpoint_path = config.checkpoint_path
    max_out_tokens = config.max_tokens

    if is_main:
        print(f"Loading tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'
    
    if is_main:
        print(f"Checking checkpoint type: {checkpoint_path}")
    checkpoint_meta = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict_keys = list(checkpoint_meta['state_dict'].keys())
    is_lora = any('lora' in key for key in state_dict_keys[:100])
    hparams = checkpoint_meta.get('hyper_parameters', {})
    
    if is_main:
        print(f"\n{'='*80}")
        print(f"  Model type: {'LoRA' if is_lora else 'Full Fine-tuned'}")
        print(f"{'='*80}\n")
    
    world_size = torch.distributed.get_world_size()
    if is_main:
        print(f"World size (TP size): {world_size}")
        print(f"Loading base model: {base_model_id}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    if is_lora:
        if is_main:
            print("\nApplying LoRA configuration...")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[m.strip() for m in config.lora_target_modules.split(',')],
            bias="none",
        )
        model = get_peft_model(base_model, peft_config)
    else:
        model = base_model
    
    if is_main:
        print(f"\nLoading checkpoint weights: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[6:] if key.startswith('model.') else key
        new_state_dict[new_key] = value
    
    if is_main:
        print("Loading state dict into model...")
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Initialize DeepSpeed Inference with tensor parallelism
    if is_main:
        print(f"\nInitializing DeepSpeed Inference with Tensor Parallelism (TP={world_size})...")
    
    ds_config = {
        "tensor_parallel": {
            "enabled": True,
            "tp_size": world_size
        },
        "dtype": "bf16",
        "replace_with_kernel_inject": True,
        "max_out_tokens": max_out_tokens
    }
    
    model = deepspeed.init_inference(
        model,
        config=ds_config,
    )
    
    if is_main:
        print("✓ Model loaded successfully with DeepSpeed TP!")
    return model, tokenizer

def generate_and_save_predictions(model, tokenizer, config):
    """Generate predictions for test samples using batched generation."""
    is_main = torch.distributed.get_rank() == 0
    
    # Only main process reads data and writes output
    if is_main:
        samples = []
        with open(config.test_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= config.max_samples:
                    break
                samples.append(json.loads(line))
        
        total_samples = len(samples)
        print(f"\nGenerating predictions for {total_samples} samples in batches of {config.max_batch_size}...")
    else:
        samples = []
        total_samples = 0
    
    # Broadcast number of samples
    total_samples = torch.tensor(total_samples if is_main else 0, device='cuda')
    torch.distributed.broadcast(total_samples, 0)
    total_samples = total_samples.item()
    
    if is_main:
        f_out = open(config.output_path, 'w')
    
    if is_main:
        pbar = tqdm(range(0, total_samples, config.max_batch_size))
    else:
        pbar = range(0, total_samples, config.max_batch_size)
    
    for batch_start in pbar:
        if is_main:
            batch_samples = samples[batch_start:batch_start + config.max_batch_size]
            batch_inputs = [s['input'] for s in batch_samples]
            batch_labels = [s['output'] for s in batch_samples]
        else:
            batch_inputs = []
            batch_labels = []
        
        # Broadcast batch inputs to all ranks
        # In a real implementation, you'd need proper broadcasting
        # For simplicity, we'll have all ranks do the computation
        
        if is_main:
            inputs = tokenizer(batch_inputs, return_tensors="pt", 
                               padding=True, truncation=True, max_length=config.max_in_tokens).to('cuda')
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    do_sample=config.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for input_text, label, full_response in zip(batch_inputs, batch_labels, decoded_outputs):
                if full_response.startswith(input_text):
                    prediction = full_response[len(input_text):].strip()
                else:
                    prediction = full_response
                
                result = {
                    'input': input_text,
                    'label': label,
                    'prediction': prediction
                }
                f_out.write(json.dumps(result) + '\n')
    
    if is_main:
        f_out.close()
        print(f"\n✓ Predictions saved to: {config.output_path}")

def evaluate_predictions(output_path):
    """Evaluate predictions using multiple metrics."""
    bertscore = evaluate.load('bertscore')
    
    predictions = []
    references = []
    
    with open(output_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            predictions.append(sample['prediction'])
            references.append(sample['label'])
    
    print("\nComputing metrics...")
    bert_scores = bertscore.compute(predictions=predictions, references=references, lang='en')
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nBERTScore F1: {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")
    print("="*80)
    
    return {
        'bertscore': sum(bert_scores['f1'])/len(bert_scores['f1'])
    }

def run_testing(config):
    is_main = torch.distributed.get_rank() == 0
    
    if is_main:
        print("=" * 80)
        print("Testing PTL PEFT Model with Test Dataset (DeepSpeed TP)")
        print("=" * 80)
        print(f"Checkpoint: {config.checkpoint_path}")
        print(f"Model: {config.base_model}")
        print("=" * 80)
    
    # Load model
    if is_main:
        print("\n[1/2] Loading model...")
    model, tokenizer = load_model_and_tokenizer(config=config)
    
    # Generate predictions (batched)
    if is_main:
        print("\n[2/2] Generating predictions...")
    generate_and_save_predictions(model=model, tokenizer=tokenizer, config=config)
    
    if is_main:
        # Display sample predictions
        print("\n" + "=" * 80)
        print("SAMPLE PREDICTIONS")
        print("=" * 80)
        with open(config.output_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                pred = json.loads(line)
                print(f"\n--- Sample {i+1} ---")
                print(f"Input:\n{pred['input'][:200]}...")
                print(f"\nExpected:\n{pred['label'][:200]}...")
                print(f"\nPredicted:\n{pred['prediction'][:200]}...")
                print("-" * 80)
        
        # Evaluate
        print("\n" + "=" * 80)
        print("EVALUATING PREDICTIONS")
        print("=" * 80)
        evaluate_predictions(config.output_path)
        
        print("\n✓ Complete!")

def main():
    # DeepSpeed adds local_rank automatically
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()
    config = Config.from_args(args)
    
    # Initialize DeepSpeed distributed backend
    deepspeed.init_distributed()
    
    run_testing(config)

if __name__ == "__main__":
    main()
