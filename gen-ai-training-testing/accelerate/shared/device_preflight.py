"""Device detection and capacity preflight checks.

These examples previously hardcoded 8 GPUs per node, which silently oversubscribed or
divided to zero on any other machine. The helpers here let an example adapt to whatever
hardware it is on, and fail with an actionable message when the requested model simply
does not fit, instead of dying later in a CUDA OOM.

Deliberately dependency-light so it can be copied into each framework directory: only
that directory is bind-mounted into the training container.
"""
import json
import os
import subprocess
import sys

# bf16 weights + bf16 grads + fp32 master + fp32 Adam m/v
BYTES_PER_PARAM_FULL_FINETUNE = 16
# frozen bf16 base weights, plus a small allowance for adapters, grads and their states
BYTES_PER_PARAM_PEFT = 2.4
# leave room for activations, fragmentation and the CUDA context
USABLE_MEMORY_FRACTION = 0.85


def detected_gpu_count() -> int:
    """Number of visible GPUs, honouring CUDA_VISIBLE_DEVICES."""
    try:
        import torch
        count = torch.cuda.device_count()
        if count:
            return count
    except Exception:
        pass
    try:
        listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                text=True, check=True).stdout
        return len([line for line in listed.splitlines() if line.strip()])
    except Exception:
        return 0


def total_gpu_memory_bytes() -> int:
    try:
        import torch
        total = sum(torch.cuda.get_device_properties(i).total_memory
                    for i in range(torch.cuda.device_count()))
        if total:
            return total
    except Exception:
        pass
    # Fall back to nvidia-smi so the check still runs where torch is unavailable,
    # rather than silently skipping itself
    try:
        listed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        return sum(int(line) * 1024 ** 2 for line in listed.split() if line.strip())
    except Exception:
        return 0


def _resolve_model_dir(model_id: str) -> str:
    """Local snapshot path if the id points at one, else the id itself."""
    return model_id if os.path.isdir(model_id) else model_id


def model_config(model_id: str) -> dict:
    local = os.path.join(_resolve_model_dir(model_id), "config.json")
    if os.path.isfile(local):
        with open(local) as handle:
            return json.load(handle)
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_id, trust_remote_code=True).to_dict()


def model_param_count(model_id: str) -> int:
    """Parameter count, preferring the safetensors index which states total bytes."""
    index = os.path.join(_resolve_model_dir(model_id), "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as handle:
            total_bytes = json.load(handle).get("metadata", {}).get("total_size")
        if total_bytes:
            return int(total_bytes) // 2          # stored in bf16/fp16

    cfg = model_config(model_id)
    hidden = cfg.get("hidden_size")
    layers = cfg.get("num_hidden_layers")
    vocab = cfg.get("vocab_size")
    inter = cfg.get("intermediate_size", 4 * hidden if hidden else None)
    if not all((hidden, layers, vocab, inter)):
        return 0
    per_layer = 4 * hidden * hidden + 3 * hidden * inter
    return layers * per_layer + 2 * vocab * hidden


def select_tensor_parallel_size(gpu_count: int, model_id: str,
                                requested: int = None) -> int:
    """Largest tensor-parallel size valid for both the machine and the model.

    Tensor parallelism splits attention heads, so the size must divide the GPU count and
    the key/value head count. Qwen3-8B, for instance, has 32 attention heads but only 8
    KV heads, so a 6-GPU machine cannot use TP=6 even though 6 divides 6.
    """
    if gpu_count <= 0:
        sys.exit("No GPUs detected. This example requires at least one GPU.")

    cfg = model_config(model_id)
    kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 1

    if requested:
        if gpu_count % requested or kv_heads % requested:
            sys.exit(
                f"tensor_parallel_size={requested} is not usable: it must divide both the "
                f"visible GPU count ({gpu_count}) and the model's key/value head count "
                f"({kv_heads}). Valid choices here: "
                f"{[t for t in range(1, min(gpu_count, kv_heads) + 1) if gpu_count % t == 0 and kv_heads % t == 0]}"
            )
        return requested

    for size in range(min(gpu_count, kv_heads), 0, -1):
        if gpu_count % size == 0 and kv_heads % size == 0:
            return size
    return 1


def check_capacity(model_id: str, gpu_count: int, full_finetune: bool,
                   label: str = "training") -> None:
    """Exit with guidance if the model clearly cannot fit on the visible GPUs."""
    params = model_param_count(model_id)
    available = total_gpu_memory_bytes()
    if not params or not available:
        print(f"[preflight] skipped: could not size model '{model_id}' or GPU memory")
        return

    per_param = BYTES_PER_PARAM_FULL_FINETUNE if full_finetune else BYTES_PER_PARAM_PEFT
    needed = params * per_param
    usable = available * USABLE_MEMORY_FRACTION
    mode = "full fine-tune" if full_finetune else "parameter-efficient (LoRA)"
    gib = 1024 ** 3

    print(f"[preflight] {label}: {params/1e9:.1f}B params, {mode}, "
          f"~{needed/gib:.0f} GiB needed vs ~{usable/gib:.0f} GiB usable "
          f"across {gpu_count} GPU(s)")

    if needed > usable:
        sys.exit(
            f"\n[preflight] This model does not fit on the available GPUs.\n"
            f"  model            : {model_id} ({params/1e9:.1f}B parameters)\n"
            f"  mode             : {mode} (~{per_param} bytes/parameter)\n"
            f"  estimated need   : {needed/gib:.0f} GiB\n"
            f"  usable GPU memory: {usable/gib:.0f} GiB ({gpu_count} GPU(s), "
            f"{USABLE_MEMORY_FRACTION:.0%} of {available/gib:.0f} GiB)\n\n"
            f"Options: use a smaller model, switch to LoRA instead of a full fine-tune, "
            f"reduce max_seq_length or batch size, or run on more/larger GPUs.\n"
            f"This is an estimate of weights, gradients and optimizer state; it excludes "
            f"activations, so a run that passes this check can still need tuning.\n"
        )

    if needed > 0.7 * usable:
        print(f"[preflight] warning: estimated need is {needed/usable:.0%} of usable "
              f"memory. Activations are not included, so this may still OOM. Consider "
              f"activation checkpointing, a shorter sequence length or a smaller batch.")
