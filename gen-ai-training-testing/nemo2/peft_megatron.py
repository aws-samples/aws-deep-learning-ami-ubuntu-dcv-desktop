"""PEFT (LoRA/DoRA) fine-tuning on Megatron, via NeMo Megatron-Bridge.

Rewritten for NeMo Framework 26.x. Releases from 25.11 onward split NeMo by domain:
the `nemo` package became NeMo Speech, and LLM training moved to Megatron-Bridge. The
NeMo 2.0 API this example previously used (`nemo.collections.llm`, `nemo.lightning`,
`nemo_run.Experiment`, recipe `finetune_recipe(...)`) no longer exists, and NVIDIA
publishes no migration guide, so this is a rewrite against the current API rather than
a port.

What changed for callers:
  * Launch with torchrun, e.g.
        torchrun --nproc_per_node=4 peft_megatron.py --hf_model_id /snapshots/.../Qwen3-8B
    The script no longer re-launches itself through a NeMo-Run executor.
  * --hf_model_id must be a LOCAL directory holding config.json plus safetensors.
    Megatron-Bridge accepts a Hugging Face snapshot directory directly as the
    pretrained checkpoint, so no separate import/convert step is needed, but a bare
    repo id is rejected because the path must exist on disk.
  * Flags tied to removed NeMo machinery are gone: --nemo_ckpt_dir, --node_rank,
    --early_stopping_*, the --enable_* profiler callbacks and --data_dir. Profiling,
    progress bars and callbacks are configured differently in Megatron-Bridge.

Hardware: the NeMo 26.x containers are built for Hopper and newer. This example
checks compute capability up front and exits with an explanation on older GPUs;
see require_hopper_or_newer() for what specifically fails and why it is not
configurable. The accelerate, ptl and ray_train examples have no such limit.

Dataset: the chosen recipe's default (SQuAD) is used unless --dataset_root points at a
directory of training/validation/test .jsonl files with prompt/completion columns.
"""
import argparse
import importlib
import os
import pkgutil
import sys
from dataclasses import dataclass, field, fields, MISSING

from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

from device_preflight import detected_gpu_count, select_tensor_parallel_size, check_capacity


@dataclass
class Config:
    """Fine-tuning configuration."""

    # Model. Must be a local snapshot directory: Megatron-Bridge asserts the
    # pretrained checkpoint path exists, so a bare "Qwen/Qwen3-8B" will not do.
    hf_model_id: str = "Qwen/Qwen3-8B"

    # Megatron-Bridge recipe family, e.g. "qwen3_8b" -> qwen3_8b_peft_config /
    # qwen3_8b_sft_config. Run with --list_recipes to see what is available.
    recipe_cls_name: str = "qwen3_8b"

    output_dir: str = None

    # Distributed. Detected rather than assumed; tensor_parallel_size 0 means auto.
    num_nodes: int = 1
    gpus_per_node: int = field(default_factory=detected_gpu_count)
    tensor_parallel_size: int = 0
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1

    # Training
    max_steps: int = 10000
    val_check_interval: int = 800
    log_every_n_steps: int = 10
    micro_batch_size: int = 8
    accumulate_grad_batches: int = 8
    limit_val_batches: int = 80
    max_seq_length: int = 2048

    # PEFT
    peft_scheme: str = "lora"
    full_ft: bool = False
    lora_dim: int = 0           # 0 keeps the recipe's value
    lora_alpha: int = 0         # 0 keeps the recipe's value

    # Optional local JSONL dataset directory (training/validation/test .jsonl)
    dataset_root: str = None

    # Gradient clipping threshold; 0.0 disables clipping. -1 keeps the recipe's value.
    clip_grad: float = -1.0

    use_wandb: bool = False

    @property
    def data_parallel_size(self) -> int:
        world = self.num_nodes * self.gpus_per_node
        divisor = self.tensor_parallel_size * self.pipeline_parallel_size * self.context_parallel_size
        dp = world // divisor
        if dp < 1:
            raise ValueError(
                f"No data-parallel degree left: {self.num_nodes} node(s) x "
                f"{self.gpus_per_node} GPU(s) / tensor_parallel_size="
                f"{self.tensor_parallel_size} / pipeline_parallel_size="
                f"{self.pipeline_parallel_size} / context_parallel_size="
                f"{self.context_parallel_size} gives {dp}. Lower one of them."
            )
        return dp

    @property
    def global_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulate_grad_batches * self.data_parallel_size

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create Config from argparse Namespace, only using provided args"""
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items()
                  if k in config_fields and v is not None}
        return cls(**kwargs)


def require_hopper_or_newer() -> None:
    """Exit early unless the GPUs can actually run this container's kernels.

    The NeMo 26.x containers are built for Hopper and newer. On earlier hardware
    training fails partway in, in two separate places, neither of which is reachable by
    configuration:

      * Transformer Engine has no THD (packed-sequence) attention backend below sm90, so
        the recipes' offline sequence packing leaves no usable attention backend at all
        ("No dot product attention backend is available for the provided inputs").
      * Megatron Core computes a gradient norm on every optimizer step through
        Transformer Engine's multi_tensor_apply, whose kernels are not built for sm89.
        That one fails with "CUDA Error: no kernel image is available for execution on
        the device" and cannot be skipped -- it happens even with clip_grad disabled,
        because the norm is computed unconditionally in MegatronOptimizer.step().

    Verified on 4x L40S (sm89): the model loads, LoRA attaches correctly (5,308,416 of
    8,196,043,776 parameters trainable), and the run then dies on the first optimizer
    step. Failing here instead is clearer than failing there.
    """
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception:
        print("[preflight] could not query GPU compute capability; continuing")
        return

    if major < 9:
        sys.exit(
            f"\n[preflight] This example needs a GPU of compute capability sm90 or newer "
            f"(Hopper: H100/H200, or Blackwell).\n"
            f"  detected          : {name} (sm{major}{minor})\n"
            f"  why               : the NeMo 26.x container's Transformer Engine build has "
            f"no sm{major}{minor} kernel for multi_tensor_apply, which Megatron Core calls "
            f"on every optimizer step to compute the gradient norm, and no THD attention "
            f"backend for packed sequences.\n"
            f"  not configurable  : the gradient norm is computed unconditionally in "
            f"MegatronOptimizer.step(), so disabling gradient clipping does not avoid it.\n\n"
            f"Options: run this example on sm90+ hardware, or use the accelerate, ptl or "
            f"ray_train examples in this repository, which train fine on this machine.\n"
        )
    print(f"[preflight] {name} (sm{major}{minor}) meets the sm90+ requirement")


def _recipe_modules():
    """Every submodule under megatron.bridge.recipes that may expose config factories."""
    import megatron.bridge.recipes as recipes
    for info in pkgutil.walk_packages(recipes.__path__, prefix="megatron.bridge.recipes."):
        try:
            yield importlib.import_module(info.name)
        except Exception:
            continue    # optional extras (vlm, diffusion) may not import on every image


def resolve_recipe(recipe_cls_name: str, full_ft: bool):
    """Find the Megatron-Bridge config factory for this recipe family.

    Replaces the NeMo 2.0 `import_module("nemo.collections.llm.recipes.<name>")`
    lookup. Megatron-Bridge names its factories "<family>_peft_config" for
    LoRA/DoRA and "<family>_sft_config" for a full fine-tune.
    """
    suffix = "_sft_config" if full_ft else "_peft_config"
    wanted = f"{recipe_cls_name}{suffix}"
    for module in _recipe_modules():
        fn = getattr(module, wanted, None)
        if callable(fn):
            return wanted, fn
    sys.exit(
        f"No Megatron-Bridge recipe '{wanted}' found. Run with --list_recipes to see "
        f"what this container provides."
    )


def list_recipes() -> None:
    names = set()
    for module in _recipe_modules():
        names.update(n for n in dir(module)
                     if n.endswith(("_peft_config", "_sft_config", "_pretrain_config")))
    for name in sorted(names):
        print(f"  {name}")


def build_config(config: Config):
    """Build and adjust the Megatron-Bridge ConfigContainer for this run."""
    recipe_name, recipe_fn = resolve_recipe(config.recipe_cls_name, config.full_ft)
    print(f"Using Megatron-Bridge recipe: {recipe_name}")

    # Only the PEFT factories accept peft_scheme; the SFT ones take no arguments.
    cfg = recipe_fn() if config.full_ft else recipe_fn(peft_scheme=config.peft_scheme)

    # A local HF snapshot directory is accepted directly as the pretrained checkpoint.
    cfg.checkpoint.pretrained_checkpoint = config.hf_model_id
    cfg.tokenizer.tokenizer_model = config.hf_model_id

    output_dir = config.output_dir or os.path.join(os.getcwd(), "outputs", config.recipe_cls_name)
    cfg.checkpoint.save = os.path.join(output_dir, "checkpoints")
    cfg.checkpoint.load = cfg.checkpoint.save      # recipes default to None, i.e. no resume
    cfg.logger.tensorboard_dir = os.path.join(output_dir, "tb_logs")

    cfg.model.tensor_model_parallel_size = config.tensor_parallel_size
    cfg.model.pipeline_model_parallel_size = config.pipeline_parallel_size
    cfg.model.context_parallel_size = config.context_parallel_size

    # model.seq_length and dataset.seq_length must agree: ConfigContainer asserts it
    cfg.model.seq_length = config.max_seq_length
    cfg.dataset.seq_length = config.max_seq_length

    cfg.train.train_iters = config.max_steps
    cfg.train.micro_batch_size = config.micro_batch_size
    cfg.train.global_batch_size = config.global_batch_size
    cfg.validation.eval_interval = config.val_check_interval
    cfg.validation.eval_iters = config.limit_val_batches
    cfg.logger.log_interval = config.log_every_n_steps
    if config.clip_grad >= 0:
        cfg.optimizer.clip_grad = config.clip_grad

    # Keep the LR schedule inside the (possibly shortened) run
    cfg.scheduler.lr_decay_iters = min(cfg.scheduler.lr_decay_iters or config.max_steps,
                                       config.max_steps)
    cfg.scheduler.lr_warmup_iters = min(cfg.scheduler.lr_warmup_iters, max(config.max_steps // 10, 1))

    if cfg.peft is not None:
        if config.lora_dim:
            cfg.peft.dim = config.lora_dim
        if config.lora_alpha:
            cfg.peft.alpha = config.lora_alpha
        print(f"PEFT: {type(cfg.peft).__name__} dim={cfg.peft.dim} alpha={cfg.peft.alpha} "
              f"target_modules={cfg.peft.target_modules}")


    if config.dataset_root:
        from megatron.bridge.data.builders import (GPTSFTDatasetConfig,
                                                   PromptCompletionSFTPreprocessingConfig)
        cfg.dataset = GPTSFTDatasetConfig(
            dataset_root=config.dataset_root,
            seq_length=config.max_seq_length,
            preprocessing=PromptCompletionSFTPreprocessingConfig(
                prompt_column="input", completion_column="output", separator=" ",
            ),
        )

    return cfg


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Build an argparse parser from the dataclass fields."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list_recipes", action="store_true",
                        help="list the recipe config factories this container provides")
    for f in fields(dataclass_type):
        default = None if f.default is MISSING else f.default
        if f.type is bool or isinstance(default, bool):
            parser.add_argument(f"--{f.name}", action="store_true", default=None)
        else:
            arg_type = int if f.type in (int, 'int') else (float if f.type in (float, 'float') else str)
            parser.add_argument(f"--{f.name}", type=arg_type, default=None)
    return parser


def main() -> None:
    parser = create_parser_from_dataclass(Config)
    args = parser.parse_args()

    if args.list_recipes:
        list_recipes()
        return

    config = Config.from_args(args)

    require_hopper_or_newer()

    if not os.path.isdir(config.hf_model_id):
        sys.exit(
            f"--hf_model_id must be a local directory containing config.json and model "
            f"weights; got '{config.hf_model_id}'. Megatron-Bridge accepts a Hugging Face "
            f"snapshot directory directly, but the path has to exist. Download it first, "
            f"e.g. with huggingface_hub.snapshot_download(..., local_dir=...)."
        )

    if config.tensor_parallel_size:
        # Explicit request: validate it divides both the GPU count and the KV heads
        config.tensor_parallel_size = select_tensor_parallel_size(
            config.gpus_per_node, config.hf_model_id,
            requested=config.tensor_parallel_size)
    else:
        # Default to pure data parallelism. Tensor parallelism exists to make a model
        # fit, not to go faster: it splits attention heads across ranks, which costs
        # communication every layer and shrinks the data-parallel degree. These recipes
        # are tuned for TP=1, and raising TP can also leave too few key/value heads per
        # rank for Transformer Engine to pick an attention backend. Raise it explicitly
        # only when a model will not otherwise fit.
        config.tensor_parallel_size = 1
    print(f"Using {config.gpus_per_node} GPU(s) per node, "
          f"tensor_parallel_size={config.tensor_parallel_size}, "
          f"data_parallel_size={config.data_parallel_size}, "
          f"global_batch_size={config.global_batch_size}")
    check_capacity(config.hf_model_id, config.gpus_per_node,
                   full_finetune=config.full_ft, label="nemo2 fine-tune")

    cfg = build_config(config)
    finetune(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
