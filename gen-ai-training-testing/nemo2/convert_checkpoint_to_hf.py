"""Export a Megatron-Bridge LoRA checkpoint as a Hugging Face model directory.

Rewritten for NeMo Framework 26.x. The previous version used
`nemo.collections.llm.peft.merge_lora` / `export_lora`, which disappeared when NeMo
25.11 split the framework and LLM training moved to Megatron-Bridge.

This delegates to the exporters that ship inside the container rather than
reimplementing the conversion:

  * merged weights   -> /opt/Megatron-Bridge/examples/peft/merge_lora.py
  * adapter only     -> /opt/Megatron-Bridge/examples/conversion/adapter/export_adapter.py

Those are NVIDIA's own tested code paths, they understand Megatron-Bridge's
`LoRALinear` wrapper for every architecture `AutoBridge` supports, and merge_lora
can run entirely on CPU. Wrapping them keeps this script thin and keeps the output
naming (`.hf_model`, `.hf_peft`) that the rest of this example expects.

Typical use, after training:

    python convert_checkpoint_to_hf.py \
        --megatron_checkpoint ./outputs/qwen3_8b/checkpoints \
        --base_model /snapshots/huggingface/Qwen/Qwen3-8B

then evaluate the result:

    python test_checkpoint.py --model_dir ./outputs/qwen3_8b/checkpoints.hf_model \
        --test_path ./data/alpaca/test.jsonl

Note: merging needs a LoRA checkpoint produced by training, so unlike
dataset_module.py and test_checkpoint.py this script cannot be exercised on a GPU
that cannot run the training step (see require_hopper_or_newer in peft_megatron.py).
"""
import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, fields, MISSING
from pathlib import Path

MERGE_SCRIPT = "/opt/Megatron-Bridge/examples/peft/merge_lora.py"
ADAPTER_SCRIPT = "/opt/Megatron-Bridge/examples/conversion/adapter/export_adapter.py"


@dataclass
class ExportConfig:
    """Export configuration."""

    # Directory holding the LoRA fine-tuning checkpoint. If it contains iter_* /
    # latest_train_state.pt subdirectories the exporters pick the right one.
    megatron_checkpoint: str = None

    # Hugging Face model id or local snapshot supplying the architecture config.
    base_model: str = "Qwen/Qwen3-8B"

    # Base dense checkpoint. Optional: merge_lora reads it from run_config.yaml in the
    # LoRA checkpoint when omitted.
    pretrained_checkpoint: str = None

    output_dir: str = None

    # Export the adapter on its own instead of merging it into the base weights.
    no_merge: bool = False

    # Merge on CPU. Slower, but needs no GPU and avoids parallelism bookkeeping.
    cpu: bool = False

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1

    trust_remote_code: bool = False
    debug: bool = False

    @property
    def resolved_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        suffix = ".hf_peft" if self.no_merge else ".hf_model"
        return str(Path(self.megatron_checkpoint).with_suffix('')) + suffix

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ExportConfig':
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


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Pick the newest iter_* subdirectory, or the directory itself.

    Megatron-Bridge writes iter_XXXXXXX subdirectories under the save directory. Both
    exporters accept either the parent or a specific iteration, so this only has to
    skip directories we produced ourselves on a previous run.
    """
    path = Path(checkpoint_dir)
    iters = [d for d in path.glob("iter_*") if d.is_dir()]
    if iters:
        latest = max(iters, key=lambda d: d.name)
        print(f"Selected checkpoint: {latest}")
        return str(latest)
    return checkpoint_dir


def build_command(config: ExportConfig) -> list:
    out = config.resolved_output_dir
    checkpoint = find_latest_checkpoint(config.megatron_checkpoint)

    if config.no_merge:
        if not os.path.isfile(ADAPTER_SCRIPT):
            sys.exit(f"Adapter exporter not found at {ADAPTER_SCRIPT}. This script "
                     f"expects to run inside the NeMo Framework container.")
        cmd = [sys.executable, ADAPTER_SCRIPT, "--output", out,
               "--tp", str(config.tensor_parallel_size),
               "--pp", str(config.pipeline_parallel_size),
               "--ep", str(config.expert_parallel_size)]
        if config.trust_remote_code:
            cmd.append("--trust-remote-code")
        return cmd

    if not os.path.isfile(MERGE_SCRIPT):
        sys.exit(f"Merge exporter not found at {MERGE_SCRIPT}. This script expects to "
                 f"run inside the NeMo Framework container.")
    cmd = [sys.executable, MERGE_SCRIPT,
           "--lora-checkpoint", checkpoint,
           "--hf-model-path", config.base_model,
           "--output", out,
           "--tp", str(config.tensor_parallel_size),
           "--pp", str(config.pipeline_parallel_size),
           "--ep", str(config.expert_parallel_size)]
    if config.pretrained_checkpoint:
        cmd += ["--pretrained", config.pretrained_checkpoint]
    if config.cpu:
        cmd.append("--cpu")
    if config.debug:
        cmd.append("--debug")
    return cmd


def export_to_hf(config: ExportConfig) -> str:
    """Run the appropriate exporter and return the output directory."""
    out = config.resolved_output_dir
    os.makedirs(out, exist_ok=True)

    cmd = build_command(config)
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Export failed with exit code {result.returncode}. Re-run with "
                 f"--debug for verbose logging.")

    print("=" * 80)
    print(f"Export complete: {out}")
    print(f"Evaluate it with: python test_checkpoint.py --model_dir {out} "
          f"--test_path <test.jsonl>")
    print("=" * 80)
    return out


def main() -> None:
    parser = create_parser_from_dataclass(ExportConfig)
    args = parser.parse_args()
    config = ExportConfig.from_args(args)

    if not config.megatron_checkpoint or not os.path.isdir(config.megatron_checkpoint):
        raise SystemExit(
            f"--megatron_checkpoint must be the directory holding the LoRA checkpoint "
            f"written by training; got '{config.megatron_checkpoint}'."
        )

    export_to_hf(config)


if __name__ == "__main__":
    main()
