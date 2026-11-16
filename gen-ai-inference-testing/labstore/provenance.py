#!/usr/bin/env python3
"""Provenance for accelerator compile caches -- hardware-agnostic.

A compiled artifact is only reusable on a machine whose *relevant* environment
matches. On Trainium that artifact is a NEFF produced through TorchNeuron's
torch.compile backend; on CUDA it may be an inductor cache or a TensorRT-LLM
engine. Both invalidate on things the filename never captures: SDK version,
compiler version, accelerator model, parallelism degree.

So cache entries are keyed by a *stack hash* over the facts that actually
matter, and every entry carries a manifest. On restore we diff the manifest
against the live machine and report WHICH FIELD changed rather than silently
recompiling -- a silent miss means a full recompile and looks like a mystery.

Design rules:
  * Weights and results are hardware-independent, so they are keyed WITHOUT
    accelerator fields and shared between GPU and Trainium machines.
  * Compile caches are keyed WITH them, so a Trainium NEFF can never collide
    with a CUDA engine.
  * Detection never raises: an undetectable field becomes "unknown" and is
    recorded as such, because a wrong key is worse than a coarse one.

Standalone: no third-party imports, so it runs on a bare instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


def _run(cmd: list[str], timeout: int = 20) -> str:
    """Best-effort command output. Never raises; missing tools are normal."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return ""
    except ImportError:
        return ""


# --------------------------------------------------------------------------
# accelerator detection
# --------------------------------------------------------------------------

def detect_accel_class() -> str:
    """'neuron', 'cuda', or 'cpu'. Device nodes first -- they cannot lie."""
    if any(n.startswith("neuron") for n in os.listdir("/dev")) if os.path.isdir("/dev") else False:
        return "neuron"
    if os.path.exists("/dev/nvidiactl") or shutil.which("nvidia-smi"):
        # nvidia-smi can exist without a working GPU; require it to answer.
        if _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]):
            return "cuda"
    return "cpu"


def detect_neuron() -> dict[str, Any]:
    """Trainium topology. nc_count is LOGICAL cores per device at the active
    LNC setting -- verified on trn2.48xlarge, where it reported 4
    (not the 8 physical), with logical-neuroncore-config: 2."""
    info: dict[str, Any] = {}
    ls = _run(["/opt/aws/neuron/bin/neuron-ls"]) or _run(["neuron-ls"])
    if ls:
        m = re.search(r"instance-type:\s*(\S+)", ls)
        if m:
            info["instance_type"] = m.group(1)
        m = re.search(r"logical-neuroncore-config:\s*(\d+)", ls)
        if m:
            info["lnc"] = int(m.group(1))
    js = _run(["/opt/aws/neuron/bin/neuron-ls", "--json-output"]) or _run(["neuron-ls", "--json-output"])
    if js:
        try:
            data = json.loads(js)
            if isinstance(data, list) and data:
                info["device_count"] = len(data)
                nc = data[0].get("nc_count")
                if nc:
                    info["logical_cores_per_device"] = nc
                    info["logical_cores_total"] = nc * len(data)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    if "device_count" not in info and os.path.isdir("/dev"):
        n = len([x for x in os.listdir("/dev") if re.match(r"^neuron\d+$", x)])
        if n:
            info["device_count"] = n
    # Model name: derive from instance type rather than guessing silicon.
    it = info.get("instance_type", "")
    info["accel_model"] = (
        "trainium2" if it.startswith("trn2") else
        "trainium1" if it.startswith("trn1") else
        "inferentia2" if it.startswith("inf2") else "unknown"
    )
    # Driver + runtime, which gate NEFF compatibility.
    for pkg in ("aws-neuronx-dkms", "aws-neuronx-runtime-lib", "aws-neuronx-collectives", "aws-neuronx-tools"):
        v = _run(["dpkg-query", "-W", "-f=${Version}", pkg])
        if v:
            info[pkg] = v
    return info


def detect_cuda() -> dict[str, Any]:
    info: dict[str, Any] = {}
    names = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if names:
        lines = [l.strip() for l in names.splitlines() if l.strip()]
        info["device_count"] = len(lines)
        if lines:
            first = lines[0].split(",")[0].strip()
            info["accel_model"] = re.sub(r"[^a-z0-9]+", "-", first.lower()).strip("-")
            info["device_name"] = first
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if drv:
        info["driver_version"] = drv.splitlines()[0].strip()
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        m = re.search(r"release (\d+\.\d+)", nvcc)
        if m:
            info["cuda_version"] = m.group(1)
    info.setdefault("accel_model", "unknown")
    return info


def detect_frameworks(accel_class: str) -> dict[str, str]:
    """Only the packages that actually affect the compiled artifact."""
    common = ["torch", "vllm", "transformers"]
    extra = {
        "neuron": ["libtorch-neuronx-lite", "torch-neuronx", "neuronx-cc", "nki",
                   "vllm-neuron", "neuronx-distributed-inference"],
        "cuda": ["tensorrt-llm", "tensorrt", "flash-attn", "triton", "xformers"],
        "cpu": [],
    }[accel_class]
    out = {}
    for name in common + extra:
        v = _pkg_version(name)
        if v:
            out[name] = v
    return out


# --------------------------------------------------------------------------
# provenance record
# --------------------------------------------------------------------------

@dataclass
class Provenance:
    schema_version: int = SCHEMA_VERSION
    accel_class: str = "cpu"
    accel_model: str = "unknown"
    accel: dict[str, Any] = field(default_factory=dict)
    frameworks: dict[str, str] = field(default_factory=dict)
    python: str = ""
    os_release: str = ""
    # Workload shape -- supplied by the caller, not detected.
    model_id: str = ""
    model_revision: str = ""
    topology: str = ""
    extra_keys: dict[str, str] = field(default_factory=dict)

    def stack_fields(self) -> dict[str, Any]:
        """Exactly the fields the stack hash is computed over.

        Deliberately EXCLUDES device_count: the compiled artifact depends on
        the parallelism degree (carried in `topology`), not on how many
        devices the host happens to expose. Including it would cause spurious
        misses when the same TP8 config runs on a differently-sized host.
        """
        accel = {k: v for k, v in self.accel.items()
                 if k not in ("device_count", "logical_cores_total")}
        return {
            "schema_version": self.schema_version,
            "accel_class": self.accel_class,
            "accel_model": self.accel_model,
            "accel": accel,
            "frameworks": self.frameworks,
            "python": self.python,
            "topology": self.topology,
            "extra_keys": self.extra_keys,
        }

    def stack_hash(self, length: int = 12) -> str:
        blob = json.dumps(self.stack_fields(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:length]

    def model_slug(self) -> str:
        base = self.model_id or "no-model"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-").lower()
        if self.model_revision and self.model_revision not in ("main", ""):
            slug += f"@{self.model_revision[:12]}"
        return slug

    def cache_key(self) -> str:
        """Key for a HARDWARE-DEPENDENT artifact (compile cache)."""
        return (f"cache/{self.model_slug()}/"
                f"{self.accel_class}-{self.accel_model}/"
                f"{self.stack_hash()}/{self.topology or 'default'}")

    def weights_key(self) -> str:
        """Key for a HARDWARE-INDEPENDENT artifact. No accelerator fields, so
        a GPU box and a Trainium box share one weight cache."""
        return f"models/{self.model_slug()}"

    def diff(self, other: dict[str, Any]) -> list[str]:
        """Human-readable reasons this provenance differs from a manifest."""
        reasons: list[str] = []
        mine = self.stack_fields()
        theirs = other.get("stack_fields", other)

        def walk(a: Any, b: Any, path: str) -> None:
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
            elif a != b:
                reasons.append(f"{path}: cached={b!r} live={a!r}")

        walk(mine, theirs, "")
        return reasons


def collect(model_id: str = "", model_revision: str = "", topology: str = "",
            extra_keys: dict[str, str] | None = None) -> Provenance:
    ac = detect_accel_class()
    accel = {"neuron": detect_neuron, "cuda": detect_cuda, "cpu": dict}[ac]()
    p = Provenance(
        accel_class=ac,
        accel_model=accel.pop("accel_model", "unknown") if accel else "unknown",
        accel=accel,
        frameworks=detect_frameworks(ac),
        python=platform.python_version(),
        os_release=platform.platform(),
        model_id=model_id,
        model_revision=model_revision,
        topology=topology,
        extra_keys=extra_keys or {},
    )
    # Default topology from what we can see, so callers may omit it.
    if not p.topology:
        if ac == "neuron" and p.accel.get("lnc"):
            p.topology = f"lnc{p.accel['lnc']}"
        elif ac == "cuda" and p.accel.get("device_count"):
            p.topology = f"gpu{p.accel['device_count']}"
    return p


def manifest(p: Provenance) -> dict[str, Any]:
    import datetime
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stack_hash": p.stack_hash(),
        "cache_key": p.cache_key(),
        "weights_key": p.weights_key(),
        "stack_fields": p.stack_fields(),
        "full": asdict(p),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID", ""))
    ap.add_argument("--model-revision", default=os.environ.get("MODEL_REVISION", ""))
    ap.add_argument("--topology", default=os.environ.get("TOPOLOGY", ""),
                    help="e.g. tp8-lnc2 (neuron) or tp8 (cuda)")
    ap.add_argument("--json", action="store_true", help="emit the full manifest")
    ap.add_argument("--key", choices=["cache", "weights", "hash"],
                    help="print just one value, for shell use")
    args = ap.parse_args()

    p = collect(args.model_id, args.model_revision, args.topology)

    if args.key == "cache":
        print(p.cache_key()); return
    if args.key == "weights":
        print(p.weights_key()); return
    if args.key == "hash":
        print(p.stack_hash()); return
    if args.json:
        print(json.dumps(manifest(p), indent=2, sort_keys=True)); return

    print(f"accelerator : {p.accel_class} / {p.accel_model}")
    for k, v in sorted(p.accel.items()):
        print(f"  {k:28}: {v}")
    print(f"topology    : {p.topology or '(unset)'}")
    print(f"python      : {p.python}")
    print("frameworks  :" + (" (none detected)" if not p.frameworks else ""))
    for k, v in sorted(p.frameworks.items()):
        print(f"  {k:28}: {v}")
    print(f"stack hash  : {p.stack_hash()}")
    print(f"cache key   : {p.cache_key()}")
    print(f"weights key : {p.weights_key()}")


if __name__ == "__main__":
    main()
