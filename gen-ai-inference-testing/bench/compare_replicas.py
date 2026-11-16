#!/usr/bin/env python3
"""Does the per-replica SLO survive instance-level contention?

Compares each replica's cells, measured while N replicas served load at the
same time, against the same cells from a single-replica baseline sweep.
Uses the steady-state view from parse_results.py.

  python3 bench/compare_replicas.py --baseline results/<r1-run> \
      --replica-dirs results/<run>-r2-rep0 results/<run>-r2-rep1 [--tolerance 0.10]

For each cell, the reported delta is the WORST replica's, against baseline:
latency metrics higher is worse, throughput lower is worse. A cell "holds"
if every metric stays within tolerance. Also reports instance aggregate
output throughput (sum over replicas) against N x baseline. That ratio is
the scaling efficiency, which feeds capacity planning and EKS bin-packing.

Exit status: 0 if every compared cell holds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_results  # noqa: E402
from parse_results import f1, load  # noqa: E402

# (column, higher_is_worse)
METRICS = [("ss_ttft_p50_ms", True), ("ss_ttft_p99_ms", True), ("ss_itl_p50_ms", True),
           ("ss_itl_p99_ms", True), ("ss_maxgap_p99_ms", True), ("ss_out_throughput", False)]


def key(r: dict) -> tuple[int, int, int]:
    return r["input_len"], r["output_len"], r["concurrency"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--replica-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    parse_results.load_context(args.baseline)
    base = {key(r): r for r in load(args.baseline)}
    reps = [{key(r): r for r in load(d)} for d in args.replica_dirs]
    n = len(reps)
    cells = sorted(set.intersection(*(set(r) for r in reps)) & set(base))
    if not cells:
        print("no cells common to baseline and every replica", file=sys.stderr)
        return 1

    head = ["| cell | " + " | ".join(m.replace("ss_", "") for m, _ in METRICS)
            + " | instance out tok/s (sum) | scaling eff. | verdict |",
            "|---|" + "---|" * (len(METRICS) + 3)]
    lines, broken = [], []
    for c in cells:
        b = base[c]
        worst = {}
        for m, hiw in METRICS:
            ds = []
            for rep in reps:
                v, bv = rep[c].get(m), b.get(m)
                if v is None or bv is None or v != v or bv != bv or bv == 0:
                    continue
                ds.append((v - bv) / bv)
            if ds:
                worst[m] = max(ds) if hiw else min(ds)
        bad = [m for m, hiw in METRICS if m in worst
               and ((worst[m] > args.tolerance) if hiw else (worst[m] < -args.tolerance))]
        agg = sum(rep[c]["ss_out_throughput"] for rep in reps)
        eff = agg / (n * b["ss_out_throughput"]) if b["ss_out_throughput"] else float("nan")
        verdict = "holds" if not bad else "**DEGRADES** (" + ", ".join(m.replace("ss_", "") for m in bad) + ")"
        if bad:
            broken.append(f"{c[0]}/{c[1]} c{c[2]}")
        lines.append(f"| {c[0]}/{c[1]} c{c[2]} | "
                     + " | ".join(f"{worst[m]:+.1%}" if m in worst else "n/a" for m, _ in METRICS)
                     + f" | {f1(agg)} | {eff:.0%} | {verdict} |")

    text = "\n".join([
        f"# Per-replica SLO under contention: {n} replicas vs 1",
        "",
        f"Context: {parse_results.CONTEXT}; concurrency is per replica; all {n} replicas loaded at the same time.",
        f"Delta = worst replica vs the single-replica baseline (steady state). Tolerance ±{args.tolerance:.0%}.",
        f"Baseline: `{args.baseline}`",
        "Replicas: " + ", ".join(f"`{d}`" for d in args.replica_dirs),
        "",
        *head, *lines, "",
        (f"**{len(broken)} cell(s) degrade:** " + ", ".join(broken)) if broken
        else f"All {len(cells)} cells hold within ±{args.tolerance:.0%} at {n} replicas.",
        "",
    ])
    print(text)
    if args.out:
        args.out.write_text(text)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
