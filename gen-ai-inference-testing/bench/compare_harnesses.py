#!/usr/bin/env python3
"""Cross-validate locustfile.py against vllm bench serve, cell by cell.

For every locust_in{I}_out{O}_c{C}.json in --results-dir with a matching
vllmbench_in{I}_out{O}_c{C}.json, compare median TTFT, output throughput and
median ITL on both the raw and steady-state views. Any |delta| > --tolerance
on median TTFT or throughput is a BLOCKER (PLAN.md Phase 5): one harness is
mis-instrumented, so do not average the two -- find out which is wrong.

Exit status: 0 if every compared cell agrees, 1 if any cell is a blocker,
2 if nothing could be compared.

Note: Locust's own "Response time" table measures to the response headers
when stream=True, so it is not a latency metric here. Use the summary JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_results import pct, steady_state  # noqa: E402

LOCUST_RE = re.compile(r"locust_in(\d+)_out(\d+)_c(\d+)\.json$")
GATED = ("median_ttft_ms", "output_throughput")       # the PLAN.md gate
INFO = ("median_itl_ms",)                              # reported, not gated


def vllm_views(data: dict, conc: int) -> dict:
    raw = {"median_ttft_ms": data.get("median_ttft_ms"),
           "output_throughput": data.get("output_throughput"),
           "median_itl_ms": data.get("median_itl_ms")}
    ss = steady_state(data, conc)
    steady = {"median_ttft_ms": ss["ss_ttft_p50_ms"],
              "output_throughput": ss["ss_out_throughput"],
              "median_itl_ms": ss["ss_itl_p50_ms"]}
    return {"raw": raw, "steady_state": steady}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=None, help="write markdown here")
    args = ap.parse_args()

    lines = ["| cell | view | metric | vllm bench | locust | delta | verdict |",
             "|---|---|---|---|---|---|---|"]
    compared, blockers = 0, []
    for lf in sorted(args.results_dir.glob("locust_*.json")):
        m = LOCUST_RE.search(lf.name)
        if not m:
            continue
        i, o, c = m.groups()
        vf = args.results_dir / f"vllmbench_in{i}_out{o}_c{c}.json"
        if not vf.exists():
            print(f"skip {lf.name}: no matching {vf.name}", file=sys.stderr)
            continue
        loc = json.loads(lf.read_text())
        ref = vllm_views(json.loads(vf.read_text()), int(c))
        cell = f"{i}/{o} c{c}"
        for view in ("raw", "steady_state"):
            if not loc.get(view):
                continue
            for metric in GATED + INFO:
                a, b = ref[view].get(metric), loc[view].get(metric)
                if not a or b is None or a != a:
                    continue
                d = (b - a) / a
                gated = metric in GATED
                bad = gated and abs(d) > args.tolerance
                verdict = "**BLOCKER**" if bad else ("ok" if gated else "info")
                if bad:
                    blockers.append(f"{cell} {view} {metric} {d:+.1%}")
                lines.append(f"| {cell} | {view} | {metric} | {a:,.2f} | {b:,.2f} | {d:+.1%} | {verdict} |")
        compared += 1
    if not compared:
        print("nothing to compare", file=sys.stderr)
        return 2
    summary = (f"\n{compared} cell(s) compared, tolerance ±{args.tolerance:.0%} on "
               f"{' and '.join(GATED)}. " +
               (f"**{len(blockers)} BLOCKER(S):** " + "; ".join(blockers) if blockers
                else "All gated metrics agree."))
    text = "# Harness cross-validation\n\n" + "\n".join(lines) + "\n" + summary + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
