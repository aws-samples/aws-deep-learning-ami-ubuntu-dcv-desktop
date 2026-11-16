#!/usr/bin/env python3
"""Fold `vllm bench serve` JSON results into an SLO table and plots.

Produces, in --results-dir (or --out-dir):
  slo_table.csv        one row per (input, output, concurrency) cell
  slo_knee.md          latency floor, knee per metric, proposed SLO
  slo_feasibility.md   only with --slo-* targets: max load inside each target
  slo_curves.png       p99 TTFT and p99 ITL against output throughput

Two views of every cell (see docs/neuron/README.md section 4.1):
  raw      what `vllm bench serve` itself reports
  steady   recomputed from --save-detailed per-request data after dropping the
           first and last N requests (N = concurrency) by start time. In a
           closed loop with --request-rate inf, the first N arrive together
           and Neuron prefills them one at a time, so the ramp sets the raw
           p90/p99 TTFT; the drain runs below the target concurrency.
SLOs are derived from the STEADY view by default (--basis raw to override).

The knee is found separately for p99 TTFT and p99 ITL: the last concurrency
where the relative gain in throughput still exceeds the relative rise in that
latency. Whichever knee comes at the lower concurrency is the binding metric.

Every reported number carries its full context: input/output tokens,
concurrency, replicas, dtype (BF16) and topology (TP8/LNC2).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

FNAME_RE = re.compile(r"vllmbench_in(\d+)_out(\d+)_c(\d+)\.json$")
# Run context (model, dtype, TP, instance, stack, replica slots, KV admission cap).
# bench/client.sh records it as server-context.json next to the results; nothing
# about a particular model or instance is hardcoded here.
CONTEXT = "context unknown (no server-context.json)"
REPLICA_SLOTS: int | None = None
MODEL_NAME = "model unknown"


def load_context(results_dir: Path) -> dict:
    """Read server-context.json and set the module-level context strings."""
    global CONTEXT, REPLICA_SLOTS, MODEL_NAME
    path = results_dir / "server-context.json"
    try:
        ctx = json.loads(path.read_text())
    except (OSError, ValueError):
        print(f"warn: no readable {path}; numbers will carry no hardware context", file=sys.stderr)
        return {}
    MODEL_NAME = ctx.get("model") or MODEL_NAME
    REPLICA_SLOTS = ctx.get("replica_slots")
    parts = [ctx.get("dtype"), f"TP{ctx['tp']}" if ctx.get("tp") else None,
             ctx.get("instance_type"), f"{ctx['neuron_stack']} stack" if ctx.get("neuron_stack") else None]
    CONTEXT = ", ".join(p for p in parts if p) or CONTEXT
    return ctx

# (csv column, key in the vllm bench serve 0.29 result JSON)
RAW_FIELDS: list[tuple[str, str]] = [
    ("duration_s", "duration"),
    ("completed", "completed"),
    ("failed", "failed"),
    ("req_throughput", "request_throughput"),
    ("out_throughput", "output_throughput"),
    ("total_throughput", "total_token_throughput"),
    ("ttft_p50_ms", "median_ttft_ms"),
    ("ttft_p90_ms", "p90_ttft_ms"),
    ("ttft_p99_ms", "p99_ttft_ms"),
    ("itl_p50_ms", "median_itl_ms"),
    ("itl_p99_ms", "p99_itl_ms"),
    ("tpot_p50_ms", "median_tpot_ms"),
    ("tpot_p99_ms", "p99_tpot_ms"),
    ("e2el_p50_ms", "median_e2el_ms"),
    ("e2el_p99_ms", "p99_e2el_ms"),
]
SS_FIELDS = ["ss_requests", "ss_out_throughput", "ss_ttft_p50_ms", "ss_ttft_p90_ms",
             "ss_ttft_p99_ms", "ss_itl_p50_ms", "ss_itl_p99_ms", "ss_tpot_p50_ms",
             "ss_tpot_p99_ms", "ss_e2el_p99_ms", "ss_stall_frac",
             "ss_maxgap_p50_ms", "ss_maxgap_p99_ms"]
NAN = math.nan


def pct(values: list[float], p: float) -> float:
    """Linear interpolation, matching numpy.percentile (what vllm bench uses)."""
    if not values:
        return NAN
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def steady_state(data: dict, conc: int) -> dict:
    """Recompute metrics over the closed loop's steady-state window."""
    starts, ttfts, itls = data.get("start_times"), data.get("ttfts"), data.get("itls")
    outs, errs = data.get("output_lens"), data.get("errors") or []
    if not (starts and ttfts and itls and outs):
        return {k: NAN for k in SS_FIELDS}
    reqs = []
    for i, st in enumerate(starts):
        if (errs[i] if i < len(errs) else "") or not ttfts[i]:
            continue
        gaps = itls[i] or []
        reqs.append({"start": st, "ttft": ttfts[i], "itls": gaps, "out": outs[i],
                     "end": st + ttfts[i] + sum(gaps)})
    reqs.sort(key=lambda r: r["start"])
    window = reqs[conc:-conc] if len(reqs) > 2 * conc else []
    if not window:
        return {k: NAN for k in SS_FIELDS}
    span = max(r["end"] for r in window) - min(r["start"] for r in window)
    all_itl = [g * 1e3 for r in window for g in r["itls"]]
    tpots = [sum(r["itls"]) * 1e3 / (r["out"] - 1) for r in window if r["out"] > 1]
    itl_med = pct(all_itl, 50)
    return {
        "ss_requests": len(window),
        # Tokens of requests that START in the window over the window's span.
        # Close to, not identical to, vllm's definition (all tokens / wall).
        "ss_out_throughput": sum(r["out"] for r in window) / span if span > 0 else NAN,
        "ss_ttft_p50_ms": pct([r["ttft"] * 1e3 for r in window], 50),
        "ss_ttft_p90_ms": pct([r["ttft"] * 1e3 for r in window], 90),
        "ss_ttft_p99_ms": pct([r["ttft"] * 1e3 for r in window], 99),
        "ss_itl_p50_ms": itl_med,
        "ss_itl_p99_ms": pct(all_itl, 99),
        "ss_tpot_p50_ms": pct(tpots, 50),
        "ss_tpot_p99_ms": pct(tpots, 99),
        "ss_e2el_p99_ms": pct([(r["end"] - r["start"]) * 1e3 for r in window], 99),
        # Longest single token gap per request, summarised across requests.
        # Phase-locked closed loops put one long multi-prefill stall in each
        # stream, which is under 1% of gaps (invisible to p99 ITL) but is a
        # visible freeze of the stream (docs/neuron/README.md section 4.1).
        "ss_maxgap_p50_ms": pct([max(r["itls"]) * 1e3 for r in window if r["itls"]], 50),
        "ss_maxgap_p99_ms": pct([max(r["itls"]) * 1e3 for r in window if r["itls"]], 99),
        # Share of token gaps > 3x the median gap: decode steps frozen behind
        # another request's prefill (docs/neuron/README.md section 4.1).
        "ss_stall_frac": (sum(1 for g in all_itl if g > 3 * itl_med) / len(all_itl)
                          if all_itl and itl_med == itl_med else NAN),
    }


def server_sidecar(path: Path) -> dict:
    """Prefix-cache hit rate and preemptions during the cell, from the
    <cell>.server.json counter snapshots run_vllm_bench.sh writes."""
    side = path.with_name(path.stem + ".server.json")
    try:
        d = json.loads(side.read_text())
        b, a = d["before"], d["after"]
        q = a["prefix_queries"] - b["prefix_queries"]
        return {"prefix_hit_rate": (a["prefix_hits"] - b["prefix_hits"]) / q if q > 0 else 0.0,
                "preemptions": a["preemptions"] - b["preemptions"], "seed": d.get("seed", NAN)}
    except (OSError, KeyError, ValueError, TypeError):
        return {"prefix_hit_rate": NAN, "preemptions": NAN, "seed": NAN}


def load(results_dir: Path) -> list[dict]:
    rows = []
    files = sorted(p for p in results_dir.glob("vllmbench_*.json")
                   if not p.name.endswith(".server.json"))
    if not files:
        sys.exit(f"No vllmbench_*.json found in {results_dir}")
    for path in files:
        m = FNAME_RE.search(path.name)
        if not m:
            print(f"skip (unparseable name): {path.name}", file=sys.stderr)
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip (unreadable): {path.name}: {exc}", file=sys.stderr)
            continue
        conc = int(m.group(3))
        row = {"input_len": int(m.group(1)), "output_len": int(m.group(2)), "concurrency": conc}
        missing = [key for _, key in RAW_FIELDS if key not in data]
        if missing:
            print(f"warn: {path.name} lacks {missing} (vllm version drift?)", file=sys.stderr)
        for col, key in RAW_FIELDS:
            v = data.get(key)
            row[col] = v if v is not None else NAN
        row.update(steady_state(data, conc))
        row.update(server_sidecar(path))
        if row["prefix_hit_rate"] == row["prefix_hit_rate"] and row["prefix_hit_rate"] > 0.01:
            print(f"WARN: {path.name}: prefix-cache hit rate {row['prefix_hit_rate']:.0%} -- prompts were "
                  "reused; on Neuron that changes TTFT (docs/neuron/README.md section 4.4). Cell is CONTAMINATED.", file=sys.stderr)
        if row["preemptions"] == row["preemptions"] and row["preemptions"] > 0:
            print(f"WARN: {path.name}: {row['preemptions']:.0f} preemptions during the cell.", file=sys.stderr)
        rows.append(row)
    rows.sort(key=lambda r: (r["input_len"], r["output_len"], r["concurrency"]))
    # Queueing share of TTFT: only concurrency 1 is compute-only (docs/neuron/README.md section 4.1).
    for r in rows:
        base = next((b for b in rows if b["input_len"] == r["input_len"]
                     and b["output_len"] == r["output_len"] and b["concurrency"] == 1), None)
        t, t1 = r.get("ss_ttft_p50_ms", NAN), (base or {}).get("ss_ttft_p50_ms", NAN)
        r["ttft_queue_share"] = (t - t1) / t if base and t == t and t1 == t1 and t > 0 else NAN
    return rows


def col(basis: str, name: str) -> str:
    """Map a metric name (e.g. 'ttft_p99_ms') to its raw or steady column.
    maxgap_* exists only in the steady view (vllm bench does not report it)."""
    return f"ss_{name}" if basis == "steady" or name.startswith("maxgap") else name


def usable(r: dict, *cols: str) -> bool:
    return all(isinstance(r.get(c), (int, float)) and r[c] == r[c] for c in cols)


def find_knee(srows: list[dict], lat_col: str, tput_col: str) -> dict | None:
    """Last point whose relative throughput gain still exceeds the relative
    latency rise. Returns None if fewer than two usable points."""
    pts = [r for r in srows if usable(r, lat_col, tput_col)]
    if len(pts) < 2:
        return None
    knee = pts[0]
    for prev, cur in zip(pts, pts[1:]):
        if prev[tput_col] <= 0 or prev[lat_col] <= 0:
            continue
        gain = (cur[tput_col] - prev[tput_col]) / prev[tput_col]
        rise = (cur[lat_col] - prev[lat_col]) / prev[lat_col]
        if gain <= 0 or rise > gain:
            break
        knee = cur
    return knee


def label(r: dict, replicas: int) -> str:
    return (f"{r['input_len']} in / {r['output_len']} out, concurrency {r['concurrency']}, "
            f"{replicas} replica(s), {CONTEXT}")


def f1(v) -> str:
    return "n/a" if not isinstance(v, (int, float)) or v != v else f"{v:,.1f}"


def write_csv(rows: list[dict], path: Path, replicas: int) -> None:
    cols = (["input_len", "output_len", "concurrency", "replicas", "dtype", "topology"]
            + [c for c, _ in RAW_FIELDS] + SS_FIELDS
            + ["ttft_queue_share", "prefix_hit_rate", "preemptions", "seed"])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "replicas": replicas, "dtype": "bfloat16", "topology": "TP8/LNC2"})
    print(f"wrote {path}")


def write_knee(rows: list[dict], path: Path, replicas: int,
               basis: str, admission_cap: int | None) -> None:
    tput = col(basis, "out_throughput")
    shapes = sorted({(r["input_len"], r["output_len"]) for r in rows})
    lines = [
        "# Derived SLO recommendation", "",
        f"- Context for every number: {CONTEXT}, {replicas} replica(s) serving concurrently.",
        f"- Basis: **{basis}** ({'ramp/drain trimmed, docs/neuron/README.md section 4.1' if basis == 'steady' else 'as reported by vllm bench serve'}).",
        "- Concurrency = closed-loop clients per replica (`--max-concurrency`, `--request-rate inf`).",
    ]
    if admission_cap:
        lines.append(f"- Scheduler admission cap (worst-case KV): **{admission_cap}** running requests. "
                     "Cells above it are marked OVER CAP: the excess requests queue and do not run concurrently.")
    lines.append("")
    for in_len, out_len in shapes:
        srows = [r for r in rows if r["input_len"] == in_len and r["output_len"] == out_len]
        lines += [
            f"## {in_len} in / {out_len} out ({replicas} replica(s), {CONTEXT})", "",
            "| conc | p50 TTFT | p99 TTFT | raw p99 TTFT | TTFT queue share | p50 ITL | p99 ITL | stall frac "
            "| worst gap/req p50 | worst gap/req p99 | p50 TPOT | out tok/s | n |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in srows:
            cap = " OVER CAP" if admission_cap and r["concurrency"] > admission_cap else ""
            ph = r.get("prefix_hit_rate", NAN)
            if ph == ph and ph > 0.01:
                cap += f" CONTAMINATED({ph:.0%} prefix hits)"
            qs = r.get("ttft_queue_share", NAN)
            sf = r.get("ss_stall_frac", NAN)
            lines.append(
                f"| {r['concurrency']}{cap} | {f1(r[col(basis, 'ttft_p50_ms')])} | {f1(r[col(basis, 'ttft_p99_ms')])} "
                f"| {f1(r['ttft_p99_ms'])} | {'n/a' if qs != qs else f'{qs:.0%}'} "
                f"| {f1(r[col(basis, 'itl_p50_ms')])} | {f1(r[col(basis, 'itl_p99_ms')])} "
                f"| {'n/a' if sf != sf else f'{sf:.1%}'} | {f1(r.get('ss_maxgap_p50_ms'))} "
                f"| {f1(r.get('ss_maxgap_p99_ms'))} | {f1(r[col(basis, 'tpot_p50_ms')])} "
                f"| {f1(r[tput])} | {r.get('ss_requests') if basis == 'steady' else r['completed']} |")
        lines += ["", "Latencies in ms. `worst gap/req` = each request's longest token gap (a stream "
                  "freeze behind other requests' prefills), as p50/p99 over requests. It is the stall a user "
                  "sees and p99 ITL hides it once stalls are < 1% of gaps (docs/neuron/README.md section 4.1).", ""]

        floor = next((r for r in srows if r["concurrency"] == 1), None)
        if floor:
            lines.append(
                f"- **Latency floor** ({label(floor, replicas)}): "
                f"TTFT p50 {f1(floor[col(basis, 'ttft_p50_ms')])} / p99 {f1(floor[col(basis, 'ttft_p99_ms')])} ms, "
                f"ITL p50 {f1(floor[col(basis, 'itl_p50_ms')])} / p99 {f1(floor[col(basis, 'itl_p99_ms')])} ms. "
                "No SLO on this configuration can beat it; replicas do not lower it.")
        else:
            lines.append("- **Latency floor:** not measured (no concurrency-1 cell).")

        in_cap = [r for r in srows if not (admission_cap and r["concurrency"] > admission_cap)
                  and not (r.get("prefix_hit_rate", 0) > 0.01)]
        k_ttft = find_knee(in_cap, col(basis, "ttft_p99_ms"), tput)
        k_itl = find_knee(in_cap, col(basis, "itl_p99_ms"), tput)
        last_c = max((r["concurrency"] for r in in_cap), default=None)
        for name, k in (("p99 TTFT", k_ttft), ("p99 ITL", k_itl)):
            if not k:
                continue
            if k["concurrency"] == last_c and len(in_cap) > 1:
                ceiling = (f"the admission cap ({admission_cap})" if admission_cap
                           else "the highest concurrency tested")
                lines.append(f"- **Knee by {name}: not reached.** Throughput still outgrew {name} at "
                             f"concurrency {last_c}, the last usable in-cap point; the knee lies between it "
                             f"and {ceiling}. Reported points at {last_c} are a lower bound on capacity.")
            else:
                lines.append(f"- **Knee by {name}:** concurrency {k['concurrency']} "
                             f"({f1(k[tput])} out tok/s per replica).")
        knees = [(n, k) for n, k in (("p99 TTFT", k_ttft), ("p99 ITL", k_itl)) if k]
        if knees:
            bind_name, knee = min(knees, key=lambda nk: nk[1]["concurrency"])
            ties = [n for n, k in knees if k["concurrency"] == knee["concurrency"]]
            lines.append(f"- **Binds first:** {' and '.join(ties)} "
                         f"(knee at concurrency {knee['concurrency']}).")
            if k_ttft and k_ttft["concurrency"] == min(r["concurrency"] for r in in_cap):
                lines.append("  On Neuron the TTFT knee sits at the lowest concurrency by construction: prefill "
                             "is serialized, so TTFT grows about in proportion to concurrency while throughput "
                             "grows less (docs/neuron/README.md section 4.1). Use target feasibility (`--slo-ttft-p99`) for TTFT "
                             "SLOs and the ITL knee for capacity.")

            def slo_line(tag: str, k: dict) -> str:
                return (f"- **{tag}:** p99 TTFT <= {k[col(basis, 'ttft_p99_ms')] * 1.1:,.0f} ms and "
                        f"p99 ITL <= {k[col(basis, 'itl_p99_ms')] * 1.1:,.0f} ms "
                        f"(+10% headroom) at {label(k, replicas)}; {f1(k[tput])} out tok/s per replica.")
            lines.append(slo_line("Proposed SLO, latency-optimal (first knee)", knee))
            if k_itl and k_itl is not knee:
                tag = ("Proposed SLO, capacity point (ITL knee)" if k_itl["concurrency"] != last_c
                       else f"Proposed SLO, capacity point (highest in-cap point, ITL knee not reached)")
                lines.append(slo_line(tag, k_itl))
        else:
            lines.append("- Knee not determinable: fewer than two usable in-cap points.")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def check_slo(rows: list[dict], targets: dict[str, float], basis: str,
              admission_cap: int | None) -> list[dict]:
    """Given SLO targets, the most load each shape carries while meeting all."""
    tput = col(basis, "out_throughput")
    out = []
    for in_len, out_len in sorted({(r["input_len"], r["output_len"]) for r in rows}):
        srows = [r for r in rows if r["input_len"] == in_len and r["output_len"] == out_len
                 and not (admission_cap and r["concurrency"] > admission_cap)
                 and not (r.get("prefix_hit_rate", 0) > 0.01)]
        passing, fails = [], []
        for r in srows:
            v = [f"{m}={r[col(basis, m)]:,.1f} > {lim:,.1f}" for m, lim in targets.items()
                 if usable(r, col(basis, m)) and r[col(basis, m)] > lim]
            (fails if v else passing).append((r, v))
        best = max((r for r, _ in passing), key=lambda r: r[tput]) if passing else None
        c1 = next(((r, v) for r, v in fails if r["concurrency"] == 1), None)
        out.append({"input_len": in_len, "output_len": out_len, "best": best,
                    "first_fail": fails[0] if fails else None, "c1_fail": c1,
                    "n_passing": len(passing), "n_total": len(srows),
                    "has_c1": any(r["concurrency"] == 1 for r in srows)})
    return out


def write_feasibility(verdicts: list[dict], targets: dict[str, float], path: Path,
                      replicas: int, basis: str) -> None:
    tput = col(basis, "out_throughput")
    lines = ["# SLO feasibility report", "",
             f"Targets ({basis} basis): " + ", ".join(f"`{m}` <= {v:g} ms" for m, v in sorted(targets.items())),
             f"Context: {CONTEXT}, {replicas} replica(s).", "",
             "| in | out | max concurrency meeting all targets | out tok/s per replica | verdict |",
             "|---|---|---|---|---|"]
    notes = []
    for v in verdicts:
        shape = f"{v['input_len']} in / {v['output_len']} out"
        b = v["best"]
        if v["c1_fail"]:
            r, why = v["c1_fail"]
            lines.append(f"| {v['input_len']} | {v['output_len']} | — | — | **INFEASIBLE: below latency floor** |")
            notes.append(f"**{shape}: infeasible.** Concurrency 1 already misses ({'; '.join(why)}). "
                         "The target is below this configuration's floor. Replicas will NOT help: they add "
                         "throughput, not single-request speed. Relax the target, shorten the prompt, or "
                         "change TP/dtype.")
        elif b is None:
            lines.append(f"| {v['input_len']} | {v['output_len']} | — | — | **INFEASIBLE at every tested concurrency** |")
            notes.append(f"**{shape}:** fails at every tested concurrency"
                         + ("" if v["has_c1"] else " (concurrency 1 not measured, so the floor is unknown)") + ".")
        else:
            lines.append(f"| {v['input_len']} | {v['output_len']} | **{b['concurrency']}** | {f1(b[tput])} "
                         f"| meets ({v['n_passing']}/{v['n_total']} points) |")
            if v["first_fail"]:
                r, why = v["first_fail"]
                notes.append(f"**{shape}:** binds at concurrency {r['concurrency']} ({'; '.join(why)}). "
                             f"Max safe load is concurrency {b['concurrency']} per replica "
                             f"({label(b, replicas)}). Beyond that, add replicas, not concurrency.")
            else:
                notes.append(f"**{shape}:** met at every tested concurrency up to {b['concurrency']}; "
                             "the true ceiling is higher than the sweep.")
    path.write_text("\n".join(lines + [""] + [n + "\n" for n in notes]))
    print(f"wrote {path}")


def plot(rows: list[dict], path: Path, basis: str, replicas: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots", file=sys.stderr)
        return
    tput = col(basis, "out_throughput")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for in_len, out_len in sorted({(r["input_len"], r["output_len"]) for r in rows}):
        srows = [r for r in rows if r["input_len"] == in_len and r["output_len"] == out_len]
        for ax, m in ((axes[0], "ttft_p99_ms"), (axes[1], "itl_p99_ms")):
            pts = [r for r in srows if usable(r, tput, col(basis, m))]
            ax.plot([r[tput] for r in pts], [r[col(basis, m)] for r in pts], marker="o",
                    label=f"{in_len}/{out_len}")
            for r in pts:
                ax.annotate(f"c{r['concurrency']}", (r[tput], r[col(basis, m)]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[0].set(xlabel="output tok/s per replica", ylabel="p99 TTFT (ms)", title="p99 TTFT vs throughput")
    axes[1].set(xlabel="output tok/s per replica", ylabel="p99 ITL (ms)", title="p99 ITL vs throughput")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="in/out tokens", fontsize=8)
    fig.suptitle(f"{MODEL_NAME}, {CONTEXT}, {replicas} replica(s), {basis} basis")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("./results"))
    ap.add_argument("--out-dir", type=Path, default=None, help="default: --results-dir")
    ap.add_argument("--replicas", type=int, default=1,
                    help="TP8 replicas serving concurrently during the sweep")
    ap.add_argument("--basis", choices=("steady", "raw"), default="steady")
    ap.add_argument("--admission-cap", type=int, default=None,  # default: from server-context.json
                    help="Scheduler's worst-case-KV concurrency cap from the server log")
    ap.add_argument("--slo-ttft-p99", type=float, default=None, metavar="MS")
    ap.add_argument("--slo-itl-p99", type=float, default=None, metavar="MS")
    ap.add_argument("--slo-tpot-p99", type=float, default=None, metavar="MS")
    ap.add_argument("--slo-e2e-p99", type=float, default=None, metavar="MS")
    ap.add_argument("--slo-maxgap-p99", type=float, default=None, metavar="MS",
                    help="Target: p99 over requests of each request's longest token gap (stream freeze)")
    args = ap.parse_args()
    out = args.out_dir or args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    ctx = load_context(args.results_dir)
    if args.admission_cap is None and ctx.get("kv_admission_cap"):
        args.admission_cap = int(ctx["kv_admission_cap"])
        print(f"admission cap {args.admission_cap} (from the server's KV cache config)")
    rows = load(args.results_dir)
    print(f"loaded {len(rows)} result cells")
    write_csv(rows, out / "slo_table.csv", args.replicas)
    write_knee(rows, out / "slo_knee.md", args.replicas, args.basis,
               args.admission_cap)
    plot(rows, out / "slo_curves.png", args.basis, args.replicas)

    targets = {k: v for k, v in {"ttft_p99_ms": args.slo_ttft_p99, "itl_p99_ms": args.slo_itl_p99,
                                 "tpot_p99_ms": args.slo_tpot_p99, "e2el_p99_ms": args.slo_e2e_p99,
                                 "maxgap_p99_ms": args.slo_maxgap_p99,
                                 }.items() if v is not None}
    if targets:
        verdicts = check_slo(rows, targets, args.basis, args.admission_cap)
        write_feasibility(verdicts, targets, out / "slo_feasibility.md", args.replicas, args.basis)
        for v in verdicts:
            b = v["best"]
            shape = f"{v['input_len']}/{v['output_len']}"
            print(f"  {shape}: " + (f"max concurrency {b['concurrency']}" if b else "INFEASIBLE"))


if __name__ == "__main__":
    main()
