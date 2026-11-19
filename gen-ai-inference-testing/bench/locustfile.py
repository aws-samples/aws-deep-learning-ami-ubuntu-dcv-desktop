#!/usr/bin/env python3
"""Locust cross-validation harness for an SLO sweep (any OpenAI-compatible vLLM server).

This is the SECOND opinion, not the source of truth -- `vllm bench serve` is.
Its job is to catch mis-instrumentation: if median TTFT and output throughput
from the two harnesses disagree by more than ~10%, one of them is wrong and
the numbers are not publishable. Treat disagreement as a blocker, not as
something to average away.

What is deliberately the SAME as `vllm bench serve --dataset-name random`
(so a disagreement can only come from measurement, not from workload):
  - endpoint /v1/completions, streamed, temperature 0, ignore_eos
  - exactly INPUT_TOKENS prompt tokens, sent as token ids
  - a DIFFERENT random prompt per request. The server runs with prefix
    caching on, so a repeated prompt would make every TTFT after the first a
    cache hit. (An earlier version of this file sent one fixed prompt.)
  - closed loop: USERS concurrent streams, a fixed request count
What is independently implemented (the part being cross-checked):
  - TTFT = wall time to the first SSE chunk carrying text
  - ITL  = gaps between successive text-carrying chunks
  - output tokens from the server's `usage` block, not a chunk count
  - throughput = total output tokens / (last end - first start)

Run SINGLE-PROCESS (no --master/--worker). Results are collected in this
process, so a distributed run would summarise only one shard. This is why it
is not driven by the repo's run_locust.sh (master/worker) + endpoint_user.py:
that harness is non-streaming, so it cannot measure TTFT or ITL at all, and
this file is a measurement cross-check rather than a load generator.

  bench/client.sh locust <run-name> 4096 256 8 80    # in=4096 out=256 users=8 requests=80

or directly:
  MODEL=<served model name> TOKENIZER=<checkpoint dir or HF id> INPUT_TOKENS=4096 OUTPUT_TOKENS=256 \
  NUM_REQUESTS=80 LOCUST_SUMMARY_JSON=./locust_in4096_out256_c8.json \
  locust -f bench/locustfile.py -H http://localhost:8080 --headless \
         -u 8 -r 8 -t 30m --only-summary

Environment:
  MODEL              served model name (required; bench/client.sh reads it from /v1/models)
  TOKENIZER          tokenizer path or HF id (required), used only to pick valid token ids
  INPUT_TOKENS       exact prompt length in tokens (default 4096)
  OUTPUT_TOKENS      max_tokens, with ignore_eos (default 256)
  NUM_REQUESTS       stop after this many completed requests (default 10*users)
  SEED               prompt RNG seed (default: current epoch second, recorded in the summary)
  LOCUST_SUMMARY_JSON  write the summary here

The summary reports raw percentiles and steady-state ones, which drop the
first and last USERS requests by start order (closed-loop ramp and drain).
This is the same trimming bench/parse_results.py applies (docs/neuron/README.md section 4.1).
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time

import gevent
from locust import HttpUser, constant, events, task
from locust.exception import StopUser
from locust.runners import MasterRunner, WorkerRunner

MODEL = os.environ.get("MODEL", "")
TOKENIZER = os.environ.get("TOKENIZER", "")
if not MODEL or not TOKENIZER:
    raise SystemExit("locustfile.py: MODEL and TOKENIZER must be set (bench/client.sh sets both)")
INPUT_TOKENS = int(os.environ.get("INPUT_TOKENS", "4096"))
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "256"))
# Default differs per run: a fixed seed resends the same prompts on every run,
# and with prefix caching on those become cache hits (docs/neuron/README.md section 4.4).
SEED = int(os.environ.get("SEED", str(int(time.time()))))
_NUM_REQUESTS_ENV = os.environ.get("NUM_REQUESTS")


def _valid_token_ids() -> range:
    """Ordinary (non-special) token ids, so a random prompt never contains a
    control token."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    special = set(tok.all_special_ids)
    return range(0, min(special) if special else tok.vocab_size)


_VOCAB = _valid_token_ids()
_issue = itertools.count()          # request index, in issue order
_records: list[dict] = []           # one per successful request
_target = {"n": None}               # resolved at test_start
_t0 = {"wall": None}


def _prompt(i: int) -> list[int]:
    rng = random.Random(SEED * 1_000_003 + i)
    return [rng.choice(_VOCAB) for _ in range(INPUT_TOKENS)]


@events.test_start.add_listener
def _on_start(environment, **_kw) -> None:
    if isinstance(environment.runner, (MasterRunner, WorkerRunner)):
        raise RuntimeError("locustfile.py must run single-process; see the docstring")
    users = environment.parsed_options.num_users if environment.parsed_options else 1
    _target["n"] = int(_NUM_REQUESTS_ENV) if _NUM_REQUESTS_ENV else 10 * users
    _target["users"] = users


class StreamUser(HttpUser):
    wait_time = constant(0)

    @task
    def completion_stream(self) -> None:
        i = next(_issue)
        if _target["n"] is not None and i >= _target["n"]:
            raise StopUser()  # enough issued; in-flight requests still finish
        payload = {
            "model": MODEL,
            "prompt": _prompt(i),
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        start = time.perf_counter()
        ttft = None
        last = start
        itls: list[float] = []
        usage_tokens = None

        with self.client.post("/v1/completions", json=payload, stream=True,
                              catch_response=True,
                              name=f"completions in{INPUT_TOKENS} out{OUTPUT_TOKENS}") as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage_tokens = obj["usage"].get("completion_tokens")
                    choices = obj.get("choices") or []
                    if not choices or not choices[0].get("text"):
                        continue
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    else:
                        itls.append(now - last)
                    last = now
            except Exception as exc:  # noqa: BLE001 - report, don't crash the run
                resp.failure(f"stream error: {exc}")
                return
            if ttft is None:
                resp.failure("no text chunks received")
                return
            resp.success()

        end = time.perf_counter()
        _records.append({"i": i, "start": start, "end": end, "ttft_ms": ttft * 1e3,
                         "itls_ms": [x * 1e3 for x in itls],
                         "out_tokens": usage_tokens if usage_tokens is not None else len(itls) + 1})
        if len(_records) >= _target["n"]:
            gevent.spawn_later(0.5, self.environment.runner.quit)


def _pct(values: list[float], p: float) -> float:
    """Linear-interpolated percentile -- the method numpy (and so vllm bench
    serve) uses, so the two harnesses do not differ on the statistic itself."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summarise(recs: list[dict]) -> dict:
    ttfts = [r["ttft_ms"] for r in recs]
    itls = [x for r in recs for x in r["itls_ms"]]
    span = max(r["end"] for r in recs) - min(r["start"] for r in recs)
    out = sum(r["out_tokens"] for r in recs)
    return {
        "requests": len(recs),
        "median_ttft_ms": round(_pct(ttfts, 50), 2),
        "p90_ttft_ms": round(_pct(ttfts, 90), 2),
        "p99_ttft_ms": round(_pct(ttfts, 99), 2),
        "median_itl_ms": round(_pct(itls, 50), 2),
        "p99_itl_ms": round(_pct(itls, 99), 2),
        "mean_output_tokens": round(out / len(recs), 1),
        "output_throughput": round(out / span, 2) if span > 0 else float("nan"),
        "duration_s": round(span, 2),
    }


@events.test_stop.add_listener
def report(environment, **_kwargs) -> None:
    if not _records:
        print("\n[locust] no successful streamed requests -- nothing to compare\n")
        return
    users = _target.get("users", 1)
    by_issue = sorted(_records, key=lambda r: r["i"])
    steady = by_issue[users:-users] if len(by_issue) > 2 * users else []
    summary = {
        "input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS, "users": users, "seed": SEED,
        "raw": _summarise(by_issue),
        "steady_state": _summarise(steady) if steady else None,
    }
    print("\n=== locust cross-validation summary ===")
    print(json.dumps(summary, indent=2))
    out = os.environ.get("LOCUST_SUMMARY_JSON")
    if out:
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {out}\n")
