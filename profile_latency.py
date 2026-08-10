"""Single-request latency breakdown for answer_question: where does the time go?

Wraps each injected port in a timing proxy (no change to answer_question), runs the
real pipeline over a real contract K times, and reports per-stage cost. Needs a live
model endpoint (dev: local MLX).  Run:  python profile_latency.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGE_OF = {  # port method -> reported stage
    "inspect_input": "guardrail_in", "inspect_output": "guardrail_out",
    "embed_query": "embed", "search": "search",
    "get": "prompt", "resolve_production": "prompt", "complete": "llm",
}


class Timed:
    """Transparent proxy that accumulates wall-time per called method into `sink`."""

    def __init__(self, inner, sink: dict):
        self._inner = inner
        self._sink = sink

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return attr(*args, **kwargs)
            finally:
                self._sink[name] = self._sink.get(name, 0.0) + (time.perf_counter() - start)
        return wrapped


def _percent_table(per_request: list[dict], totals: list[float]) -> None:
    stages = ["guardrail_in", "embed", "search", "prompt", "llm", "guardrail_out", "overhead"]
    print(f"\n  stage          mean_ms   p50_ms   p95_ms   %of_total")
    for stage in stages:
        vals = [r[stage] * 1000 for r in per_request]
        mean = statistics.mean(vals)
        share = 100 * statistics.mean(vals) / (statistics.mean(totals) * 1000)
        p50 = statistics.median(vals)
        p95 = sorted(vals)[max(0, int(0.95 * len(vals)) - 1)]
        print(f"  {stage:13s} {mean:8.1f} {p50:8.1f} {p95:8.1f}   {share:6.1f}%")
    print(f"  {'TOTAL':13s} {statistics.mean(totals)*1000:8.1f}")


def main() -> int:
    from app.config import build_pipeline
    from app.application.answer import answer_question
    from evals.generate import build_contract_index, load_contract_text
    from evals.run_eval import load_gold

    pipeline = build_pipeline()
    gold = load_gold()
    contract_id = gold[0]["contract_id"]
    questions = [g["question"] for g in gold if g["contract_id"] == contract_id][:5] or [gold[0]["question"]]

    index = build_contract_index(text=load_contract_text(contract_id), filename=contract_id,parser=pipeline.parser, chunker=pipeline.chunker,embedder=pipeline.embedder, embed_model=pipeline.embed_model)

    warmup, reps = 3, 20
    per_request, totals = [], []
    print(f"-> profiling answer_question  (model={pipeline.model}, warmup={warmup}, reps={reps})")
    for i in range(warmup + reps):
        sink: dict = {}
        question = questions[i % len(questions)]
        t0 = time.perf_counter()
        answer_question(
            question,
            llm=Timed(pipeline.llm, sink), embedder=Timed(pipeline.embedder, sink), # type: ignore
            search=Timed(index, sink), guardrail=Timed(pipeline.guardrail, sink), # type: ignore
            registry=Timed(pipeline.registry, sink), # pyright: ignore[reportArgumentType]
            embed_model=pipeline.embed_model, model=pipeline.model,
        )
        total = time.perf_counter() - t0
        if i < warmup:
            continue
        stages = {STAGE_OF.get(m, m): 0.0 for m in STAGE_OF}
        for method, secs in sink.items():
            stages[STAGE_OF.get(method, method)] = stages.get(STAGE_OF.get(method, method), 0.0) + secs # type: ignore
        stages["overhead"] = max(0.0, total - sum(sink.values()))
        per_request.append(stages)
        totals.append(total)

    _percent_table(per_request, totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
