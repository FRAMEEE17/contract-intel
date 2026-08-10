"""Eval runner: tie generation + jury + scoring into one metric report.

The honesty logic lives in evals/score.py (classify/aggregate); this module is
the thin orchestration around it:

  gold item -> answer_fn (RAG pipeline) -> AnswerResult
            -> classify(has_answer, result, judge)   # jury called ONLY if answered
            -> record ; aggregate(records) -> report

Both the answer_fn and the jury are INJECTED, so the loop is unit-tested offline
with scripted stand-ins. The live path (`make_answer_fn`) wires the real adapters;
running it needs a model endpoint (dev: local MLX) and the Groq jury.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from evals.generate import build_contract_index, load_contract_text
from evals.score import aggregate, classify

GOLD_PATH = Path(__file__).resolve().parent / "gold" / "gold.jsonl"
SYSTEM_CASSETTE_DIR = Path(__file__).resolve().parent / "cassettes" / "system"


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def score_items(items, *, answer_fn: Callable[[dict], object], judge_fn: Callable[[str, str, str], object]) -> dict:
    """Run the pipeline over gold items and produce {report, records}.

    `answer_fn(item) -> AnswerResult`. `judge_fn(question, reference, answer) -> JuryResult`;
    it is invoked (via classify) only for `answered` items, so the jury cost (and its
    laundering surface) is limited to items that produced a substantive answer.
    """
    records = []
    for item in items:
        result = answer_fn(item)
        # bind per-iteration values so the closure is not affected by loop rebinding
        judge = lambda q=item["question"], ref=item["ground_truth"], res=result: judge_fn(q, ref, res.answer)
        bucket, majority, per_judge = classify(
            has_answer=bool(item["has_answer"]), result=result, judge=judge,
        )
        disagreement = (len({v for v in per_judge if v}) > 1) if per_judge else None
        records.append({
            "item_id": item.get("id"), "has_answer": bool(item["has_answer"]),
            "bucket": bucket, "jury_majority": majority,
            "per_judge": per_judge, "disagreement": disagreement,
        })
    return {"report": aggregate(records), "records": records}


def make_answer_fn(pipeline, *, top_k: int = 8, prompt_version: Optional[str] = None) -> Callable[[dict], object]:
    """answer_fn that caches one index per contract, grounding answers in its full text.

    Full contract text (evals/gold/contracts/), not the gold snippet, so answers are
    grounded in the source (METRICS §0). `pipeline` is a config.Pipeline.
    """
    from app.application.answer import answer_question

    index_by_contract: dict = {}

    def index_for(item: dict):
        cid = item["contract_id"]
        if cid not in index_by_contract:
            index_by_contract[cid] = build_contract_index(
                text=load_contract_text(cid), filename=cid, parser=pipeline.parser,
                chunker=pipeline.chunker, embedder=pipeline.embedder, embed_model=pipeline.embed_model,
            )
        return index_by_contract[cid]

    def answer_fn(item: dict):
        return answer_question(
            item["question"], llm=pipeline.llm, embedder=pipeline.embedder, search=index_for(item),
            guardrail=pipeline.guardrail, registry=pipeline.registry,
            embed_model=pipeline.embed_model, model=pipeline.model, top_k=top_k,
            prompt_version=prompt_version,
        )

    return answer_fn


# system-answer cassettes: reproducible offline replay (METRICS §5)
@dataclass(frozen=True)
class ReplayResult:
    """Minimal AnswerResult stand-in reconstructed from a cassette.

    classify() only reads the four flags; the jury closure only reads `answer`.
    """
    answer: str
    abstained: bool
    malformed: bool
    blocked: bool
    no_context: bool


def _cassette_path(system_version: str, item_id: str) -> Path:
    key = hashlib.sha256(f"{system_version}|{item_id}".encode()).hexdigest()[:16]
    return SYSTEM_CASSETTE_DIR / f"{key}.json"


def save_system_answer(system_version: str, item: dict, result) -> None:
    path = _cassette_path(system_version, item["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "system_version": system_version, "item_id": item["id"], "answer": result.answer,
        "abstained": result.abstained, "malformed": result.malformed,
        "blocked": result.blocked, "no_context": result.no_context,
    }, ensure_ascii=False), encoding="utf-8")


def load_system_answer(system_version: str, item: dict) -> Optional[ReplayResult]:
    path = _cassette_path(system_version, item["id"])
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return ReplayResult(answer=d["answer"], abstained=d["abstained"], malformed=d["malformed"],
                        blocked=d["blocked"], no_context=d["no_context"])


def replay_answer_fn(system_version: str) -> Callable[[dict], ReplayResult]:
    """answer_fn that replays committed system cassettes offline."""
    def answer_fn(item: dict) -> ReplayResult:
        cached = load_system_answer(system_version, item)
        if cached is None:
            raise SystemExit(
                f"no system cassette for {item['id']!r} at system_version={system_version!r}; "
                f"run `make eval-live` first to record answers."
            )
        return cached
    return answer_fn


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def print_summary(report: dict) -> None:
    m1 = report["M1_hallucination_rate"]
    up = m1["conservative_upper"]
    print(f"\n  items={report['total_items']}  answerable={report['answerable']}  unanswerable={report['unanswerable']}")
    print(f"  M1 hallucination_rate = {_fmt(m1['value'])}  [{_fmt(m1['value'])}, {_fmt(up['value'])}]"
          f"   (incorrect={m1['incorrect']['count']}, fabricated={m1['fabricated']['count']})")
    print(f"     sensitivity any-judge = {_fmt(m1['sensitivity_any_judge']['value'])}")
    print(f"  M2 answer_correctness = {_fmt(report['M2_answer_correctness']['value'])}  (den={report['M2_answer_correctness']['denominator']})")
    m3 = report["M3_abstention"]
    print(f"  M3 correct_abstain = {_fmt(m3['correct_abstain_rate']['value'])}   false_abstention = {_fmt(m3['false_abstention_rate']['value'])}")
    print(f"  M4 malformed_rate = {_fmt(report['M4_malformed_rate']['value'])}")
    d = report["diagnostics"]
    print(f"  diagnostics: no_context={_fmt(d['no_context_rate'])} blocked={_fmt(d['blocked_rate'])} "
          f"indeterminate={_fmt(d['indeterminate_rate'])} jury_disagreement={_fmt(d['jury_disagreement_rate'])}")
    print(f"  audit: partition_ok={report['audit']['partition_ok']}\n")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Contract-RAG eval runner (METRICS.md).")
    ap.add_argument("--mode", choices=["replay", "live"], default="replay",
                    help="replay committed cassettes (offline, default) | live (regenerate vs models)")
    ap.add_argument("--system-version", default="v1", help="tags the system-answer cassettes")
    ap.add_argument("--prompt-version", default=None, help="pin the answer prompt version (live only); default = production")
    ap.add_argument("--limit", type=int, default=0, help="only the first N gold items (0 = all)")
    ap.add_argument("--out", default="", help="write the full report JSON here")
    args = ap.parse_args(argv)

    items = load_gold()
    if args.limit:
        items = items[:args.limit]

    from evals.jury import judge as jury_judge
    judge_fn = lambda q, ref, a: jury_judge(q, ref, a)  # jury.py caches verdicts (offline on replay)

    if args.mode == "live":
        from app.config import build_pipeline
        base = make_answer_fn(build_pipeline(), prompt_version=args.prompt_version)

        def answer_fn(item):
            result = base(item)
            save_system_answer(args.system_version, item, result)
            return result
    else:
        answer_fn = replay_answer_fn(args.system_version)

    out = score_items(items, answer_fn=answer_fn, judge_fn=judge_fn)
    report = out["report"]
    report["meta"] = {"mode": args.mode, "system_version": args.system_version, "n_items": len(items)}

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  report -> {args.out}")
    print_summary(report)
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `import app...`/`evals...` resolve
    raise SystemExit(main())
