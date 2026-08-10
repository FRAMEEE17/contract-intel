"""LLM jury — three diverse judges vote on one answer's verdict.

Implements the judging rule in evals/METRICS.md:
  - panel of 3 different model families (decorrelated errors)
  - majority 2-of-3 per verdict; disagreement is recorded
  - verdicts are cached so `make eval` reproduces the same number offline

The Groq API key (GROQ_API_KEY) is read from .env by the program at runtime
and is never logged or printed.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

CONTRACT_INTEL_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = CONTRACT_INTEL_DIR / ".env"
CACHE_DIR = Path(__file__).resolve().parent / "cassettes" / "judge"
load_dotenv(ENV_PATH)  # load .env at import so JUDGE_* / GROQ_API_KEY are visible below

# Judge provider — Groq free tier (OpenAI-compatible). Override without code edits.
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.groq.com/openai/v1")

# 3 DIFFERENT model families -> decorrelated errors (METRICS.md D2). Groq slugs change
# over time: verify at console.groq.com/docs/models and override via
# JUDGE_MODELS="slug1,slug2,slug3" in .env if any of these are retired.
_DEFAULT_JUDGES = (
    "llama-3.3-70b-versatile",  # Meta
    "openai/gpt-oss-120b",      # OpenAI (open-weight)
    "qwen/qwen3.6-27b",         # Qwen — NOTE partial correlation with the Qwen generator
)
JUDGES = tuple(m.strip() for m in os.environ.get("JUDGE_MODELS", "").split(",") if m.strip()) or _DEFAULT_JUDGES

# Verdict taxonomy — mirrors METRICS.md §2.
ALL_VERDICTS = ("correct", "incorrect", "abstained", "correct_abstain", "fabricated")
HALLUCINATION_VERDICTS = frozenset({"incorrect", "fabricated"})

RUBRIC = """You grade one answer about a single legal-contract clause.
Compare the SYSTEM ANSWER against the REFERENCE ANSWER for the QUESTION, then
return exactly one verdict:
- "correct"        : reference is a real answer AND the system answer matches its meaning.
- "incorrect"      : reference is a real answer BUT the system asserts a different/wrong answer.
- "abstained"      : reference is a real answer BUT the system says it is not specified / unknown.
- "correct_abstain": reference is "NOT SPECIFIED" AND the system also says not specified / unknown.
- "fabricated"     : reference is "NOT SPECIFIED" BUT the system invents a specific answer.
Reply with strict JSON only: {"verdict": "<one-of-the-five>", "reason": "<short>"}"""


@dataclass(frozen=True)
class JuryResult:
    per_judge: tuple[str | None, ...]   # one verdict per judge (None = judge failed)
    majority: str | None                # None when there is no 2-of-3 agreement
    is_hallucination: bool
    disagreement: bool


_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(f"GROQ_API_KEY not found (checked {ENV_PATH} and env)")
        _client = OpenAI(base_url=JUDGE_BASE_URL, api_key=api_key, timeout=90)
    return _client


def _grading_prompt(question: str, reference: str, answer: str) -> str:
    return (f"QUESTION: {question}\n"
            f"REFERENCE ANSWER: {reference}\n"
            f"SYSTEM ANSWER: {answer}\n\n{RUBRIC}")


def _extract_verdict(raw: str) -> str | None:
    """Pull the verdict out of a reply, robust to reasoning models.

    A greedy `\\{.*\\}` breaks on models that emit a <think>…</think> block with
    stray braces before the JSON. Instead we scan every balanced JSON object in the
    text and return the verdict of the LAST one that carries a valid verdict — the
    real answer comes after any reasoning.
    """
    decoder = json.JSONDecoder()
    verdict: str | None = None
    i = 0
    while True:
        start = raw.find("{", i)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            candidate = str(obj["verdict"]).strip().lower()
            if candidate in ALL_VERDICTS:
                verdict = candidate  # keep scanning; prefer the last valid verdict
        i = start + max(end, 1)
    return verdict


def _cache_file(model: str, question: str, reference: str, answer: str) -> Path:
    digest = hashlib.sha256("|".join([model, question, reference, answer]).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def _ask_judge(model: str, prompt: str) -> str | None:
    """One judge call, retried once on error. Returns None if it ultimately fails
    (rate limit, timeout, ...). The caller must NOT freeze a None as a verdict."""
    for _ in range(2):
        try:
            response = _client_singleton().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception:
            continue
    return None


def judge_one(model: str, question: str, reference: str, answer: str) -> str | None:
    cache = _cache_file(model, question, reference, answer)
    if cache.exists():
        return json.loads(cache.read_text())["verdict"]
    verdict = _extract_verdict(_ask_judge(model, _grading_prompt(question, reference, answer)) or "")
    if verdict is None:
        return None  # a failed/unparseable call is NOT frozen -> it retries on the next run
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"model": model, "verdict": verdict}, ensure_ascii=False))
    return verdict


def _majority_vote(verdicts: tuple[str | None, ...]) -> str | None:
    counts: dict[str, int] = {}
    for verdict in verdicts:
        if verdict:
            counts[verdict] = counts.get(verdict, 0) + 1
    if not counts:
        return None
    winner = max(counts, key=lambda verdict: counts[verdict])
    return winner if counts[winner] >= 2 else None


def judge(question: str, reference: str, answer: str) -> JuryResult:
    per_judge = tuple(judge_one(model, question, reference, answer) for model in JUDGES)
    majority = _majority_vote(per_judge)
    distinct = {verdict for verdict in per_judge if verdict}
    return JuryResult(
        per_judge=per_judge,
        majority=majority,
        is_hallucination=majority in HALLUCINATION_VERDICTS,
        disagreement=len(distinct) > 1,
    )


def _load_gold_item(contract_keyword: str, category: str) -> dict:
    gold_path = Path(__file__).resolve().parent / "gold" / "gold.jsonl"
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if contract_keyword.lower() in item["contract_id"].lower() and item["category"] == category:
            return item
    raise LookupError(f"gold item not found: {contract_keyword} / {category}")


def _selftest() -> None:
    """Feed known-good / known-bad answers so we can see the jury classifies correctly."""
    answered = _load_gold_item("Fuse", "Governing Law")       # has a real answer
    absent = _load_gold_item("Fuse", "Exclusivity")           # reference = NOT SPECIFIED

    cases = [
        ("answered / correct answer", answered, answered["ground_truth"], "correct"),
        ("answered / wrong answer", answered,
         "This agreement is governed by the laws of the State of California, USA.", "incorrect"),
        ("no-answer / correct abstain", absent,
         "The contract does not specify any exclusivity obligation.", "correct_abstain"),
        ("no-answer / fabricated", absent,
         "Yes, the distributor is granted exclusive rights in the territory.", "fabricated"),
    ]

    for label, item, answer, expected in cases:
        result = judge(item["question"], item["ground_truth"], answer)
        ok = "OK" if result.majority == expected else "??"
        print(f"\n[{label}]  expect={expected}  got={result.majority}  [{ok}]")
        print(f"   per-judge: {result.per_judge}")
        print(f"   hallucination={result.is_hallucination}  disagreement={result.disagreement}")


if __name__ == "__main__":
    _selftest()
