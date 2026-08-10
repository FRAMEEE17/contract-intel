# Metric Spec — Contract RAG Evaluation

Status: locked via grill (Q1–Q4, 2026-08-02). Every number must be **honest + reproducible**.

## 0. Honesty rules
- numbers from **real runs only**; ground vs the **source contract**, not a summary
- **same frozen `gold.jsonl` + same denominator** for baseline vs mitigated
- **malformed output = counted failure** (never silently dropped)
- report **grounding vs correctness separately**

## 1. Headline metric (Q1)
**"Hallucination rate" = rate of items where the system asserts a clause term that is NOT the contract's true term** — either:
- **incorrect** on an answerable question (contradicts/misses CUAD gold), or
- **fabricated** on an unanswerable question (invents an answer when gold = `NOT SPECIFIED`)

```
hallucination_rate = (incorrect_answerable + fabricated_unanswerable) / total_items
```

NOT headlined: pure RAGAS `faithfulness` (grounding-to-*retrieved-context*) — kept as a diagnostic.
Rationale (interview): *"faithful to what?"* — a system can be faithful to a wrong/summary context and still be wrong. We have CUAD gold, so we anchor on correctness + fabrication.

## 2. Per-item verdict
| gold | system does | verdict | hallucination? |
|---|---|---|---|
| has answer | correct (matches gold; span-level for lists e.g. Parties) | correct | no |
| has answer | wrong specific answer | incorrect | **yes** |
| has answer | says `NOT SPECIFIED` | false-abstention | no (reported separately) |
| no answer | says `NOT SPECIFIED` | correct-abstain | no |
| no answer | gives a specific answer | fabricated | **yes** (worst) |

## 3. Metrics reported
- **M1 hallucination_rate** (headline) + breakdown (incorrect vs fabricated)
- **M2 answer-correctness** (answerable only; **span-level** recall/precision for multi-span answers)
- **M3 abstention**: correct-abstain rate (unanswerable) + **false-abstention rate** (answerable)
- **M4 malformed-output rate** (counted as failure)
- **M5 diagnostics**: RAGAS faithfulness / answer-relevancy / context-precision; retrieval recall

## 4. Judge = 3-model diverse jury (Q2/Q3)
- Panel (OpenRouter, OpenAI-compatible SDK): `google/gemma-4-26b-a4b-it:free` + `openai/gpt-oss-20b:free` + `nvidia/nemotron-3-super-120b-a12b:free` — **3 different families → decorrelated errors** (cf. *Replacing Judges with Juries / PoLL, 2024*)
- Aggregation: **majority 2-of-3** per binary verdict; mean for scalar; **log disagreement rate**
- **Conservative "any-judge-flags"** reported as a **sensitivity** check (headline must not flip by rule)
- **Calibration:** ~20 human-labeled items; report **jury-vs-human agreement** — this is what makes the number defensible
- temp=0; verdicts **cached + committed** (reproducibility comes from the committed cache, not the free models)
- ⚠️ caveat: "GPT judges well" evidence is GPT-4-class; a 20B/26B is weaker → the **jury + calibration** carry the authority, not any single model

## 5. Reproducibility (Q4)
Two modes:
- **`make eval`** (default): replay from **committed cassettes** (system answers + judge verdicts) → deterministic, offline, free → used by CI gate + "reproduce my number live"
- **`make eval-live`**: regenerate vs live models → fresh number **±SD** → creates/refreshes the baseline

Official **X/Y = a live run, then frozen** (recorded with `run_id` + date + ±SD from a variance run). Re-running live lands within ±SD.

Frozen/committed:
- `gold.jsonl`
- **index = committed fixture** (Q4-A) — *plus* ingestion code+config committed so `rebuild → diff` proves the fixture isn't fabricated
- system cassettes · judge cassettes
- `manifest.json`: 3 judge slugs, judge prompt, RAGAS version, aggregation rule, seeds, gold hash, embedding model

## 6. X→Y protocol
- **X** (baseline) = live run on the current summary-grounded behavior
- **Y** (mitigated) = live run after source-grounding + fixes, **same gold, same jury**
- both committed (`run_id`s); report ±SD; **no number written before the runs exist**
- CI gate tolerance = **calibrated from the variance run** (measured SD), not asserted

## 7. Open TODO before the first number
1. build the **~20-item human calibration set** (jury-vs-human agreement)
2. run the **judge-variance measurement** → set the gate tolerance
3. acquire contract **full-texts** for retrieval + grounding checks (CUAD `full_contract_txt`)
