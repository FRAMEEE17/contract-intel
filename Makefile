VENV := .venv/bin/python
# 0 = all gold items; override for a quick smoke, e.g. `make eval-live LIMIT=5`
LIMIT ?= 0
# Each arm gets its own cassette tag; PROMPT_VERSION empty = production default.
SYSTEM_VERSION ?= v1
PROMPT_VERSION ?=
OUT ?= evals/report.json

.PHONY: help test eval eval-live ci

help:
	@echo "test                                 run the offline unit + integration suite"
	@echo "eval [SYSTEM_VERSION=v LIMIT=N]      replay committed cassettes -> deterministic, offline, free (CI gate)"
	@echo "eval-live [SYSTEM_VERSION=v PROMPT_VERSION=v LIMIT=N OUT=path]"
	@echo "                                     regenerate answers vs live models, then judge (model endpoint + GROQ_API_KEY)"
	@echo "ci                                   full local CI mirror: tests, then the offline eval-gate"

test:
	$(VENV) -m pytest -q

# Reproduce the number offline from committed system + judge cassettes.
eval:
	$(VENV) -m evals.run_eval --mode replay --system-version $(SYSTEM_VERSION) --limit $(LIMIT) --out $(OUT)

# Record a fresh run: generate answers vs the configured LLM (dev: local MLX; prod: Azure OpenAI),
# grounded in the full contract texts, then judge with the 3-model Groq jury.
# Writes system-answer cassettes (tagged SYSTEM_VERSION) so `make eval` can replay it offline afterwards.
eval-live:
	$(VENV) -m evals.run_eval --mode live --system-version $(SYSTEM_VERSION) $(if $(PROMPT_VERSION),--prompt-version $(PROMPT_VERSION)) --limit $(LIMIT) --out $(OUT)

# Full local CI mirror: unit/integration tests, then the offline eval-gate.
ci: test
	$(VENV) -m evals.gate
