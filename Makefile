VENV := .venv/bin/python
LIMIT ?= 0
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

eval:
	$(VENV) -m evals.run_eval --mode replay --system-version $(SYSTEM_VERSION) --limit $(LIMIT) --out $(OUT)

eval-live:
	$(VENV) -m evals.run_eval --mode live --system-version $(SYSTEM_VERSION) $(if $(PROMPT_VERSION),--prompt-version $(PROMPT_VERSION)) --limit $(LIMIT) --out $(OUT)

ci: test
	$(VENV) -m evals.gate
