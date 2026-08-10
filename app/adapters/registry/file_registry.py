"""File-backed prompt registry.

Layout: prompts/{name}/{version}.txt holds the prompt text; prompts/production.json
maps a prompt name to its current production version.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS_ROOT = REPO_ROOT / "prompts"


class FilePromptRegistry:
    def __init__(self, root: str | Path = DEFAULT_PROMPTS_ROOT):
        self._root = Path(root)

    def get(self, name: str, version: str) -> str:
        path = self._root / name / f"{version}.txt"
        if not path.is_file():
            raise LookupError(
                f"Prompt not found: name={name!r} version={version!r} (looked at {path})"
            )
        return path.read_text(encoding="utf-8").rstrip("\n")

    def resolve_production(self, name: str) -> str:
        path = self._root / "production.json"
        if not path.is_file():
            raise LookupError(f"Production manifest not found: {path}")
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if name not in mapping:
            raise LookupError(f"No production version registered for prompt {name!r}")
        return mapping[name]
