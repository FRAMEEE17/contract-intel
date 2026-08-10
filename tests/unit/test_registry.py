"""Unit tests for app.adapters.registry.file_registry.FilePromptRegistry."""
from __future__ import annotations

import pytest

from app.adapters.registry.file_registry import FilePromptRegistry


def test_get_answer_v1_contains_placeholders():
    registry = FilePromptRegistry()
    text = registry.get("answer", "v1")
    assert "{context}" in text
    assert "{question}" in text


def test_get_answer_system_v1_returns_system_prompt_text():
    registry = FilePromptRegistry()
    text = registry.get("answer_system", "v1")
    assert "NOT SPECIFIED" in text
    assert "ONLY the provided context" in text


def test_resolve_production_answer():
    registry = FilePromptRegistry()
    assert registry.resolve_production("answer") == "v1"


def test_resolve_production_answer_system():
    registry = FilePromptRegistry()
    assert registry.resolve_production("answer_system") == "v1"


def test_get_missing_version_raises_lookup_error():
    registry = FilePromptRegistry()
    with pytest.raises(LookupError):
        registry.get("answer", "v999")


def test_resolve_production_missing_name_raises_lookup_error():
    registry = FilePromptRegistry()
    with pytest.raises(LookupError):
        registry.resolve_production("does_not_exist")


def test_registry_resolves_repo_prompts_dir_with_no_args():
    """Constructed with no args, it must find the repo's prompts/ dir regardless of CWD."""
    registry = FilePromptRegistry()
    text = registry.get("answer", "v1")
    assert isinstance(text, str)
    assert len(text) > 0
