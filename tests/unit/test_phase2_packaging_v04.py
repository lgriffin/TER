"""Regression tests for v04 packaging and optional-integration behavior."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from ter_calculator import embedding_cache


def test_embedding_extra_error_is_actionable(monkeypatch):
    embedding_cache._MODEL_CACHE.clear()
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"\[embeddings\]"):
        embedding_cache.get_embedding_model("missing-test-model")


def test_project_metadata_declares_supported_python_and_extras():
    root = Path(__file__).resolve().parents[2]
    metadata = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in metadata
    assert "embeddings = [" in metadata
    assert (
        "sentence-transformers"
        not in metadata.split("[project.optional-dependencies]", 1)[0]
    )
    assert "fail_under = 90" in metadata


def test_constraint_files_and_ci_matrix_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "constraints" / "dev.txt").is_file()
    assert (root / "constraints" / "ci.txt").is_file()
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert '["3.11", "3.12", "3.13"]' in workflow
    assert ".[dev,embeddings]" in workflow
    assert "--cov-branch" in workflow
