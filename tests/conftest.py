"""Test-suite compatibility for optional development dependencies."""

from __future__ import annotations
import importlib.util
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # noqa: ARG001
    """Ignore BDD step modules when pytest-bdd is unavailable."""
    return (
        importlib.util.find_spec("pytest_bdd") is None
        and "tests/features/steps" in collection_path.as_posix()
    )
