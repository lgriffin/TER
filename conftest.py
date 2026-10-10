"""Repository-wide pytest configuration.

Registers the requirement-traceability plugin, which adds ``--req-trace``
(see docs/ter4/requirements.md).

Tests marked ``embeddings`` run TER 3's real sentence-transformers model.
Without the ``embeddings`` extra they are skipped with a reason that says how
to install it, so the documented ``.[dev]`` setup gives a green suite. Set
``TER_REQUIRE_EMBEDDINGS=1`` (CI does) to run them regardless, so a missing
model fails loudly instead of hiding behind a skip.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

pytest_plugins = ("ter.adapters.driving.pytest_req", "pytester")

_EMBEDDINGS_SKIP = pytest.mark.skip(
    reason=(
        "needs the embeddings extra: "
        "python -m pip install -c constraints/dev.txt -e '.[dev,embeddings]'"
    )
)


def _embeddings_installed() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get("TER_REQUIRE_EMBEDDINGS") == "1" or _embeddings_installed():
        return
    for item in items:
        if item.get_closest_marker("embeddings"):
            item.add_marker(_EMBEDDINGS_SKIP)
