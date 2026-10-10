"""The example A3 the README links to stays current and keeps its variety.

``docs/examples/a3/`` holds a synthetic session, the golden corpus's control
limits and the A3 that ``ter a3 --limits`` makes of them. A detector, scoring
or renderer change that moves the page fails here; regenerate on purpose with
``TER_UPDATE_GOLDEN=1 pytest tests/docs/test_example_a3.py`` (or
``python scripts/example_a3.py --screenshots``, which also retakes the PNGs).

The page is there to show what TER does, so the variety it shows is asserted
too: if a change would leave it with no uncertain finding or no firing
signal, the session needs another look, not just a new snapshot.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ter import bootstrap
from ter.adapters.driving.cli import main

ROOT = Path(__file__).resolve().parents[2]
UPDATE_ENV = "TER_UPDATE_GOLDEN"


@cache
def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "example_a3", ROOT / "scripts" / "example_a3.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, dict[str, Any]]:
    """The page and JSON ``ter a3`` makes of the example now."""
    script = _script()
    if os.environ.get(UPDATE_ENV) == "1":
        script.build()
    out = tmp_path_factory.mktemp("example-a3")
    html, data = out / "a3.html", out / "a3.json"
    code = main(
        script.a3_argv(html, data),
        bootstrap.cli_services(),
        stdout=io.StringIO(),
        stderr=(err := io.StringIO()),
    )
    assert code == 0
    assert "Warning" not in err.getvalue(), "the limits no longer fit the detectors"
    return html.read_text(encoding="utf-8"), json.loads(data.read_text("utf-8"))


def _stale(name: str) -> str:
    return (
        f"docs/examples/a3/{name} is stale: regenerate with "
        f"{UPDATE_ENV}=1 pytest tests/docs/test_example_a3.py and commit the diff"
    )


def test_limits_are_the_golden_corpus_limits() -> None:
    script = _script()
    assert script.LIMITS.read_bytes() == script.GOLDEN_LIMITS.read_bytes(), _stale(
        "limits.json"
    )


def test_committed_page_matches_ter_a3(built: tuple[str, dict[str, Any]]) -> None:
    html, data = built
    script = _script()
    assert script.HTML.read_text(encoding="utf-8") == html, _stale("a3.html")
    committed = json.loads(script.JSON.read_text(encoding="utf-8"))
    assert committed == data, _stale("a3.json")


def test_example_shows_the_variety_the_readme_promises(
    built: tuple[str, dict[str, Any]],
) -> None:
    _, a3 = built
    findings = a3["findings"]
    waste = [f for f in findings if f["kind"] == "waste"]
    assert any(f["uncertain"] for f in waste), "an uncertain waste finding"
    assert any(not f["uncertain"] for f in waste), "a confident waste finding"
    assert any(f["kind"] == "risk" for f in findings), "a risk to the outcome"
    assert len({f["detector"] for f in findings}) >= 8, "many detectors"
    assert len(a3["countermeasures"]) >= 5
    measures = a3["process_control"]["measures"]
    outside = [m for m in measures if m["signal"]]
    assert any(m["fires"] for m in outside), "a signal that fires"
    assert any(not m["fires"] for m in outside), "a size signal that never fires"
    assert any(not m["signal"] for m in measures), "measures inside their limits"
