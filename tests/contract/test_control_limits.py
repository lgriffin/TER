"""Contract suite for the ``ControlLimitsSource`` port (point 201).

The JSON file adapter and the in-memory fake run the same assertions, so the
fake cannot drift from the obligations the real adapter meets.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ter.adapters.driven.control_limits import JsonControlLimits
from ter.adapters.driven.in_memory import InMemoryControlLimits
from ter.domain.lean.control import (
    ControlLimits,
    ControlLimitsError,
    SessionMeasures,
    Tuning,
    compute_limits,
)
from ter.ports import ControlLimitsSource


def _limits() -> ControlLimits:
    rows = [
        SessionMeasures(f"s{i}", None, {"wip_peak": float(i % 4)}, "fp")
        for i in range(10)
    ]
    limits = compute_limits(rows, computed_on="2026-10-10")
    [wip] = limits.measures
    from dataclasses import replace

    return replace(limits, measures=(replace(wip, tuning=Tuning(2.0, None, "x")),))


Factory = Callable[[Path], tuple[ControlLimitsSource, str]]


def _json(tmp: Path) -> tuple[ControlLimitsSource, str]:
    path = tmp / "limits.json"
    path.write_text(json.dumps(_limits().as_dict()), encoding="utf-8")
    return JsonControlLimits(), str(path)


def _memory(tmp: Path) -> tuple[ControlLimitsSource, str]:
    return InMemoryControlLimits({"limits": _limits()}), "limits"


@pytest.fixture(params=[_json, _memory], ids=["json", "in-memory"])
def source(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[ControlLimitsSource, str]:
    factory: Factory = request.param
    return factory(tmp_path)


@pytest.mark.req("TER-SPC-004")
def test_satisfies_the_port_protocol(source: tuple[ControlLimitsSource, str]) -> None:
    port, _ = source
    assert isinstance(port, ControlLimitsSource) and port.name


@pytest.mark.req("TER-SPC-004")
def test_limits_are_deterministic_and_keep_tuning(
    source: tuple[ControlLimitsSource, str],
) -> None:
    port, ref = source
    first = port.limits(ref)
    assert first == port.limits(ref)
    assert first.as_dict() == _limits().as_dict()
    wip = first.get("wip_peak")
    assert wip is not None and wip.tuning is not None and wip.ucl == 2.0


@pytest.mark.req("TER-SPC-005")
def test_unknown_ref_raises(source: tuple[ControlLimitsSource, str]) -> None:
    port, _ = source
    with pytest.raises(ControlLimitsError):
        port.limits("no-such-limits.json")


@pytest.mark.req("TER-SPC-005")
@pytest.mark.parametrize(
    "text",
    ["not json", "[]", json.dumps({"schema": "ter.control-limits/1"})],
)
def test_json_adapter_rejects_invalid_documents(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ControlLimitsError, match="bad.json"):
        JsonControlLimits().limits(path)
