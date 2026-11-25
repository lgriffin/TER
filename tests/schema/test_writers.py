"""Every document TER writes validates against its JSON Schema (TER-TST-002),
and each schema rejects a document with a field missing or a field it does
not know (TER-TST-005), so a conforming result means something.

The documents are real output: the event records of every golden corpus
session and every GARE fixture run, the control measures and limits frozen
for the corpus, the shipped and an approved policy file, and the ledger a
shadow run over the corpus writes, with an observation attached.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from ter import bootstrap
from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.event_log import event_to_record
from ter.adapters.driven.gare import GareRunSource
from ter.adapters.driven.in_memory import FixedClock
from ter.adapters.driven.intervention_ledger import JsonlInterventionLedger
from ter.application.explain import ExplainedSession
from ter.application.shadow import shadow_sessions
from ter.domain.lean.control import ControlLimits
from ter.domain.ledger import Observation, Response
from ter.domain.policy_set import PolicySet

from golden.corpus import CORPUS, GOLDEN_DIR, REPO_ROOT

from .conftest import assert_conforms, errors

SNAPSHOTS = GOLDEN_DIR / "snapshots" / "control"
POLICIES = REPO_ROOT / "docs" / "examples" / "policies" / "l4-advisory.json"
GARE = REPO_ROOT / "tests" / "fixtures" / "gare"
AT = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
RECORDS = GOLDEN_DIR / "snapshots" / "record"
RECORD = "ter.record/0.1"
CALIBRATION = "ter.calibration/1"
SHIPPED_CALIBRATION = REPO_ROOT / "src" / "ter" / "data" / "calibration.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _gare_runs() -> list[Path]:
    mocks = [p for p in sorted(GARE.iterdir()) if p.is_dir() and p.name != "runs"]
    return mocks + [p for p in sorted((GARE / "runs").iterdir()) if p.is_dir()]


@cache
def _event_records() -> tuple[dict[str, Any], ...]:
    traces = [ClaudeCodeJsonlSource().read(str(p)) for p in CORPUS.values()]
    traces += [GareRunSource().read(str(p)) for p in _gare_runs()]
    return tuple(event_to_record(e) for t in traces for e in t.events)


def _approved() -> PolicySet:
    draft = PolicySet.from_mapping(_json(POLICIES))
    return draft.approve("Schema suite", date(2026, 10, 14), note="test only")


def _explain(path: Path) -> ExplainedSession:
    services = bootstrap.cli_services()
    assert services.explain_transcript is not None
    return services.explain_transcript(path, "regex", "off")


@pytest.fixture(scope="module")
def ledger_lines(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    folder = tmp_path_factory.mktemp("ledger")
    ledger = JsonlInterventionLedger(folder)
    limits = ControlLimits.from_mapping(_json(SNAPSHOTS / "corpus.limits.json"))
    run = shadow_sessions(
        sorted(CORPUS.values()), _explain, _approved(), ledger, FixedClock(AT), limits
    )
    advised = [r for r in run.records if r.delivered]
    assert advised and len(advised) < len(run.records), "want advice and suppression"
    first = advised[0]
    ledger.observe(
        first.session_id,
        first.id,
        Observation(Response.ACCEPTED, AT, 10, {"repeats": 2.0}, {"repeats": 0.0}),
    )
    lines = [
        json.loads(line)
        for path in sorted(folder.iterdir())
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert {line["type"] for line in lines} == {"record", "observation"}
    return lines


def _written(ledger_lines: list[dict[str, Any]]) -> Iterator[tuple[str, Any]]:
    """Every sample document, with the schema id it names."""
    for record in _event_records():
        yield record["schema"], record
    for name in ("corpus.limits.json", "corpus.measures.json"):
        doc = _json(SNAPSHOTS / name)
        yield doc["schema"], doc
    for doc in (_json(POLICIES), _approved().as_dict()):
        yield str(doc["schema"]), doc
    for line in ledger_lines:
        yield line["schema"], line
    for path in sorted(RECORDS.glob("*.record.json")):
        yield RECORD, _json(path)
    yield CALIBRATION, _json(SHIPPED_CALIBRATION)


@pytest.mark.req("TER-TST-002")
def test_event_records_cover_both_harnesses_tools_and_usage() -> None:
    assert len(_gare_runs()) >= 2
    assert {r["actor"] for r in _event_records()} >= {"user", "assistant", "tool"}
    assert any(r["tool"] for r in _event_records())
    assert any(r["usage"] for r in _event_records())


@pytest.mark.req("TER-TST-002")
def test_every_event_record_conforms() -> None:
    for record in _event_records():
        assert_conforms("ter.event/0.5", record)


@pytest.mark.req("TER-TST-002")
def test_control_documents_conform() -> None:
    assert_conforms("ter.control-limits/1", _json(SNAPSHOTS / "corpus.limits.json"))
    assert_conforms("ter.control-measures/1", _json(SNAPSHOTS / "corpus.measures.json"))


@pytest.mark.req("TER-TST-002")
def test_policy_files_conform_draft_and_approved() -> None:
    assert_conforms("ter.intervention-policies/1", _json(POLICIES))
    approved = _approved().as_dict()
    assert "approval" in approved
    assert_conforms("ter.intervention-policies/1", approved)


@pytest.mark.req("TER-TST-002", "TER-REC-001")
def test_every_golden_record_conforms() -> None:
    records = sorted(RECORDS.glob("*.record.json"))
    assert len(records) == len(CORPUS)
    for path in records:
        assert_conforms(RECORD, _json(path))


@pytest.mark.req("TER-TST-002", "TER-REC-004")
def test_shipped_calibration_conforms() -> None:
    assert_conforms(CALIBRATION, _json(SHIPPED_CALIBRATION))


@pytest.mark.req("TER-TST-002")
def test_every_ledger_line_conforms(ledger_lines: list[dict[str, Any]]) -> None:
    for line in ledger_lines:
        assert_conforms("ter.intervention-ledger/1", line)


# --- the schemas are not vacuous (TER-TST-005) -----------------------------


def _object_paths(
    doc: Any, path: tuple[str | int, ...] = ()
) -> Iterator[tuple[str | int, ...]]:
    if isinstance(doc, dict):
        yield path
        for key, value in doc.items():
            yield from _object_paths(value, (*path, key))
    elif isinstance(doc, list):
        for i, value in enumerate(doc[:2]):
            yield from _object_paths(value, (*path, i))


def _at(doc: Any, path: tuple[str | int, ...]) -> Any:
    for step in path:
        doc = doc[step]
    return doc


def _samples(ledger_lines: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """One sample per schema and per distinct top-level shape."""
    seen: set[tuple[str, frozenset[str]]] = set()
    picked: list[tuple[str, Any]] = []
    for schema_id, doc in _written(ledger_lines):
        shape = (schema_id, frozenset(doc))
        if shape not in seen:
            seen.add(shape)
            picked.append((schema_id, doc))
    return picked


#: Objects whose keys are data (a measure, a rule, a tool argument), not
#: fields: dropping one is allowed and an extra one may be fine.
_MAPS = {"arguments", "measures", "rules", "values", "before", "after", "detectors"}
#: Fields a writer leaves out when they do not apply.
_OPTIONAL = {"approval", "note", "delivered_at"}


@pytest.mark.req("TER-TST-005")
def test_dropping_any_field_or_adding_an_unknown_one_is_rejected(
    ledger_lines: list[dict[str, Any]],
) -> None:
    checked = 0
    for schema_id, doc in _samples(ledger_lines):
        for path in _object_paths(doc):
            if path and path[-1] in _MAPS:
                continue
            if any(step in _MAPS for step in path):
                continue
            target = _at(doc, path)
            for key in list(target):
                if key in _OPTIONAL:
                    continue
                broken = copy.deepcopy(doc)
                del _at(broken, path)[key]
                assert errors(schema_id, broken), (
                    f"{schema_id}: dropping {'/'.join(map(str, (*path, key)))} "
                    "still conforms"
                )
                checked += 1
            extra = copy.deepcopy(doc)
            _at(extra, path)["not_a_field"] = 1
            assert errors(schema_id, extra), (
                f"{schema_id}: an unknown field at /{'/'.join(map(str, path))} "
                "still conforms"
            )
            checked += 1
    assert checked > 100
