"""Session sources on hostile input (TER-TST-009).

A source either reads a file into a well-formed trace or refuses it with
:class:`SessionFormatError`. Anything else (an ``AttributeError`` from a
record of the wrong shape, a ``KeyError``, a ``TypeError``) is a bug that
would reach the developer as a traceback.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from golden.corpus import CORPUS
from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.gare import GareRunSource
from ter.adapters.driven.otel_genai import OtelGenAiSource
from ter.domain import SessionFormatError, SessionTrace

from .strategies import fuzz, json_values, jsonl_lines, mutated

GARE_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "gare" / "failover-run"
FUZZ = fuzz(150)


def _records(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


CLAUDE_RECORDS = [r for path in sorted(CORPUS.values())[:6] for r in _records(path)]
GARE_EXPLAIN = json.loads((GARE_RUN / "explain.json").read_text(encoding="utf-8"))
GARE_USAGE = _records(GARE_RUN / "gare-ter-usage.jsonl")


def _well_formed(trace: SessionTrace) -> None:
    ids = [e.id for e in trace.events]
    assert len(ids) == len(set(ids)), "event ids must be unique"
    assert [e.sequence for e in trace.events] == list(range(len(ids)))


def _read_or_refuse(read: Any, ref: Path) -> None:
    try:
        trace = read(ref)
    except SessionFormatError:
        event("refused")
        return
    event("read")
    _well_formed(trace)


@pytest.mark.req("TER-TST-009")
@FUZZ
@given(lines=jsonl_lines(CLAUDE_RECORDS))
def test_claude_code_source_reads_or_refuses(lines: list[str]) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "session.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _read_or_refuse(ClaudeCodeJsonlSource().read, path)


@pytest.mark.req("TER-TST-009")
@FUZZ
@given(data=st.binary(max_size=200))
def test_claude_code_source_reads_or_refuses_any_bytes(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "session.jsonl"
        path.write_bytes(data)
        _read_or_refuse(ClaudeCodeJsonlSource().read, path)


@pytest.mark.req("TER-TST-009")
@FUZZ
@given(
    explain=st.one_of(mutated(GARE_EXPLAIN, max_mutations=4), json_values),
    usage=jsonl_lines(GARE_USAGE),
)
def test_gare_source_reads_or_refuses(explain: Any, usage: list[str]) -> None:
    with tempfile.TemporaryDirectory() as folder:
        run = Path(folder)
        (run / "explain.json").write_text(json.dumps(explain), encoding="utf-8")
        (run / "gare-ter-usage.jsonl").write_text("\n".join(usage), encoding="utf-8")
        _read_or_refuse(GareRunSource().read, run)


OTEL_EXPORT = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures" / "otel" / "lean_mix.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.req("TER-TST-009")
@FUZZ
@given(document=st.one_of(mutated(OTEL_EXPORT, max_mutations=4), json_values))
def test_otel_genai_source_reads_or_refuses(document: Any) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "spans.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        _read_or_refuse(OtelGenAiSource().read, path)


@pytest.mark.req("TER-TST-009")
@FUZZ
@given(data=st.binary(max_size=200))
def test_otel_genai_source_reads_or_refuses_any_bytes(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "spans.json"
        path.write_bytes(data)
        _read_or_refuse(OtelGenAiSource().read, path)
