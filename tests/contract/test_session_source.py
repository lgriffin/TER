"""Contract suite for the ``SessionSource`` port.

The real Claude Code adapter, the GARE adapter (on mock runs and on real
recorded runs) and the in-memory fake run the same assertions, so the fake
cannot drift from the obligations the real adapters meet.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.gare import GareRunSource
from ter.adapters.driven.in_memory import InMemorySessionSource
from ter.adapters.driven.otel_genai import OtelGenAiSource
from ter.domain import Actor, EventKind, SessionFormatError
from ter.ports import SessionSource

from golden.corpus import CORPUS

Factory = Callable[[], tuple[SessionSource, list[str]]]


def _claude() -> tuple[SessionSource, list[str]]:
    return ClaudeCodeJsonlSource(), [str(p) for p in CORPUS.values()]


GARE_RUNS = Path(__file__).resolve().parents[1] / "fixtures" / "gare"
#: Real recorded GARE runs (issue #55); the folders beside ``runs/`` are mocks.
GARE_REAL_RUNS = GARE_RUNS / "runs"


def _gare() -> tuple[SessionSource, list[str]]:
    mocks = [
        p for p in sorted(GARE_RUNS.iterdir()) if p.is_dir() and p != GARE_REAL_RUNS
    ]
    return GareRunSource(), [str(p) for p in mocks]


def _gare_real() -> tuple[SessionSource, list[str]]:
    runs = [str(p) for p in sorted(GARE_REAL_RUNS.iterdir()) if p.is_dir()]
    assert runs, "no real recorded GARE run"
    return GareRunSource(), runs


OTEL_EXPORTS = Path(__file__).resolve().parents[1] / "fixtures" / "otel"


def _otel() -> tuple[SessionSource, list[str]]:
    exports = [str(p) for p in sorted(OTEL_EXPORTS.glob("*.json"))]
    assert exports, "no OpenTelemetry GenAI export fixture"
    return OtelGenAiSource(), exports


def _memory() -> tuple[SessionSource, list[str]]:
    real = ClaudeCodeJsonlSource()
    traces = {name: real.read(path) for name, path in CORPUS.items()}
    return InMemorySessionSource(traces), list(traces)


@pytest.fixture(
    params=[
        pytest.param(_claude, id="claude-code-jsonl"),
        pytest.param(_gare, id="gare-run-mock"),
        # A second harness's real sessions pass the same suite (TER-SRC-010).
        pytest.param(
            _gare_real, id="gare-run-real", marks=pytest.mark.req("TER-SRC-010")
        ),
        pytest.param(_otel, id="otel-genai", marks=pytest.mark.req("TER-SRC-027")),
        pytest.param(_memory, id="in-memory"),
    ]
)
def source(request: pytest.FixtureRequest) -> tuple[SessionSource, list[str]]:
    factory: Factory = request.param
    return factory()


def test_satisfies_the_port_protocol(source: tuple[SessionSource, list[str]]) -> None:
    adapter, _ = source
    assert isinstance(adapter, SessionSource)
    assert adapter.format_name


@pytest.mark.req("TER-SRC-004")
def test_reading_is_deterministic(source: tuple[SessionSource, list[str]]) -> None:
    adapter, refs = source
    for ref in refs:
        assert adapter.read(ref) == adapter.read(ref)


def test_event_ids_are_unique_and_sequences_contiguous(
    source: tuple[SessionSource, list[str]],
) -> None:
    adapter, refs = source
    for ref in refs:
        trace = adapter.read(ref)
        ids = [e.id for e in trace.events]
        assert len(ids) == len(set(ids)), ref
        assert [e.sequence for e in trace.events] == list(range(len(trace.events)))
        assert trace.source_format == adapter.format_name


def test_events_belong_to_their_session_and_chain_in_order(
    source: tuple[SessionSource, list[str]],
) -> None:
    adapter, refs = source
    for ref in refs:
        trace = adapter.read(ref)
        assert trace.events, ref
        assert {e.session_id for e in trace.events} == {trace.session_id}
        assert trace.events[0].parent_id is None
        for earlier, later in zip(trace.events, trace.events[1:]):
            assert later.parent_id == earlier.id


@pytest.mark.req("TER-ANL-001")
def test_generated_events_exclude_user_input_and_tool_output(
    source: tuple[SessionSource, list[str]],
) -> None:
    adapter, refs = source
    for ref in refs:
        generated = adapter.read(ref).generated()
        assert generated
        assert all(e.actor is Actor.ASSISTANT for e in generated)
        assert not {e.kind for e in generated} & {
            EventKind.PROMPT,
            EventKind.TOOL_COMPLETED,
        }


def test_unknown_reference_raises_file_not_found(
    source: tuple[SessionSource, list[str]], tmp_path: Path
) -> None:
    adapter, _ = source
    with pytest.raises(FileNotFoundError):
        adapter.read(str(tmp_path / "missing.jsonl"))


@pytest.mark.req("TER-TST-007")
@pytest.mark.parametrize(
    "content",
    [b"not json\n", b'{"type": "user"\n', b"\xff\xfe{}\n", b""],
    ids=["text", "torn", "not-utf8", "empty"],
)
def test_a_file_that_is_not_a_session_raises_a_format_error(
    content: bytes, tmp_path: Path
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(content)
    with pytest.raises(SessionFormatError):
        ClaudeCodeJsonlSource().read(path)


@pytest.mark.req("TER-TST-007")
@pytest.mark.parametrize(
    "content",
    [b"not json\n", b'{"resourceSpans": [}', b"\xff\xfe{}\n", b"", b"{}"],
    ids=["text", "torn", "not-utf8", "empty", "not-otlp"],
)
def test_a_file_that_is_not_a_span_export_raises_a_format_error(
    content: bytes, tmp_path: Path
) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(SessionFormatError):
        OtelGenAiSource().read(path)


@pytest.mark.req("TER-TST-007")
@pytest.mark.parametrize(
    "explain", ["{}", "not json", '{"run": {}, "events": [], "tasks": []}']
)
def test_a_folder_that_is_not_a_gare_run_raises_a_format_error(
    explain: str, tmp_path: Path
) -> None:
    (tmp_path / "explain.json").write_text(explain, encoding="utf-8")
    with pytest.raises(SessionFormatError):
        GareRunSource().read(tmp_path)
