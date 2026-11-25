"""OpenTelemetry GenAI spans as a session source (Stage U2, ADR 0008)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from tests.golden.corpus import CORPUS
from tests.golden.otel_export import FIXTURES, otlp_export

from ter import bootstrap
from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.otel_genai import (
    CONVENTION_UNSTATED,
    INTENTS_FROM_TRACES,
    KNOWN_CONVENTIONS,
    NO_RESPONSE_TEXT,
    OtelGenAiSource,
)
from ter.adapters.driving.cli import main as cli_main
from ter.domain import Actor, EventKind, SessionFormatError, ToolKind
from ter.domain.events import describe_limit

SOURCE = OtelGenAiSource()
TRACE = "0af7651916cd43dd8448eb211c80319c"


def _attr(key: str, value: object) -> dict[str, Any]:
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _span(
    span_id: str,
    op: str | None,
    start: int,
    *,
    parent: str = "",
    trace: str = TRACE,
    **attrs: object,
) -> dict[str, Any]:
    attributes = [] if op is None else [_attr("gen_ai.operation.name", op)]
    attributes += [_attr(k.replace("__", "."), v) for k, v in attrs.items()]
    return {
        "traceId": trace,
        "spanId": span_id,
        "parentSpanId": parent,
        "name": f"{op} span",
        "startTimeUnixNano": str(start * 1_000_000_000),
        "endTimeUnixNano": str(start * 1_000_000_000 + 500_000_000),
        "attributes": attributes,
    }


def _export(
    tmp_path: Path, spans: list[dict[str, Any]], url: str | None = "1.45.0"
) -> Path:
    scope: dict[str, Any] = {"scope": {"name": "agent"}, "spans": spans}
    if url is not None:
        scope["schemaUrl"] = f"https://opentelemetry.io/schemas/{url}"
    path = tmp_path / "spans.json"
    path.write_text(
        json.dumps({"resourceSpans": [{"resource": {}, "scopeSpans": [scope]}]}),
        encoding="utf-8",
    )
    return path


def _chat(span_id: str, start: int, **attrs: object) -> dict[str, Any]:
    base: dict[str, object] = {
        "gen_ai__provider__name": "anthropic",
        "gen_ai__request__model": "claude-sonnet-5-5",
        "gen_ai__conversation__id": "conv-1",
        "gen_ai__usage__input_tokens": 1_000,
        "gen_ai__usage__output_tokens": 50,
    }
    base.update(attrs)
    return _span(span_id, "chat", start, **base)


# -- mapping (TER-SRC-027) ---------------------------------------------------------


@pytest.mark.req("TER-SRC-027")
def test_inference_and_tool_spans_become_turns_and_tool_events(tmp_path: Path) -> None:
    calls = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "content": "Reading it."},
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "Read",
                        "arguments": {"file_path": "a.py"},
                    },
                ],
            }
        ]
    )
    prompt = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": "fix a.py"}]}]
    )
    path = _export(
        tmp_path,
        [
            _chat(
                "s1",
                1,
                gen_ai__input__messages=prompt,
                gen_ai__output__messages=calls,
            ),
            _span(
                "s2",
                "execute_tool",
                2,
                gen_ai__tool__name="Read",
                gen_ai__tool__call__id="call-1",
                gen_ai__tool__call__result="print(1)",
                gen_ai__conversation__id="conv-1",
            ),
        ],
    )
    trace = SOURCE.read(path)
    assert trace.session_id == "conv-1"
    assert [(e.kind, e.actor) for e in trace.events] == [
        (EventKind.PROMPT, Actor.USER),
        (EventKind.RESPONSE, Actor.ASSISTANT),
        (EventKind.TOOL_REQUESTED, Actor.ASSISTANT),
        (EventKind.TOOL_COMPLETED, Actor.TOOL),
    ]
    prompt_event, response, request, result = trace.events
    assert prompt_event.text == "fix a.py"
    assert response.text == "Reading it." and response.usage is not None
    assert request.usage is None  # one turn's usage, once
    assert request.tool is not None and request.tool.kind is ToolKind.FS_READ
    assert request.tool.arguments == {"file_path": "a.py"}
    assert result.text == "print(1)" and result.tool is not None
    assert result.tool.call_id == "call-1"
    assert trace.coverage == 1.0
    assert trace.usage_limits == ()


@pytest.mark.req("TER-SRC-027")
def test_every_span_it_cannot_map_counts_against_coverage(tmp_path: Path) -> None:
    path = _export(
        tmp_path,
        [
            _chat("s1", 1),
            _span("s2", "embeddings", 2, gen_ai__conversation__id="conv-1"),
            _span("s3", None, 3),  # an HTTP span with no GenAI operation
            {"spanId": "no-trace"},
        ],
    )
    trace = SOURCE.read(path)
    assert trace.unrecognised_by_type == {
        "span:embeddings": 1,
        "span:not-gen_ai": 1,
        "span:invalid": 1,
    }
    assert trace.coverage == pytest.approx(1 / 4)


@pytest.mark.req("TER-SRC-027")
def test_an_agent_under_an_agent_is_a_handoff_and_a_top_agent_is_metadata(
    tmp_path: Path,
) -> None:
    path = _export(
        tmp_path,
        [
            _span("a1", "invoke_agent", 1, gen_ai__agent__name="lead"),
            _chat("s1", 2, parent="a1"),
            _span("a2", "invoke_agent", 3, parent="s1", gen_ai__agent__name="search"),
            _chat("s2", 4, parent="a2"),
        ],
    )
    trace = SOURCE.read(path)
    handoffs = [
        e for e in trace.events if e.tool and e.tool.kind is ToolKind.AGENT_HANDOFF
    ]
    assert [e.text for e in handoffs] == ["search"]
    assert EventKind.SUBAGENT_COMPLETED in {e.kind for e in trace.events}
    assert trace.metadata_by_type == {"span:invoke_agent": 1}
    assert trace.coverage == 1.0


# -- usage ------------------------------------------------------------------------


@pytest.mark.req("TER-SRC-029")
def test_cached_tokens_are_taken_out_of_the_conventions_input_count(
    tmp_path: Path,
) -> None:
    path = _export(
        tmp_path,
        [
            _chat(
                "s1",
                1,
                gen_ai__usage__input_tokens=1_000,
                gen_ai__usage__cache_read__input_tokens=600,
                gen_ai__usage__cache_creation__input_tokens=300,
                gen_ai__response__model="claude-sonnet-5-5-20261001",
            )
        ],
    )
    usage = SOURCE.read(path).events[1].usage
    assert usage is not None
    assert (
        usage.input_tokens,
        usage.cache_read_tokens,
        usage.cache_creation_tokens,
    ) == (
        100,
        600,
        300,
    )
    assert usage.output_tokens == 50
    assert usage.model == "claude-sonnet-5-5-20261001"  # the response model wins
    assert usage.cache_reported


@pytest.mark.req("TER-SRC-029")
def test_the_cache_write_count_is_read_under_either_name(tmp_path: Path) -> None:
    path = _export(
        tmp_path,
        [_chat("s1", 1, gen_ai__usage__cache_write__input_tokens=400)],
    )
    usage = SOURCE.read(path).events[1].usage
    assert usage is not None
    assert (usage.input_tokens, usage.cache_creation_tokens) == (600, 400)


@pytest.mark.req("TER-SRC-029")
def test_a_turn_with_no_cache_attribute_is_not_read_as_zero_cache(
    tmp_path: Path,
) -> None:
    usage = SOURCE.read(_export(tmp_path, [_chat("s1", 1)])).events[1].usage
    assert usage is not None
    assert usage.input_tokens == 1_000
    assert not usage.cache_reported  # "when available and non-zero": unknown


@pytest.mark.req("TER-SRC-029")
def test_cached_counts_larger_than_the_input_never_go_negative(tmp_path: Path) -> None:
    path = _export(
        tmp_path,
        [_chat("s1", 1, gen_ai__usage__cache_read__input_tokens=5_000)],
    )
    usage = SOURCE.read(path).events[1].usage
    assert usage is not None and usage.input_tokens == 0


# -- content capture off (TER-SRC-030) ---------------------------------------------


@pytest.mark.req("TER-SRC-030")
def test_without_content_each_trace_is_one_intent_and_the_limits_say_so(
    tmp_path: Path,
) -> None:
    path = _export(
        tmp_path,
        [
            _chat("s1", 1),
            _span(
                "s2",
                "execute_tool",
                2,
                gen_ai__tool__name="Bash",
                gen_ai__conversation__id="conv-1",
            ),
            _chat("s3", 3),
            _chat("s4", 5, trace="1" * 32),
        ],
    )
    trace = SOURCE.read(path)
    assert trace.usage_limits == (NO_RESPONSE_TEXT, INTENTS_FROM_TRACES)
    kinds = [e.kind for e in trace.events]
    assert kinds.count(EventKind.PROMPT) == 2  # two traces
    # The tool span is the only record of the request: it is generated.
    request = next(e for e in trace.events if e.kind is EventKind.TOOL_REQUESTED)
    assert request.actor is Actor.ASSISTANT and request.tool is not None
    assert request.tool.kind is ToolKind.EXEC_SHELL
    for limit in trace.usage_limits:
        assert describe_limit(limit) != limit  # each reads as a sentence


# -- versions (TER-SRC-028) ----------------------------------------------------------


@pytest.mark.req("TER-SRC-028")
@pytest.mark.parametrize("version", ["1.26.0", "1.46.0", "https://example.com/x"])
def test_an_unknown_convention_version_is_refused_by_name(
    tmp_path: Path, version: str
) -> None:
    path = _export(tmp_path, [_chat("s1", 1)], url=None)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["resourceSpans"][0]["scopeSpans"][0]["schemaUrl"] = (
        version
        if version.startswith("http")
        else f"https://opentelemetry.io/schemas/{version}"
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SessionFormatError, match=version.replace(".", r"\.")):
        SOURCE.read(path)


@pytest.mark.req("TER-SRC-028")
@pytest.mark.parametrize("version", KNOWN_CONVENTIONS)
def test_every_known_version_is_read(tmp_path: Path, version: str) -> None:
    assert SOURCE.read(_export(tmp_path, [_chat("s1", 1)], url=version)).events


@pytest.mark.req("TER-SRC-028")
@pytest.mark.parametrize("resource", ["1.20.0", "1.45.0"])
def test_a_resource_schema_url_does_not_version_the_spans(
    tmp_path: Path, resource: str
) -> None:
    """A resource's schemaUrl versions its resource attributes only; spans
    take their version from their scope (Qodo #114 finding 7)."""
    path = _export(tmp_path, [_chat("s1", 1)], url=None)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["resourceSpans"][0]["schemaUrl"] = (
        f"https://opentelemetry.io/schemas/{resource}"
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    trace = SOURCE.read(path)
    assert CONVENTION_UNSTATED in trace.usage_limits


@pytest.mark.req("TER-SRC-028")
def test_a_file_naming_no_version_is_read_under_the_latest_and_says_so(
    tmp_path: Path,
) -> None:
    trace = SOURCE.read(_export(tmp_path, [_chat("s1", 1)], url=None))
    assert CONVENTION_UNSTATED in trace.usage_limits


# -- one session per file ----------------------------------------------------------------


def test_several_conversations_in_one_file_are_refused(tmp_path: Path) -> None:
    path = _export(
        tmp_path, [_chat("s1", 1), _chat("s2", 2, gen_ai__conversation__id="conv-2")]
    )
    with pytest.raises(SessionFormatError, match="conv-1, conv-2"):
        SOURCE.read(path)


def test_without_conversation_ids_one_trace_is_the_session(tmp_path: Path) -> None:
    span = _span("s1", "chat", 1, gen_ai__usage__input_tokens=5)
    assert SOURCE.read(_export(tmp_path, [span])).session_id == TRACE
    other = _span("s2", "chat", 2, trace="2" * 32, gen_ai__usage__input_tokens=5)
    with pytest.raises(SessionFormatError, match="2 traces"):
        SOURCE.read(_export(tmp_path, [span, other]))


def test_collector_file_exporter_lines_are_read(tmp_path: Path) -> None:
    single = json.loads(_export(tmp_path, [_chat("s1", 1)]).read_text(encoding="utf-8"))
    second = json.loads(_export(tmp_path, [_chat("s2", 2)]).read_text(encoding="utf-8"))
    path = tmp_path / "otlp.jsonl"
    path.write_text(
        json.dumps(single) + "\n" + json.dumps(second) + "\nnot json\n",
        encoding="utf-8",
    )
    trace = SOURCE.read(path)
    assert sum(1 for e in trace.events if e.usage) == 2
    assert trace.unrecognised_by_type == {"otlp:invalid": 1}


@pytest.mark.parametrize(
    "content",
    [b"", b"not json", b"\xff\xfe", b'{"resourceSpans": "no"}', b"{}"],
    ids=["empty", "text", "not-utf8", "bad-spans", "not-otlp"],
)
def test_a_file_that_is_not_a_span_export_raises_a_format_error(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(SessionFormatError):
        SOURCE.read(path)


def test_accepts_span_exports_and_nothing_else(tmp_path: Path) -> None:
    assert OtelGenAiSource.accepts(FIXTURES / "lean_mix.json")
    assert not OtelGenAiSource.accepts(CORPUS["lean_mix"])
    assert not OtelGenAiSource.accepts(tmp_path)
    assert not OtelGenAiSource.accepts(tmp_path / "missing.json")
    assert isinstance(
        bootstrap.session_source_for(FIXTURES / "lean_mix.json"), OtelGenAiSource
    )


# -- harness neutral (TER-SRC-031) -------------------------------------------------------


@pytest.mark.req("TER-SRC-031")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_a_session_read_from_its_transcript_and_its_spans_gives_the_same_events(
    name: str,
) -> None:
    transcript = ClaudeCodeJsonlSource().read(CORPUS[name])
    spans = SOURCE.read(FIXTURES / f"{name}.json")

    def what(trace: Any) -> list[tuple[object, ...]]:
        return [(e.kind, e.actor, e.text, e.usage, e.tool) for e in trace.events]

    assert what(spans) == what(transcript)
    assert spans.coverage == 1.0


def _record(path: Path) -> dict[str, Any]:
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(
        ["record", "export", str(path), "--ter", "off"],
        bootstrap.cli_services(),
        stdout=out,
        stderr=err,
    )
    assert code == 0, err.getvalue()
    record: dict[str, Any] = json.loads(out.getvalue())
    return record


@pytest.mark.req("TER-SRC-031")
def test_the_record_of_a_session_is_the_same_whichever_way_it_was_read() -> None:
    """The document's falsification test: Records of one session read from
    its transcript and from OpenTelemetry differ only where the source is
    named (its format, its id and the digests of its events)."""
    from_transcript = _record(CORPUS["lean_mix"])
    from_spans = _record(FIXTURES / "lean_mix.json")
    for record in (from_transcript, from_spans):
        record.pop("integrity")
        record["session"].pop("source")
        record["session"].pop("id")
    assert from_spans == from_transcript


def test_the_committed_fixtures_are_what_the_exporter_writes() -> None:
    reader = ClaudeCodeJsonlSource()
    for name, path in CORPUS.items():
        written = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        assert written == otlp_export(reader.read(path)), name
    lean = reader.read(CORPUS["lean_mix"])
    written = json.loads(
        (FIXTURES / "lean_mix.no-content.json").read_text(encoding="utf-8")
    )
    assert written == otlp_export(lean, content=False)


# -- review findings (Qodo, PR #114) -------------------------------------------------


def _prompt(text: str) -> str:
    return json.dumps([{"role": "user", "parts": [{"type": "text", "content": text}]}])


@pytest.mark.req("TER-SRC-030")
def test_a_trace_without_captured_input_keeps_its_own_intent(tmp_path: Path) -> None:
    """Finding 1: capture is judged per trace, so a partly captured
    conversation loses no intent and says where intents came from traces."""
    other = "1" * 32
    path = _export(
        tmp_path,
        [
            _chat("s1", 1, gen_ai__input__messages=_prompt("first")),
            _span("s2", "chat", 5, trace=other, gen_ai__conversation__id="conv-1"),
        ],
    )
    trace = SOURCE.read(path)
    prompts = [e for e in trace.events if e.kind is EventKind.PROMPT]
    assert [p.text for p in prompts] == ["first", ""]
    assert INTENTS_FROM_TRACES in trace.usage_limits
    whole = SOURCE.read(
        _export(tmp_path, [_chat("s1", 1, gen_ai__input__messages=_prompt("x"))])
    )
    assert INTENTS_FROM_TRACES not in whole.usage_limits


@pytest.mark.req("TER-SRC-029")
def test_a_trace_naming_no_conversation_beside_one_that_does_is_refused(
    tmp_path: Path,
) -> None:
    """Finding 2: an unrelated trace is not folded into the named session."""
    stray = _span("s9", "chat", 3, trace="2" * 32)
    path = _export(tmp_path, [_chat("s1", 1), stray])
    with pytest.raises(SessionFormatError, match="1 trace.*name no conversation"):
        SOURCE.read(path)
    # A tool span of a named trace that does not repeat the id is fine.
    tool = _span("s2", "execute_tool", 2, gen_ai__tool__name="Read")
    assert SOURCE.read(_export(tmp_path, [_chat("s1", 1), tool])).session_id == "conv-1"


@pytest.mark.req("TER-SRC-027")
def test_a_call_id_reused_in_another_trace_keeps_its_own_request(
    tmp_path: Path,
) -> None:
    """Finding 3: a call named by one trace's inference does not hide the
    request of another trace's tool span with the same id."""
    calls = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [{"type": "tool_call", "id": "c1", "name": "Read"}],
            }
        ]
    )
    other = "3" * 32
    spans = [
        _chat("s1", 1, gen_ai__output__messages=calls),
        _span(
            "s2",
            "execute_tool",
            2,
            gen_ai__tool__name="Read",
            gen_ai__tool__call__id="c1",
        ),
        _chat("s3", 3, trace=other),
        _span(
            "s4",
            "execute_tool",
            4,
            trace=other,
            gen_ai__tool__name="Read",
            gen_ai__tool__call__id="c1",
            gen_ai__conversation__id="conv-1",
        ),
    ]
    trace = SOURCE.read(_export(tmp_path, spans))
    requests = [e for e in trace.events if e.kind is EventKind.TOOL_REQUESTED]
    assert len(requests) == 2  # one from trace 1's output, one from trace 2's span


@pytest.mark.req("TER-SRC-027")
def test_a_span_id_reused_in_another_trace_is_not_its_parent(tmp_path: Path) -> None:
    """Finding 4: parents are looked up within the span's own trace."""
    other = "4" * 32
    spans = [
        _span("a1", "invoke_agent", 1, gen_ai__conversation__id="conv-1"),
        _chat("s1", 2, parent="a1"),
        _span("p1", "chat", 3, trace=other, gen_ai__conversation__id="conv-1"),
        # Its parent id "a1" names an agent span only in trace 1, so within
        # its own trace it has no parent: not a nested agent.
        _span("x1", "invoke_agent", 4, trace=other, parent="a1"),
    ]
    trace = SOURCE.read(_export(tmp_path, spans))
    assert not [e for e in trace.events if e.kind is EventKind.SUBAGENT_COMPLETED]


@pytest.mark.req("TER-SRC-027")
def test_only_an_export_whose_top_level_key_is_resource_spans_is_accepted(
    tmp_path: Path,
) -> None:
    """Finding 5: a transcript that mentions resourceSpans is not an export."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"content": '{"resourceSpans": []}'}})
        + "\n",
        encoding="utf-8",
    )
    assert not OtelGenAiSource.accepts(transcript)
    export = _export(tmp_path, [_chat("s1", 1)])
    assert OtelGenAiSource.accepts(export)
    spaced = tmp_path / "spaced.json"
    spaced.write_text('\ufeff \n { "resourceSpans" : []}', encoding="utf-8")
    assert OtelGenAiSource.accepts(spaced)


@pytest.mark.req("TER-SRC-029")
def test_identifiers_from_the_file_cannot_drive_the_terminal(tmp_path: Path) -> None:
    """Finding 6: control characters in a conversation id or version are
    escaped before they reach a report or an error."""
    path = _export(tmp_path, [_chat("s1", 1, gen_ai__conversation__id="c\x1b[2Jx")])
    trace = SOURCE.read(path)
    assert trace.session_id == "c\\x1b[2Jx"
    assert all(e.session_id == trace.session_id for e in trace.events)
    path = _export(tmp_path, [_chat("s1", 1)], url=None)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["resourceSpans"][0]["scopeSpans"][0]["schemaUrl"] = "x\x07y"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SessionFormatError) as refused:
        SOURCE.read(path)
    assert "\x07" not in str(refused.value) and "x\\x07y" in str(refused.value)
