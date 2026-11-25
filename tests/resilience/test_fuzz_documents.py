"""TER's own files on hostile input (TER-TST-011, TER-TST-013).

Policy, limits and measures files are edited by hand; the live event log
and the ledger are appended by hook processes that can die mid-write. A
reader of a file a person edits refuses a broken one with its own error
type, which the CLI turns into a reason. A reader of an append-only log
skips and counts a broken line, and what it does read can still be
analysed.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from golden.corpus import CORPUS, GOLDEN_DIR, REPO_ROOT
from ter import bootstrap
from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.event_log import JsonlEventLog, event_to_record
from ter.adapters.driven.in_memory import FixedClock
from ter.adapters.driven.intervention_ledger import JsonlInterventionLedger
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.application.shadow import shadow_sessions
from ter.domain.intervention import PolicyError
from ter.domain.lean.control import ControlLimits, ControlLimitsError, MeasuresDocument
from ter.domain.policy_set import PolicySet
from ter.domain.stream import analyse_batch, explain_batch

from .strategies import fuzz, json_values, jsonl_lines, mutated

POLICIES = REPO_ROOT / "docs" / "examples" / "policies" / "l4-advisory.json"
CONTROL = GOLDEN_DIR / "snapshots" / "control"
FUZZ = fuzz(200)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


DRAFT = _json(POLICIES)
APPROVED = PolicySet.from_mapping(DRAFT).approve("Fuzz", date(2026, 10, 14)).as_dict()
LIMITS = _json(CONTROL / "corpus.limits.json")
MEASURES = _json(CONTROL / "corpus.measures.json")


def _refused_or_read(read: Any, doc: Any, error: type[Exception]) -> None:
    try:
        read(doc)
    except error:
        event("refused")
        return
    event("read")


@pytest.mark.req("TER-TST-011")
@FUZZ
@given(
    doc=st.one_of(
        mutated(DRAFT, max_mutations=2), mutated(APPROVED, max_mutations=2), json_values
    )
)
def test_a_policy_file_is_read_or_refused_with_a_policy_error(doc: Any) -> None:
    _refused_or_read(PolicySet.from_mapping, doc, PolicyError)


@pytest.mark.req("TER-TST-011")
@FUZZ
@given(doc=st.one_of(mutated(LIMITS, max_mutations=2), json_values))
def test_a_limits_file_is_read_or_refused_with_a_limits_error(doc: Any) -> None:
    _refused_or_read(ControlLimits.from_mapping, doc, ControlLimitsError)


@pytest.mark.req("TER-TST-011")
@FUZZ
@given(doc=st.one_of(mutated(MEASURES, max_mutations=2), json_values))
def test_a_measures_file_is_read_or_refused_with_a_limits_error(doc: Any) -> None:
    _refused_or_read(MeasuresDocument.from_mapping, doc, ControlLimitsError)


# --- append-only logs ------------------------------------------------------

SESSION = min(CORPUS.values())


@cache
def _event_records() -> list[Any]:
    trace = ClaudeCodeJsonlSource().read(SESSION)
    return [event_to_record(e) for e in trace.events]


@cache
def _ledger_records() -> list[Any]:
    with tempfile.TemporaryDirectory() as folder:
        ledger = JsonlInterventionLedger(folder)
        services = bootstrap.cli_services()
        assert services.explain_transcript is not None
        explain = services.explain_transcript
        shadow_sessions(
            sorted(CORPUS.values()),
            lambda path: explain(path, "regex", "off"),
            PolicySet.from_mapping(APPROVED),
            ledger,
            FixedClock(datetime(2026, 10, 14, tzinfo=UTC)),
        )
        lines = [
            json.loads(line)
            for path in sorted(Path(folder).iterdir())
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    assert lines, "the shadow run wrote no ledger records"
    return lines


def _with_sentinel(data: st.DataObject, records: list[Any]) -> tuple[list[str], str]:
    """Damaged lines with one intact record of the first record's session among
    them, and its id: a reader that drops everything fails, though it never
    raises."""
    lines = data.draw(jsonl_lines(records))
    session = records[0]["session_id"]
    sentinel = data.draw(
        st.sampled_from([r for r in records if r["session_id"] == session])
    )
    lines.insert(data.draw(st.integers(0, len(lines))), json.dumps(sentinel))
    return lines, str(sentinel["id"])


@pytest.mark.req("TER-TST-013")
@FUZZ
@given(data=st.data())
def test_a_damaged_event_log_is_read_and_analysed(data: st.DataObject) -> None:
    records = _event_records()
    lines, sentinel = _with_sentinel(data, records)
    session = records[0]["session_id"]
    with tempfile.TemporaryDirectory() as folder:
        log = JsonlEventLog(folder)
        log.path_for(session).parent.mkdir(parents=True, exist_ok=True)
        log.path_for(session).write_text("\n".join(lines) + "\n", encoding="utf-8")
        events = log.events(session)
    event(f"skipped some: {log.skipped > 0}")
    assert sentinel in {e.id for e in events}, "a valid line among damage is lost"
    analyse_batch(events, RegexTokenizer())
    explain_batch(events, RegexTokenizer())


@pytest.mark.req("TER-TST-013")
@FUZZ
@given(data=st.data())
def test_a_damaged_ledger_is_read(data: st.DataObject) -> None:
    records = _ledger_records()
    lines, sentinel = _with_sentinel(data, records)
    session = records[0]["session_id"]
    with tempfile.TemporaryDirectory() as folder:
        ledger = JsonlInterventionLedger(folder)
        path = ledger.path_for(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        entries = ledger.entries(session)
        ledger.sessions()
    assert all(e.session_id == session for e in entries)
    assert sentinel in {e.id for e in entries}, "a valid line among damage is lost"


FOCUS_PRICES = (REPO_ROOT / "tests" / "fixtures" / "focus" / "sku_price.csv").read_text(
    encoding="utf-8"
)


def _read_prices(text: str) -> None:
    from ter.adapters.driven.pricing.focus import FocusPriceError, FocusSkuPriceBook

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "prices.csv"
        path.write_text(text, encoding="utf-8")
        try:
            book = FocusSkuPriceBook(path)
        except FocusPriceError:
            event("refused")
            return
        event("read")
        for model in book.models():
            book.rate(model)


@pytest.mark.req("TER-TST-011")
@FUZZ
@given(data=st.data())
def test_a_focus_price_file_is_read_or_refused_with_a_price_error(
    data: st.DataObject,
) -> None:
    lines = FOCUS_PRICES.splitlines()
    for _ in range(data.draw(st.integers(0, 4))):
        index = data.draw(st.integers(0, len(lines) - 1))
        cells = lines[index].split(",")
        cell = data.draw(st.integers(0, len(cells) - 1))
        cells[cell] = data.draw(
            st.one_of(
                st.text(max_size=12),
                st.sampled_from(["", "List", "USD", "1M Tokens", "-1", "nan", '{"a"']),
            )
        )
        lines[index] = ",".join(cells)
    _read_prices("\n".join(lines) + "\n")


FOCUS_BILLING = (REPO_ROOT / "tests" / "fixtures" / "focus" / "billing.csv").read_text(
    encoding="utf-8"
)
RECORD = (
    REPO_ROOT / "tests" / "golden" / "snapshots" / "record" / "lean_mix.record.json"
)


def _join_billing(text: str) -> None:
    from ter.adapters.driving.focus_export import FocusExportError, export_focus

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "billing.csv"
        path.write_text(text, encoding="utf-8")
        try:
            export = export_focus(path, [RECORD])
        except FocusExportError:
            event("refused")
            return
        event("joined" if export.rows_joined else "read")
        assert export.rows_joined <= export.rows


@pytest.mark.req("TER-TST-011")
@FUZZ
@given(data=st.data())
def test_a_focus_billing_file_is_joined_or_refused_with_an_export_error(
    data: st.DataObject,
) -> None:
    lines = FOCUS_BILLING.splitlines()
    for _ in range(data.draw(st.integers(0, 4))):
        index = data.draw(st.integers(0, len(lines) - 1))
        cells = lines[index].split(",")
        cell = data.draw(st.integers(0, len(cells) - 1))
        cells[cell] = data.draw(
            st.one_of(
                st.text(max_size=12),
                st.sampled_from(
                    [
                        "",
                        "1M Tokens",
                        "-1",
                        "nan",
                        "inf",
                        '{"a"',
                        "2026-04-01T00:00:00Z",
                        '"',
                    ]
                ),
            )
        )
        lines[index] = ",".join(cells)
    _join_billing("\n".join(lines) + "\n")


def _mcp_server(root: Path) -> Any:
    from ter.adapters.driving.mcp_server import McpServer, McpServices

    def tool(path: Path) -> dict[str, Any]:
        return {"session": path.name}

    return McpServer(McpServices(tool, tool, tool), [root], quiet_seconds=0)


MCP_MESSAGES = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    },
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "session_record", "arguments": {"path": "s.jsonl"}},
    },
]


@pytest.mark.req("TER-INT-026")
@FUZZ
@given(
    message=st.one_of(*(mutated(m, max_mutations=3) for m in MCP_MESSAGES), json_values)
)
def test_an_mcp_message_is_answered_or_refused_with_a_json_rpc_error(
    message: Any,
) -> None:
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "s.jsonl").write_text("{}\n", encoding="utf-8")
        response = _mcp_server(Path(folder)).handle(message)
    if response is None:
        event("no answer")
        return
    assert response["jsonrpc"] == "2.0"
    assert ("result" in response) != ("error" in response)
    if "error" in response:
        event(f"error {response['error']['code']}")
        assert response["error"]["code"] != -32603, response
    else:
        event("answered")
    json.dumps(response)
