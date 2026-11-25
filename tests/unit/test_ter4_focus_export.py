"""TER Records joined to FOCUS billing rows (Stage U2, ADR 0008)."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from ter import bootstrap
from ter.adapters.driving.cli import main as cli_main
from ter.adapters.driving.focus_export import (
    EFFICIENCY_COLUMNS,
    FocusExportError,
    export_focus,
)
from ter.domain.focus_join import (
    BillingRow,
    RecordFacts,
    focus_token_kind,
    join_records,
    model_matches,
    tokens_per_unit,
)

DAY = date(2026, 9, 21)
START = datetime(2026, 9, 21, tzinfo=UTC)
END = datetime(2026, 9, 22, tzinfo=UTC)


def _usage(**kinds: int) -> dict[str, int]:
    return {
        k: kinds.get(k, 0) for k in ("input", "cache_read", "cache_write", "output")
    }


# -- the join (domain) ----------------------------------------------------------------


@pytest.mark.req("TER-EXP-012")
def test_a_record_takes_its_tokens_of_the_row_kind_as_its_share() -> None:
    rows = [
        BillingRow("m", START, END, "input", 1_000.0),
        BillingRow("m", START, END, "output", 200.0),
        BillingRow("m", START, END, None, 5_000.0),
    ]
    record = RecordFacts({"m": _usage(input=100, output=50, cache_read=10)}, DAY)
    result = join_records(rows, [record])
    first, second, third = result.rows
    assert first is not None and first.record_tokens == 100 and first.token_share == 0.1
    assert second is not None and second.token_share == 0.25
    assert third is not None and third.record_tokens == 160  # a row of every kind
    assert result.joined == 1 and result.unjoined == {}


@pytest.mark.req("TER-EXP-012")
def test_many_records_join_one_row() -> None:
    rows = [BillingRow("m", START, END, "output", 1_000.0)]
    records = [
        RecordFacts(
            {"m": _usage(output=100)}, DAY, 2.0, {"value_adding": 1.5, "avoidable": 0.5}
        ),
        RecordFacts(
            {"m": _usage(output=300)},
            DAY,
            6.0,
            {"value_adding": 3.0, "uncertain": 3.0},
            True,
        ),
    ]
    row = join_records(rows, records).rows[0]
    assert row is not None
    assert (row.records, row.record_tokens, row.token_share) == (2, 400, 0.4)
    assert row.cost_shares == {
        "avoidable": 0.0625,
        "uncertain": 0.375,
        "value_adding": 0.5625,
    }
    assert row.verified_outcomes == 1


@pytest.mark.req("TER-EXP-012")
def test_rows_differing_only_by_principal_share_a_record_by_their_tokens() -> None:
    rows = [
        BillingRow("m", START, END, "input", 300.0, principal="alice"),
        BillingRow("m", START, END, "input", 100.0, principal="bob"),
    ]
    result = join_records(rows, [RecordFacts({"m": _usage(input=40)}, DAY)])
    shares = [r.record_tokens for r in result.rows if r is not None]
    assert shares == [30, 10]  # 40 tokens, counted once, split 3 to 1


@pytest.mark.req("TER-EXP-012")
def test_a_record_joins_only_rows_whose_period_holds_its_day() -> None:
    rows = [
        BillingRow("m", START, END, "input", 10.0),
        BillingRow("m", END, datetime(2026, 9, 23, tzinfo=UTC), "input", 10.0),
        BillingRow(
            "m",
            datetime(2026, 9, 1, tzinfo=UTC),
            datetime(2026, 10, 1, tzinfo=UTC),
            "input",
            10.0,
        ),
    ]
    result = join_records(rows, [RecordFacts({"m": _usage(input=1)}, DAY)])
    assert [r is not None for r in result.rows] == [True, False, True]


@pytest.mark.req("TER-EXP-012")
@pytest.mark.parametrize(
    ("record", "row", "matches"),
    [
        ("claude-sonnet-5-5", "claude-sonnet-5-5", True),
        ("claude-sonnet-5-5-20261001", "claude-sonnet-5-5", True),
        ("claude-sonnet-5-5", "claude-sonnet-5", False),
        ("claude-sonnet-5-5-beta", "claude-sonnet-5-5", False),
        ("claude-sonnet-5", "claude-sonnet-5-5", False),
    ],
)
def test_a_dated_model_bills_as_its_alias_and_nothing_else(
    record: str, row: str, matches: bool
) -> None:
    assert model_matches(record, row) is matches


@pytest.mark.req("TER-EXP-013")
def test_a_record_that_joins_no_row_is_counted_with_its_reason() -> None:
    rows = [BillingRow("m", START, END, "input", 10.0)]
    records = [
        RecordFacts({"other": _usage(input=5)}, DAY),
        RecordFacts({"m": _usage(input=5)}, date(2026, 1, 1)),
        RecordFacts({"m": _usage(input=5)}, None),
        RecordFacts({}, DAY),
    ]
    result = join_records(rows, records)
    assert result.rows == (None,)
    assert result.joined == 0
    assert result.unjoined == {
        "no billing row for its models and charge period": 2,
        "no charge period": 1,
        "no model turns": 1,
    }


def test_without_consumed_tokens_or_priced_records_the_shares_are_empty() -> None:
    rows = [BillingRow("m", START, END, "input", 0.0)]
    row = join_records(rows, [RecordFacts({"m": _usage(input=5)}, DAY)]).rows[0]
    assert row is not None and row.token_share is None and row.cost_shares is None


@pytest.mark.parametrize(
    ("direction", "action", "kind"),
    [
        ("Input", "Uncached", "input"),
        ("Input", "Read", "cache_read"),
        ("Input", "Write", "cache_write"),
        ("Output", None, "output"),
        ("Output", "Read", "output"),
        ("Input", "Other", None),
        ("Input", None, None),
        (None, None, None),
    ],
)
def test_focus_token_labels_name_a_token_kind(
    direction: str | None, action: str | None, kind: str | None
) -> None:
    assert focus_token_kind(direction, action) == kind


@pytest.mark.parametrize(
    ("unit", "count"),
    [
        ("Tokens", 1),
        ("1 Token", 1),
        ("1K Tokens", 1_000),
        ("1m tokens", 1_000_000),
        ("Hours", None),
    ],
)
def test_token_units(unit: str, count: int | None) -> None:
    assert tokens_per_unit(unit) == count


# -- the exporter (adapter and command) ---------------------------------------------------------


BILLING_HEADER = [
    "BilledCost",
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "ConsumedQuantity",
    "ConsumedUnit",
    "SkuId",
    "SkuPriceDetails",
    "PrincipalId",
    "ServiceName",
]


def _details(direction: str, action: str | None = None) -> str:
    d = {"ModelId": "claude-sonnet-5-5", "TokenDirection": direction}
    if action:
        d["TokenCacheAction"] = action
    return json.dumps(d)


def _billing(tmp_path: Path) -> Path:
    rows = [
        [
            "1.20",
            "2026-09-21T00:00:00Z",
            "2026-09-22T00:00:00Z",
            "0.6",
            "1M Tokens",
            "sonnet",
            _details("Input", "Uncached"),
            "svc-a",
            "Claude API",
        ],
        [
            "0.04",
            "2026-09-21T00:00:00Z",
            "2026-09-22T00:00:00Z",
            "400000",
            "Tokens",
            "sonnet",
            _details("Input", "Read"),
            "svc-a",
            "Claude API",
        ],
        [
            "0.05",
            "2026-09-21T00:00:00Z",
            "2026-09-22T00:00:00Z",
            "20",
            "1K Tokens",
            "sonnet",
            _details("Input", "Write"),
            "svc-a",
            "Claude API",
        ],
        [
            "0.40",
            "2026-09-21T00:00:00Z",
            "2026-09-22T00:00:00Z",
            "40000",
            "Tokens",
            "sonnet",
            _details("Output"),
            "svc-a",
            "Claude API",
        ],
        [
            "9.99",
            "2026-09-21T00:00:00Z",
            "2026-09-22T00:00:00Z",
            "1",
            "Hours",
            "gpu",
            "",
            "",
            "Compute",
        ],
    ]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(BILLING_HEADER)
    writer.writerows(rows)
    path = tmp_path / "billing.csv"
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def _spans(tmp_path: Path) -> Path:
    def attr(key: str, value: object) -> dict[str, Any]:
        if isinstance(value, int):
            return {"key": key, "value": {"intValue": str(value)}}
        return {"key": key, "value": {"stringValue": str(value)}}

    spans = [
        {
            "traceId": "b" * 32,
            "spanId": f"{n:016x}",
            "startTimeUnixNano": str((1_790_000_000 + n) * 10**9),
            "endTimeUnixNano": str((1_790_000_000 + n) * 10**9 + 1),
            "attributes": [
                attr("gen_ai.operation.name", "chat"),
                attr("gen_ai.request.model", "claude-sonnet-5-5"),
                attr("gen_ai.usage.input_tokens", 120_000),
                attr("gen_ai.usage.output_tokens", 2_000),
                attr("gen_ai.usage.cache_read.input_tokens", 100_000),
                attr("gen_ai.usage.cache_creation.input_tokens", 10_000),
            ],
        }
        for n in (1, 2)
    ]
    path = tmp_path / "spans.json"
    scope = {"schemaUrl": "https://opentelemetry.io/schemas/1.45.0", "spans": spans}
    path.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [scope]}]}), encoding="utf-8"
    )
    return path


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(list(argv), bootstrap.cli_services(), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def record(tmp_path: Path) -> Path:
    path = tmp_path / "r.json"
    code, _, err = _run(
        "record", "export", str(_spans(tmp_path)), "--ter", "off", "--out", str(path)
    )
    assert code == 0, err
    return path


@pytest.mark.req("TER-EXP-011")
def test_the_exporter_adds_only_x_columns_and_changes_no_billing_value(
    tmp_path: Path, record: Path
) -> None:
    billing = _billing(tmp_path)
    joined = export_focus(billing, [record])
    original = list(csv.DictReader(io.StringIO(billing.read_text(encoding="utf-8"))))
    written = list(csv.DictReader(io.StringIO(joined.text)))
    assert list(written[0]) == [*BILLING_HEADER, *EFFICIENCY_COLUMNS]
    assert all(c.startswith("x_") for c in EFFICIENCY_COLUMNS)
    assert len(written) == len(original)
    for before, after in zip(original, written):
        assert {k: after[k] for k in BILLING_HEADER} == before


@pytest.mark.req("TER-EXP-012")
def test_a_record_joins_its_model_rows_by_kind_and_leaves_the_rest_empty(
    tmp_path: Path, record: Path
) -> None:
    joined = export_focus(_billing(tmp_path), [record])
    rows = list(csv.DictReader(io.StringIO(joined.text)))
    # Two turns: 2 x (10,000 uncached input, 100,000 read, 10,000 write, 2,000 out).
    assert [r["x_TerRecordTokens"] for r in rows] == [
        "20000",
        "200000",
        "20000",
        "4000",
        "",
    ]
    assert [r["x_TerTokenShare"] for r in rows[:4]] == ["0.033333", "0.5", "1", "0.1"]
    assert {r["x_TerRecordCount"] for r in rows[:4]} == {"1"}
    assert rows[0]["x_TerVerifiedOutcomes"] == "0"
    shares = [float(rows[0][c]) for c in EFFICIENCY_COLUMNS if c.endswith("CostShare")]
    assert sum(shares) == pytest.approx(1.0)
    assert joined.rows_joined == 4 and joined.result.joined == 1


@pytest.mark.req("TER-EXP-011")
def test_the_record_focus_command_writes_the_joined_file(
    tmp_path: Path, record: Path
) -> None:
    out = tmp_path / "joined.csv"
    code, _, err = _run(
        "record",
        "focus",
        str(record),
        "--billing",
        str(_billing(tmp_path)),
        "--out",
        str(out),
    )
    assert code == 0, err
    assert "Joined 1 of 1 record(s) to 4 of 5 billing row(s)" in err
    assert (
        out.read_text(encoding="utf-8")
        .splitlines()[0]
        .endswith("x_TerVerifiedOutcomes")
    )


@pytest.mark.req("TER-EXP-013")
def test_the_command_says_which_records_joined_nothing(
    tmp_path: Path, record: Path
) -> None:
    billing = _billing(tmp_path)
    text = (
        billing.read_text(encoding="utf-8")
        .replace("2026-09-21T", "2026-08-21T")
        .replace("2026-09-22T", "2026-08-22T")
    )
    billing.write_text(text, encoding="utf-8")
    code, out, err = _run("record", "focus", str(record), "--billing", str(billing))
    assert code == 0, err
    assert "not joined: 1 no billing row for its models and charge period" in err
    assert all(line.endswith(",,,,,,,,") for line in out.splitlines()[1:])


def test_a_billing_file_missing_a_join_column_is_refused(
    tmp_path: Path, record: Path
) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("SkuId,BilledCost\nx,1\n", encoding="utf-8")
    with pytest.raises(FocusExportError, match="ChargePeriodStart"):
        export_focus(bad, [record])
    code, _, err = _run("record", "focus", str(record), "--billing", str(bad))
    assert code == 2 and "Cannot join" in err


def test_a_file_already_joined_is_refused(tmp_path: Path, record: Path) -> None:
    once = tmp_path / "once.csv"
    once.write_text(export_focus(_billing(tmp_path), [record]).text, encoding="utf-8")
    with pytest.raises(FocusExportError, match="x_TerRecordCount"):
        export_focus(once, [record])


@pytest.mark.parametrize(
    "doc",
    [
        "not json",
        json.dumps({"schema": "other"}),
        json.dumps({"schema": "ter.record/0.1", "usage": {}}),
        json.dumps({"schema": "ter.record/0.1", "usage": {"by_model": [1]}}),
        json.dumps({"schema": "ter.record/0.1", "usage": {"by_model": [{"model": 3}]}}),
    ],
    ids=["text", "other-schema", "no-by-model", "not-object", "model-not-string"],
)
def test_a_record_that_cannot_be_joined_is_refused(tmp_path: Path, doc: str) -> None:
    path = tmp_path / "r.json"
    path.write_text(doc, encoding="utf-8")
    with pytest.raises(FocusExportError):
        export_focus(_billing(tmp_path), [path])


def test_missing_files_are_reported(tmp_path: Path, record: Path) -> None:
    code, _, err = _run(
        "record", "focus", str(record), "--billing", str(tmp_path / "no.csv")
    )
    assert code == 2 and "No such file" in err


# -- review findings (Qodo, PR #115) --------------------------------------------------


@pytest.mark.req("TER-EXP-012")
def test_hourly_rows_share_a_record_by_their_tokens() -> None:
    """Sub-day rows each take a share of the day's Record, never the whole of
    it on the midnight row (finding 10)."""
    hour = datetime(2026, 9, 21, 1, tzinfo=UTC)
    rows = [
        BillingRow("m", START, hour, "input", 100.0),
        BillingRow("m", hour, datetime(2026, 9, 21, 2, tzinfo=UTC), "input", 300.0),
        BillingRow("m", datetime(2026, 9, 20, 23, tzinfo=UTC), START, "input", 50.0),
        BillingRow("m", END, datetime(2026, 9, 22, 1, tzinfo=UTC), "input", 50.0),
    ]
    result = join_records(rows, [RecordFacts({"m": _usage(input=40)}, DAY)])
    assert [None if r is None else r.record_tokens for r in result.rows] == [
        10,
        30,
        None,
        None,
    ]


@pytest.mark.req("TER-EXP-012")
def test_a_dated_model_joins_its_own_rows_before_its_aliases() -> None:
    """A Record's tokens are never counted on both the dated and the alias
    row (finding 11)."""
    dated, alias = "claude-x-1-20261001", "claude-x-1"
    rows = [
        BillingRow(dated, START, END, "output", 100.0),
        BillingRow(alias, START, END, "output", 100.0),
    ]
    result = join_records(rows, [RecordFacts({dated: _usage(output=10)}, DAY)])
    first, second = result.rows
    assert first is not None and first.record_tokens == 10
    assert second is None
    only_alias = join_records(rows[1:], [RecordFacts({dated: _usage(output=10)}, DAY)])
    assert only_alias.rows[0] is not None and only_alias.rows[0].record_tokens == 10


@pytest.mark.req("TER-EXP-012")
def test_two_models_of_one_record_on_one_row_count_the_record_once() -> None:
    rows = [BillingRow("claude-x-1", START, END, "output", 100.0)]
    record = RecordFacts(
        {"claude-x-1": _usage(output=4), "claude-x-1-20261001": _usage(output=6)}, DAY
    )
    row = join_records(rows, [record]).rows[0]
    assert row is not None and (row.records, row.record_tokens) == (1, 10)


@pytest.mark.req("TER-EXP-013")
@pytest.mark.parametrize("unit", [", Tokens", ",,1 Tokens", ".5 Tokens", "1,, Tokens"])
def test_a_malformed_token_unit_is_not_a_unit(unit: str) -> None:
    """Finding 4: a comma-only count raised ValueError."""
    assert tokens_per_unit(unit) in (None, 1) and tokens_per_unit(", Tokens") is None


def _write_billing(tmp_path: Path, lines: list[list[str]], name: str = "b.csv") -> Path:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(BILLING_HEADER)
    writer.writerows(lines)
    path = tmp_path / name
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def _line(details: str, quantity: str = "1000", unit: str = "Tokens") -> list[str]:
    return [
        "1.00",
        "2026-09-21T00:00:00Z",
        "2026-09-22T00:00:00Z",
        quantity,
        unit,
        "claude-sonnet-5-5",
        details,
        "",
        "Claude API",
    ]


@pytest.mark.req("TER-EXP-013")
@pytest.mark.parametrize("details", ['{"ModelId": ', "[1, 2]", "not json"])
def test_billing_labels_that_cannot_be_read_leave_the_row_unjoined(
    tmp_path: Path, record: Path, details: str
) -> None:
    """Finding 5: unreadable labels are not a row of every kind."""
    billing = _write_billing(tmp_path, [_line(details), _line("")])
    joined = export_focus(billing, [record])
    rows = list(csv.DictReader(io.StringIO(joined.text)))
    assert rows[0]["x_TerRecordCount"] == ""
    assert rows[1]["x_TerRecordCount"] == "1"  # no labels: every kind, by SkuId
    assert joined.rows_joined == 1


@pytest.mark.req("TER-EXP-011")
def test_a_row_with_more_cells_than_the_header_is_refused(
    tmp_path: Path, record: Path
) -> None:
    """Finding 6: writing it back would drop the surplus cell."""
    path = tmp_path / "wide.csv"
    text = _write_billing(tmp_path, [_line("")]).read_text(encoding="utf-8")
    path.write_text(text.rstrip("\n") + ",surplus\n", encoding="utf-8")
    with pytest.raises(FocusExportError, match="line 2 has more cells than the header"):
        export_focus(path, [record])


@pytest.mark.req("TER-EXP-011")
def test_file_errors_are_reported_not_raised(tmp_path: Path, record: Path) -> None:
    """Finding 7: an unreadable billing file or unwritable output is a
    command error, not a traceback."""
    folder = tmp_path / "folder.csv"
    folder.mkdir()
    with pytest.raises(FocusExportError, match="cannot read"):
        export_focus(folder, [record])
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code, _, err = _run(
        "record",
        "focus",
        str(record),
        "--billing",
        str(_billing(tmp_path)),
        "--out",
        str(blocker / "out.csv"),
    )
    assert code == 2 and err.startswith(f"Cannot write {blocker / 'out.csv'}:")
