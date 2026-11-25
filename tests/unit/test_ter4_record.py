"""The TER Record (``ter.record/0.1``, ADR 0007, TER-REC-001 to TER-REC-006)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from ter.adapters.driven.calibration import default_calibration
from ter.adapters.driven.in_memory import InMemoryCalibration, InMemoryPriceBook
from ter.adapters.driven.json_schemas import schema
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import Event, PriceEntry, Rates, TokenUsage
from ter.domain.calibration import Calibration, JudgedCounts, JudgedSample
from ter.domain.lean import build_a3
from ter.domain.lean.analysis import LeanAnalysis, explain
from ter.domain.lean.control import detector_fingerprint
from ter.domain.outcome import (
    CheckEvidence,
    CheckStatus,
    OutcomeEvidence,
    OutcomeVerdict,
    judge,
)
from ter.domain.ter_record import (
    RECORD_SCHEMA,
    build_record,
    events_digest,
    record_differences,
)

from tests.golden.corpus import CORPUS
from tests.golden.test_record_snapshots import record as corpus_record

from ter4_lean_builder import Script

import jsonschema

ENTRIES = [PriceEntry("m", date(2020, 1, 1), Rates(1.0, 10.0, 0.1, 1.25))]
BOOK = InMemoryPriceBook(ENTRIES)
USAGE = TokenUsage(input_tokens=1_000_000, output_tokens=10_000, model="m")
SECRET_PROMPT = "please refactor the zanzibar billing module carefully"
SECRET_PATH = "src/zanzibar/billing_secret.py"
SECRET_OUTPUT = "def quokka_total(rows): return sum(rows)"


def _events() -> list[Event]:
    """A session with an unused read (an uncertain finding) and usage."""
    s = Script()
    s.prompt(SECRET_PROMPT)
    s.think("look at the module first")  # 1
    s.read(SECRET_PATH, SECRET_OUTPUT)  # 2, 3
    s.read("src/zanzibar/unused_helper.py", "def wombat(): pass")  # 4, 5
    s.think("now edit the billing module")  # 6
    s.edit(SECRET_PATH)  # 7, 8
    s.say("Refactored quokka_total in billing_secret.py.")  # 9
    return [
        replace(e, usage=USAGE) if i in (1, 6, 9) else e for i, e in enumerate(s.events)
    ]


def _record(
    analysis: LeanAnalysis,
    events: list[Event],
    *,
    outcome: OutcomeVerdict | None = None,
    calibration: Calibration | None = None,
    prices: Any = BOOK,
) -> dict[str, Any]:
    a3 = build_a3(analysis, [SECRET_PROMPT], outcome, prices=prices)
    doc = build_record(
        a3,
        events,
        source="claude-code-jsonl",
        coverage=1.0,
        event_schema="ter.event/0.5",
        tokenizer="regex",
        detectors=detector_fingerprint(analysis),
        calibration=calibration or default_calibration().calibration(),
    )
    loaded: dict[str, Any] = json.loads(json.dumps(doc))
    return loaded


def _validator() -> jsonschema.Draft7Validator:
    return jsonschema.Draft7Validator(schema(RECORD_SCHEMA))


@pytest.fixture(scope="module")
def session() -> tuple[LeanAnalysis, list[Event]]:
    events = _events()
    return explain(events, RegexTokenizer()), events


@pytest.mark.req("TER-REC-001")
class TestRecordContents:
    def test_holds_every_block_and_conforms(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        doc = _record(analysis, events)
        assert doc["schema"] == RECORD_SCHEMA
        assert list(_validator().iter_errors(doc)) == []
        sc = analysis.scorecard
        assert doc["work"]["generated_tokens"] == sc.generated_tokens
        assert doc["work"]["activity_tokens"] == {k: v for k, v in sc.activity_tokens}
        assert doc["usage"] == {
            "model_turns": 3,
            "input": 3_000_000,
            "cache_read": 0,
            "cache_write": 0,
            "output": 30_000,
            "by_model": [
                {
                    "model": "m",
                    "model_turns": 3,
                    "input": 3_000_000,
                    "cache_read": 0,
                    "cache_write": 0,
                    "output": 30_000,
                }
            ],
        }
        assert doc["cost"]["usd"] == pytest.approx(3.3)
        assert sum(doc["cost"]["by_activity_class"].values()) == pytest.approx(3.3)
        assert doc["shape"]["observed_class"] == "T(n·k)"
        assert doc["join"]["models"] == ["m"]

    def test_without_prices_the_cost_is_null(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        doc = _record(analysis, events, prices=None)
        assert doc["cost"] is None
        assert list(_validator().iter_errors(doc)) == []

    def test_findings_are_counted_per_detector(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        doc = _record(analysis, events)
        detectors = [f["detector"] for f in doc["findings"]]
        assert detectors == sorted(set(f.detector for f in analysis.findings))
        for row in doc["findings"]:
            mine = [f for f in analysis.findings if f.detector == row["detector"]]
            assert row["findings"] == len(mine)
            assert row["uncertain_findings"] == sum(f.uncertain for f in mine)

    def test_events_digest_changes_with_any_event(self) -> None:
        events = _events()
        changed = list(events)
        changed[2] = replace(changed[2], text=changed[2].text + " ")
        assert events_digest(events) == events_digest(list(events))
        assert events_digest(events) != events_digest(changed)

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_every_corpus_record_conforms(self, name: str) -> None:
        assert list(_validator().iter_errors(corpus_record(name))) == []


@pytest.mark.req("TER-REC-002")
class TestNoContent:
    def test_no_prompt_path_or_output_reaches_the_record(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        text = json.dumps(_record(analysis, events), ensure_ascii=False)
        for secret in ("zanzibar", "quokka", "wombat", "billing_secret", "refactor"):
            assert secret not in text

    def test_a_path_named_price_book_is_reported_as_custom(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        book = InMemoryPriceBook(ENTRIES)
        book.name = "C:/Users/someone/prices/book.json"  # type: ignore[misc]
        doc = _record(analysis, events, prices=book)
        assert doc["cost"]["price_book"] == "custom"
        book.name = "ter:data/price_book.json"  # type: ignore[misc]
        assert _record(analysis, events, prices=book)["cost"]["price_book"] == (
            "ter:data/price_book.json"
        )

    def test_the_schema_refuses_a_finding_with_a_title(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        doc = _record(analysis, events)
        doc["findings"][0]["title"] = "Read src/zanzibar/unused_helper.py"
        assert list(_validator().iter_errors(doc))

    def test_the_schema_refuses_a_path_as_a_model_name(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        doc = _record(analysis, events)
        doc["join"]["models"] = ["see /home/someone/notes.txt please"]
        assert list(_validator().iter_errors(doc))

    def test_negative_check_names_stay_out_of_the_outcome(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        evidence = OutcomeEvidence(
            "run",
            "junit",
            (CheckEvidence("tests/test_zanzibar.py::test_x", CheckStatus.PASSED, "j"),),
        )
        doc = _record(analysis, events, outcome=judge(evidence))
        assert "zanzibar" not in json.dumps(doc)

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_no_string_of_a_corpus_session_reaches_its_record(self, name: str) -> None:
        from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource

        trace = ClaudeCodeJsonlSource().read(CORPUS[name])
        text = json.dumps(corpus_record(name), ensure_ascii=False)
        strings: set[str] = set()
        for e in trace.events:
            strings |= {line.strip() for line in e.text.splitlines()}
            if e.tool is not None:
                strings |= {
                    v.strip() for v in e.tool.arguments.values() if isinstance(v, str)
                }
        leaked = sorted(s for s in strings if len(s) >= 8 and s in text)
        assert not leaked


@pytest.mark.req("TER-REC-003")
class TestUncertainNeverAvoidable:
    def test_uncertain_finding_tokens_and_cost_are_uncertain(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        doc = _record(analysis, events)
        unsure = [f for f in doc["findings"] if f["uncertain_findings"]]
        assert unsure, "the unused read should be an uncertain finding"
        for row in unsure:
            if row["uncertain_findings"] == row["findings"]:
                assert row["avoidable_tokens"] == 0
        claimed = sum(f["uncertain_tokens"] for f in doc["findings"])
        assert claimed == doc["work"]["activity_tokens"]["uncertain"]
        assert (doc["cost"]["by_activity_class"]["uncertain"] > 0) == (claimed > 0)

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_detector_tokens_add_up_to_the_scorecard(self, name: str) -> None:
        doc = corpus_record(name)
        work = doc["work"]["activity_tokens"]
        avoid = sum(f["avoidable_tokens"] for f in doc["findings"])
        unsure = sum(f["uncertain_tokens"] for f in doc["findings"])
        assert abs(avoid - work["avoidable"]) <= len(doc["findings"])
        assert abs(unsure - work["uncertain"]) <= len(doc["findings"])


@pytest.mark.req("TER-REC-004")
class TestCalibration:
    def test_each_detector_carries_its_judged_samples(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        detector = analysis.findings[0].detector
        samples = [
            JudgedSample("a", date(2026, 1, 1), 1, {detector: JudgedCounts(3, 1, 1)}),
            JudgedSample("b", date(2026, 2, 1), 2, {detector: JudgedCounts(2, 0, 0)}),
        ]
        doc = _record(
            analysis,
            events,
            calibration=InMemoryCalibration(samples).calibration(),
        )
        row = next(f for f in doc["findings"] if f["detector"] == detector)
        assert row["calibration"] == {
            "judged": 7,
            "waste": 5,
            "not_waste": 1,
            "unsure": 1,
            "samples": 2,
        }

    def test_negative_a_never_judged_detector_reports_zero(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        doc = _record(analysis, events, calibration=Calibration())
        assert all(f["calibration"]["judged"] == 0 for f in doc["findings"])

    def test_shipped_samples_back_the_advice_eligible_detectors(self) -> None:
        calibration = default_calibration().calibration()
        assert calibration.of("fragmented_edits").waste >= 17
        assert calibration.of("repeated_exploration").judged >= 10
        assert calibration.of("rework_cycle").judged == 0


@pytest.mark.req("TER-REC-006")
class TestOutcome:
    def test_no_evidence_is_unknown_with_no_per_outcome_figure(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        outcome = _record(analysis, events)["outcome"]
        assert outcome["status"] == "unknown"
        assert outcome["generated_tokens_per_verified_outcome"] is None
        assert outcome["required_checks"] is None

    def test_an_accepted_run_gives_tokens_per_verified_outcome(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        evidence = OutcomeEvidence(
            "run", "junit", (CheckEvidence("t", CheckStatus.PASSED, "j"),)
        )
        outcome = _record(analysis, events, outcome=judge(evidence))["outcome"]
        assert outcome["status"] == "accepted"
        assert outcome["passed_checks"] == 1 and outcome["failed_checks"] == 0
        assert outcome["generated_tokens_per_verified_outcome"] == pytest.approx(
            analysis.scorecard.generated_tokens, abs=0.1
        )

    def test_a_rejected_run_has_no_per_outcome_figure(
        self, session: tuple[LeanAnalysis, list[Event]]
    ) -> None:
        analysis, events = session
        evidence = OutcomeEvidence(
            "run", "junit", (CheckEvidence("t", CheckStatus.FAILED, "j"),)
        )
        outcome = _record(analysis, events, outcome=judge(evidence))["outcome"]
        assert outcome["status"] == "rejected"
        assert outcome["failed_checks"] == 1
        assert outcome["generated_tokens_per_verified_outcome"] is None


@pytest.mark.req("TER-REC-005")
class TestDifferences:
    def test_equal_records_have_no_difference(self) -> None:
        doc = corpus_record("lean_mix")
        assert record_differences(doc, json.loads(json.dumps(doc))) == []

    def test_each_changed_figure_is_listed_by_path(self) -> None:
        doc = corpus_record("lean_mix")
        changed = json.loads(json.dumps(doc))
        changed["work"]["generated_tokens"] += 1
        changed["findings"][0]["findings"] = 99
        changed["findings"].append({"detector": "extra"})
        del changed["shape"]
        paths = [p for p, _, _ in record_differences(doc, changed)]
        assert "work.generated_tokens" in paths
        assert "findings[0].findings" in paths
        assert f"findings[{len(doc['findings'])}]" in paths
        assert "shape" in paths


@pytest.mark.req("TER-EXP-012")
def test_usage_is_split_by_the_model_that_served_each_turn() -> None:
    """Each model's tokens, so a billing join can give each model its share;
    turns of no named model sort last under null (ADR 0008)."""
    from ter.domain.ter_record import _usage

    class Step:
        def __init__(self, usage: TokenUsage | None) -> None:
            self.usage = usage

    class Analysis:
        steps = [
            Step(TokenUsage(input_tokens=10, output_tokens=1, model="b")),
            Step(None),
            Step(TokenUsage(input_tokens=5, cache_read_tokens=7, model=None)),
            Step(TokenUsage(input_tokens=20, cache_creation_tokens=3, model="a")),
            Step(TokenUsage(input_tokens=1, output_tokens=2, model="b")),
        ]

    class A3:
        analysis = Analysis()

    usage = _usage(A3())  # type: ignore[arg-type]
    assert usage["model_turns"] == 4 and usage["input"] == 36
    assert usage["by_model"] == [
        {
            "model": "a",
            "model_turns": 1,
            "input": 20,
            "cache_read": 0,
            "cache_write": 3,
            "output": 0,
        },
        {
            "model": "b",
            "model_turns": 2,
            "input": 11,
            "cache_read": 0,
            "cache_write": 0,
            "output": 3,
        },
        {
            "model": None,
            "model_turns": 1,
            "input": 5,
            "cache_read": 7,
            "cache_write": 0,
            "output": 0,
        },
    ]


@pytest.mark.req("TER-EXP-012")
def test_by_model_is_in_model_order_whatever_order_the_turns_came_in() -> None:
    """Eight models served out of order: the list is sorted by name, null
    last, never in the order a set happens to give."""
    from ter.domain.ter_record import _usage

    names = ["h", "c", "f", "a", "g", "d", "b", "e"]

    class Step:
        def __init__(self, model: str | None) -> None:
            self.usage = TokenUsage(input_tokens=1, model=model)

    class Analysis:
        steps = [Step(None), *(Step(n) for n in names)]

    class A3:
        analysis = Analysis()

    usage = _usage(A3())  # type: ignore[arg-type]
    assert [m["model"] for m in usage["by_model"]] == [*sorted(names), None]
