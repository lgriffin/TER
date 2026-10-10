"""Persistent intent, intent changes, alignment, low-alignment periods and
drift (TER-ITN-001 to 005) and value judged against the intent (TER-LEN-003).

Positive, negative and boundary cases for the ``intent_drift`` detector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ter4_lean_builder import PASS, Script

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driving.reports import render_a3_html
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import AnalysisEngine, EventKind, ToolKind, explain_batch
from ter.domain.lean.grounding import RepositoryGrounding
from ter.domain.lean import (
    ActivityClass,
    AlignmentBand,
    Finding,
    IntentConfig,
    IntentRelation,
    LeanAnalysis,
    LeanWaste,
    build_a3,
    explain,
    key_terms,
)

TESTS = Path(__file__).resolve().parents[1]
CORPUS = {
    "intent_shift": TESTS / "golden" / "sessions" / "intent_shift.jsonl",
    "example_session": TESTS.parent / "sample_sessions" / "example_session.jsonl",
}

MEDIAN = (
    "Write a function in src/stats.py that returns the median of a list of numbers."
)
MODE = (
    "Actually, forget the median. Instead add a mode function that returns the "
    "most common value, and raise ValueError on an empty list."
)
MEAN_EDIT = (
    "def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
    "def variance(xs):\n    m = mean(xs)\n    return sum((x - m) ** 2 for x in xs) / len(xs)\n"
)
MODE_EDIT = (
    "def mode(xs):\n    if not xs:\n        raise ValueError('mode of empty list')\n"
    "    from collections import Counter\n    return Counter(xs).most_common(1)[0][0]\n"
)


def run(script: Script, **kw: Any) -> LeanAnalysis:
    return explain(script.events, RegexTokenizer(), **kw)


def drift(script: Script, **kw: Any) -> list[Finding]:
    return [f for f in run(script, **kw).findings if f.detector == "intent_drift"]


def golden(name: str = "intent_shift") -> LeanAnalysis:
    trace = ClaudeCodeJsonlSource().read(CORPUS[name])
    return explain_batch(trace.events, RegexTokenizer())


# --- key terms ---------------------------------------------------------------


@pytest.mark.req("TER-ITN-004")
def test_key_terms_split_identifiers_and_drop_generic_words() -> None:
    terms = key_terms("Raise ValueError in most_common(); parsing parsed parses")
    assert {"rais", "valu", "error", "common", "pars"} <= terms
    assert not terms & {"in", "most", "the", "def", "function"}
    assert key_terms("retries") == key_terms("retry")


# --- TER-ITN-001: one persistent record, updated by every prompt --------------


@pytest.mark.req("TER-ITN-001")
def test_every_prompt_updates_the_one_intent_record() -> None:
    s = Script()
    first = s.prompt("Add a slugify helper to src/text.py")
    s.edit("src/text.py", "", "def slugify(text): ...")
    second = s.prompt("Make slugify also strip leading dashes")
    third = s.prompt("ok")
    fourth = s.prompt("New task: write release notes for the changelog")
    a = run(s)
    record = a.intent.record
    assert [r.event_id for r in record.revisions] == [
        first.id,
        second.id,
        third.id,
        fourth.id,
    ]
    assert [r.relation for r in record.revisions] == [
        IntentRelation.OPENED,
        IntentRelation.REFINED,
        IntentRelation.ACKNOWLEDGED,
        IntentRelation.CHANGED,
    ]
    assert [r.revision for r in record.revisions] == [1, 2, 3, 4]
    # A refinement keeps what was asked and adds to it; an acknowledgement keeps it.
    assert record.revision(1).terms < record.revision(2).terms
    assert record.revision(3).terms == record.revision(2).terms
    assert record.current is record.revisions[-1]
    assert "slugify" not in record.current.terms


@pytest.mark.req("TER-ITN-001")
def test_the_intent_record_is_kept_with_the_session_state_and_redelivery_is_idempotent() -> (
    None
):
    s = Script()
    s.prompt(MEDIAN)
    s.write("src/stats.py", "def median(xs): ...")
    s.prompt(MODE)
    engine = AnalysisEngine(RegexTokenizer())
    for event in s.events:
        engine.apply(event)
        engine.apply(event)  # redelivered: no second revision
    live = engine.explain().intent
    assert len(live.record.revisions) == 2
    assert live == run(s).intent


@pytest.mark.req("TER-ITN-001")
def test_a_session_without_prompts_has_an_empty_record() -> None:
    s = Script()
    s.edit("src/a.py", "", "def helper(): ...")
    a = run(s)
    assert a.intent.record.revisions == () and a.intent.record.current is None
    [aligned] = a.intent.alignments
    assert aligned.revision is None and aligned.band is AlignmentBand.UNKNOWN


# --- TER-ITN-002: intent changes carry the prompt that caused them ------------


@pytest.mark.req("TER-ITN-002")
def test_an_explicit_redirect_records_a_change_with_its_prompt() -> None:
    s = Script()
    s.prompt(MEDIAN)
    redirect = s.prompt(MODE)
    [change] = run(s).intent.changes
    assert change.event_id == redirect.id
    assert (change.from_revision, change.to_revision) == (1, 2)
    assert "Actually" in change.reason
    assert change.abandoned == frozenset({"median"})


@pytest.mark.req("TER-ITN-002")
def test_an_unrelated_goal_without_a_marker_is_a_change() -> None:
    s = Script()
    s.prompt("Fix the retry decorator in src/net.py so backoff doubles")
    new = s.prompt("Write release notes for version two in the changelog")
    a = run(s)
    [change] = a.intent.changes
    assert change.event_id == new.id and "unrelated goal" in change.reason
    assert a.intent.record.current is not None
    assert "backoff" not in a.intent.record.current.terms


@pytest.mark.req("TER-ITN-002")
@pytest.mark.parametrize(
    "follow_up",
    [
        "Keep the retry decorator backoff but cap the doubling",  # related: refines
        "Actually, make the retry backoff doubling start at one second",  # marker, same goal
        "thanks",  # no goal stated
    ],
)
def test_related_follow_ups_and_acknowledgements_are_not_changes(
    follow_up: str,
) -> None:
    s = Script()
    s.prompt("Fix the retry decorator in src/net.py so backoff doubles")
    s.prompt(follow_up)
    assert run(s).intent.changes == ()


@pytest.mark.req("TER-ITN-002", "TER-ITN-003")
def test_golden_intent_shift_records_the_change_and_the_drift() -> None:
    a = golden()
    prompts = [s.event_id for s in a.steps if s.kind is EventKind.PROMPT]
    [change] = a.intent.changes
    assert change.event_id == prompts[1]
    [f] = a.drift_findings
    assert f.waste is LeanWaste.OVERPRODUCTION and not f.uncertain
    assert "mean" in f.title and "variance" in f.title
    assert prompts[1] in f.evidence
    edit = a.intent.alignment_of(f.waste_events[0])
    assert (
        edit is not None and edit.revision == 2 and edit.names == ("mean", "variance")
    )
    # The mode edit, on the new intent, is not drift.
    names = [x.names for x in a.intent.alignments if x.names]
    assert ("mode",) in names


# --- TER-ITN-003: drift is departure without a recorded change ----------------


def _shift(
    *, redirect: str, announce: bool, edit: str = MEAN_EDIT, created: bool = True
) -> Script:
    s = Script()
    s.prompt(MEDIAN)
    if created:
        s.write("src/stats.py", "def median(xs): ...")
    else:
        s.read("src/stats.py", "")
        s.edit("src/stats.py", "", "def median(xs): ...")
    s.prompt(redirect)
    if announce:
        s.think("I will also add a mean and a variance function to src/stats.py.")
    s.edit("src/stats.py", "def median(xs):", edit + "\ndef median(xs):")
    s.edit("src/stats.py", "def median(xs):", MODE_EDIT + "\ndef median(xs):")
    s.bash("pytest -q", PASS)
    s.say("Added mode().")
    return s


@pytest.mark.req("TER-ITN-003")
class TestIntentDrift:
    def test_unrequested_new_names_alone_are_uncertain_drift(self) -> None:
        # Real sessions showed new names alone are no evidence: implementing
        # anything defines them, so the finding is shown but never counted.
        [f] = drift(_shift(redirect=MODE, announce=False, created=False))
        assert f.confidence == 0.55 and f.uncertain
        assert f.kind.value == "waste" and f.activity_class is ActivityClass.AVOIDABLE
        assert len(f.waste_events) == 2  # the edit and its result

    def test_agent_announcing_extra_work_is_more_certain(self) -> None:
        s = _shift(redirect=MODE, announce=True)
        [f] = drift(s)
        assert f.confidence == 0.85
        thought = next(e for e in s.events if e.kind is EventKind.REASONING)
        assert thought.id in f.evidence and thought.id not in f.waste_events

    def test_continuing_a_dropped_goal_is_more_certain(self) -> None:
        s = Script()
        s.prompt(MEDIAN)
        s.prompt(MODE)
        s.edit("src/stats.py", "", "def median_sorted(xs): ...")
        [f] = drift(s)
        assert f.confidence == 0.85 and "dropped" in f.explanation

    def test_abandonment_only_prompt_drops_the_goal(self) -> None:
        # Too short to state a new goal, but it still abandons "median".
        s = Script()
        s.prompt(MEDIAN)
        s.prompt("Forget the median.")
        s.edit("src/stats.py", "", "def median_sorted(xs): ...")
        a = run(s)
        current = a.intent.record.current
        assert current is not None
        assert current.relation is IntentRelation.ACKNOWLEDGED
        assert "median" not in current.terms and "median" in current.abandoned
        [edit] = a.intent.alignments
        assert edit.band is AlignmentBand.LOW
        [f] = drift(s)
        assert f.confidence == 0.85 and "dropped" in f.explanation

    def test_departure_after_a_recorded_change_is_not_drift(self) -> None:
        s = _shift(
            redirect="Actually, forget the median. Add mean and variance functions, "
            "and a mode function that raises ValueError on an empty list.",
            announce=True,
        )
        assert drift(s) == []

    def test_aligned_edit_is_not_drift(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.edit("src/stats.py", "", MODE_EDIT)
        assert drift(s) == []

    def test_edit_without_names_is_uncertain(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.edit("src/stats.py", "x = 1", "logger.warning('deprecated path called')")
        [f] = drift(s)
        assert f.uncertain and f.confidence == 0.55

    def test_too_few_changed_words_is_no_finding(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.edit("src/stats.py", "x = 1", "x = int(y)")
        assert drift(s) == []

    def test_terse_intent_is_never_judged(self) -> None:
        s = Script()
        s.prompt("fix stats")
        s.edit("src/stats.py", "", MEAN_EDIT)
        assert drift(s) == []

    def test_edit_before_any_prompt_is_never_drift(self) -> None:
        s = Script()
        s.edit("src/stats.py", "", MEAN_EDIT)
        assert drift(s) == []

    def test_boundary_score_at_the_drift_band_is_not_drift(self) -> None:
        s = Script()
        s.prompt("Add mean and median and mode and range helpers")
        s.edit(
            "src/stats.py",
            "",
            "def mean(xs): ...\ndef variance(xs): ...\n"
            "def spread(xs): ...\ndef skew(xs): ...",
        )
        a = run(s)
        edit = next(x for x in a.intent.alignments if x.names)
        assert edit.score == pytest.approx(0.25)
        assert [f for f in a.findings if f.detector == "intent_drift"] == []
        tighter = run(s, intent_config=IntentConfig(drift_below=0.26))
        assert [f.detector for f in tighter.findings].count("intent_drift") == 1


@pytest.mark.req("TER-ITN-007")
class TestDriftInCreatedFiles:
    """New names alone in a file the session created are not drift."""

    def test_new_names_in_a_file_the_session_wrote_first_are_not_drift(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.write("src/extra.py", MEAN_EDIT)
        s.edit("src/extra.py", "def mean(xs):", "def mean(xs, *, weights=None):")
        s.edit(
            "src/extra.py",
            "def variance(xs):",
            "def spread(xs): ...\ndef variance(xs):",
        )
        assert drift(s) == []

    def test_new_names_in_a_file_the_session_read_first_are_still_drift(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.read("src/extra.py", "")
        s.write("src/extra.py", MEAN_EDIT)
        [f] = drift(s)
        assert f.uncertain and f.confidence == 0.55

    def test_a_created_file_still_drifts_when_the_agent_calls_it_extra(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.think("I will also add a mean and a variance function to src/extra.py.")
        s.write("src/extra.py", MEAN_EDIT)
        [f] = drift(s)
        assert f.confidence == 0.85

    def test_a_refused_first_write_does_not_create_the_file(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.write(
            "src/extra.py",
            "",
            output="<tool_use_error>File has not been read yet.</tool_use_error>",
        )
        s.edit("src/extra.py", "", MEAN_EDIT)
        [f] = drift(s)
        assert f.uncertain and f.confidence == 0.55

    def test_a_first_write_with_no_observed_result_does_not_create_the_file(
        self,
    ) -> None:
        s = Script()
        s.prompt(MODE)
        s.call(
            "Write",
            ToolKind.FS_WRITE,
            {"file_path": "src/extra.py", "content": ""},
            output=None,
        )
        s.edit("src/extra.py", "", MEAN_EDIT)
        [f] = drift(s)
        assert f.uncertain and f.confidence == 0.55

    def test_two_spellings_of_one_path_are_one_file(self) -> None:
        s = Script()
        s.prompt(MODE)
        s.read("./src/extra.py", "")
        s.write("src/extra.py", MEAN_EDIT)
        [f] = drift(s)
        assert f.uncertain and f.confidence == 0.55

    def test_a_grounded_file_that_existed_at_the_start_is_not_created(
        self,
    ) -> None:
        s = Script()
        s.prompt(MODE)
        s.write("src/extra.py", MEAN_EDIT)
        g = RepositoryGrounding(
            engine="fake",
            syntax=False,
            files=frozenset({"src/extra.py"}),
            paths={"src/extra.py": "src/extra.py"},
        )
        a = explain(s.events, RegexTokenizer(), repository=g)
        [f] = [f for f in a.findings if f.detector == "intent_drift"]
        assert f.uncertain and f.confidence == 0.55

    def test_a_file_first_touched_by_an_edit_is_not_counted_as_created(
        self,
    ) -> None:
        # Only a write marks a file as created; an edit may change a file the
        # session never read.
        s = Script()
        s.prompt(MODE)
        s.edit("src/extra.py", "", MEAN_EDIT)
        [f] = drift(s)
        assert f.uncertain and f.confidence == 0.55


# --- TER-ITN-004: an alignment score for every agent event --------------------


@pytest.mark.req("TER-ITN-004")
def test_every_agent_event_is_scored_against_the_intent_in_force() -> None:
    s = _shift(redirect=MODE, announce=True)
    a = run(s)
    agent = [
        e.id for e in s.events if e.kind.is_generated and e.actor.value == "assistant"
    ]
    assert [x.event_id for x in a.intent.alignments] == agent
    for x in a.intent.alignments:
        assert x.revision is not None and x.score is not None
        assert 0.0 <= x.score <= 1.0
    by_names = {x.names: x for x in a.intent.alignments if x.names}
    assert by_names[("median",)].revision == 1
    assert by_names[("mode",)].band is AlignmentBand.ALIGNED
    assert by_names[("mean", "variance")].band is AlignmentBand.LOW


@dataclass(frozen=True)
class _Constant:
    value: float
    name: str = "constant"
    rule: str = "always the same"

    def score(self, intent: frozenset[str], activity: frozenset[str]) -> float:
        return self.value


@pytest.mark.req("TER-ITN-004")
def test_the_scorer_is_injected() -> None:
    s = _shift(redirect=MODE, announce=False)
    a = run(s, alignment=_Constant(1.0))
    assert a.intent.scorer == "constant"
    assert {x.score for x in a.intent.alignments} == {1.0}
    assert a.drift_findings == ()
    low = run(s, alignment=_Constant(0.0))
    assert {x.band for x in low.intent.alignments} == {AlignmentBand.LOW}


# --- TER-ITN-005: low-alignment periods ---------------------------------------


def _wander(n: int) -> Script:
    s = Script()
    s.prompt("Add a slugify helper to src/text.py that strips leading dashes")
    for i in range(n):
        s.read(f"docs/unrelated_{i}.md")
    s.edit("src/text.py", "", "def slugify(text): ...")
    return s


@pytest.mark.req("TER-ITN-005")
class TestLowAlignmentPeriods:
    def test_a_run_of_the_configured_length_is_reported(self) -> None:
        s = _wander(3)
        a = run(s)
        [p] = a.intent.periods
        reads = [e.id for e in s.events if e.kind is EventKind.TOOL_REQUESTED][:3]
        assert list(p.events) == reads and p.revision == 1
        assert p.start == reads[0] and p.end == reads[-1] and p.mean_score < 0.25

    def test_a_shorter_run_is_not_reported(self) -> None:
        assert run(_wander(2)).intent.periods == ()

    def test_length_and_band_are_configurable(self) -> None:
        assert (
            len(
                run(_wander(2), intent_config=IntentConfig(min_events=2)).intent.periods
            )
            == 1
        )
        assert (
            run(_wander(3), intent_config=IntentConfig(min_events=4)).intent.periods
            == ()
        )
        partial = _Constant(0.3)
        assert run(_wander(3), alignment=partial).intent.periods == ()
        wider = IntentConfig(low_below=0.31)
        assert (
            len(run(_wander(3), alignment=partial, intent_config=wider).intent.periods)
            == 1
        )

    def test_a_prompt_ends_a_run(self) -> None:
        s = Script()
        s.prompt("Add a slugify helper to src/text.py that strips leading dashes")
        s.read("docs/unrelated_0.md")
        s.read("docs/unrelated_1.md")
        s.prompt("Add a slugify helper to src/text.py that strips trailing dashes")
        s.read("docs/unrelated_2.md")
        assert run(s).intent.periods == ()

    def test_unscorable_events_neither_extend_nor_break_a_run(self) -> None:
        s = Script()
        s.prompt("Add a slugify helper to src/text.py that strips leading dashes")
        s.read("docs/unrelated_0.md")
        s.think("...")  # no key terms: unknown
        s.read("docs/unrelated_1.md")
        s.read("docs/unrelated_2.md")
        [p] = run(s).intent.periods
        assert len(p.events) == 3

    @pytest.mark.parametrize(
        "kw", [{"low_below": 0.0}, {"drift_below": 1.5}, {"min_events": 0}]
    )
    def test_invalid_configuration_is_rejected(self, kw: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            IntentConfig(**kw)


@pytest.mark.req("TER-ITN-005")
def test_golden_example_session_has_a_low_alignment_period() -> None:
    [p] = golden("example_session").intent.periods
    assert len(p.events) >= 3


# --- TER-LEN-003: value judged against the intent -----------------------------


@pytest.mark.req("TER-LEN-003")
def test_the_same_edit_is_valued_against_the_current_intent() -> None:
    def session(prompt: str) -> tuple[LeanAnalysis, str]:
        s = Script()
        s.prompt(prompt)
        s.think("I will also add a mean and a variance function to src/stats.py.")
        request, _ = s.edit("src/stats.py", "", MEAN_EDIT)
        return run(s), request.id

    asked, edit = session("Add mean and variance functions to src/stats.py")
    other, same = session(MODE)
    c_asked = next(c for c in asked.classifications if c.event_id == edit)
    c_other = next(c for c in other.classifications if c.event_id == same)
    assert c_asked.activity_class is ActivityClass.VALUE_ADDING
    assert c_other.activity_class is ActivityClass.AVOIDABLE
    assert c_other.basis.startswith("intent_drift:")
    assert other.scorecard.waste_tokens > asked.scorecard.waste_tokens == 0


# --- the timeline in the analysis and the A3 ----------------------------------


@pytest.mark.req("TER-ITN-001", "TER-ITN-003", "TER-ITN-005")
def test_the_intent_timeline_is_in_the_json_and_the_a3() -> None:
    a = golden()
    data = json.loads(json.dumps(a.as_dict()))["intent"]
    assert [r["relation"] for r in data["revisions"]] == ["opened", "changed"]
    assert data["changes"][0]["event_id"] == data["revisions"][1]["event_id"]
    assert data["drift"] == [f.id for f in a.drift_findings]
    assert {"low_below", "min_events", "drift_below"} == set(data["config"])
    a3 = build_a3(a, ("x",))
    assert a3.as_dict()["background"]["intent"] == data  # type: ignore[index]
    html = render_a3_html(a3)
    assert "Intent timeline" in html
    assert data["revisions"][1]["event_id"] in html
    for event_id in a.drift_findings[0].evidence:
        assert event_id in html
    periods = golden("example_session")
    html = render_a3_html(build_a3(periods, ("x",)))
    assert "low alignment" in html and periods.intent.periods[0].start in html
