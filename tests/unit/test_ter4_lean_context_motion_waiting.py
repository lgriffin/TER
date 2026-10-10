"""Unit tests for the context band, traversal motion, failed-route waiting and
exploration-driver detectors: positive, negative and boundary cases."""

from __future__ import annotations

from pathlib import Path

import pytest
from ter4_lean_builder import PASS, Script

from ter.adapters.driven.gare import GareRunSource
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import EventKind
from ter.domain.lean import (
    ActivityClass,
    DetectorRegistry,
    ExplorationDriver,
    Finding,
    FindingKind,
    FlowState,
    LeanAnalysis,
    LeanWaste,
    build_countermeasures,
    explain,
)
from ter.domain.lean.countermeasures import follow_ups
from ter.domain.lean.detectors import ContextBand, ExcessiveContext

FAILOVER_RUN = Path(__file__).parents[1] / "fixtures" / "gare" / "failover-run"


def run(script: Script, registry: DetectorRegistry | None = None) -> LeanAnalysis:
    if registry is None:
        return explain(script.events, RegexTokenizer())
    return explain(script.events, RegexTokenizer(), registry=registry)


def found(script: Script, detector: str) -> list[Finding]:
    return [f for f in run(script).findings if f.detector == detector]


# --- excessive_context (TER-DET-003, point 21) ------------------------------


@pytest.mark.req("TER-DET-003")
class TestExcessiveContext:
    def test_context_above_the_band_is_inventory(self) -> None:
        s = Script()
        s.prompt("fix the bug in src/a.py")
        reads = [s.read(f"src/m{i}.py")[0] for i in range(7)]
        edit, _ = s.edit("src/a.py")
        [f] = found(s, "excessive_context")
        # Band for one changed file: 3 x 1 + 3 = 6 items; 7 is one over.
        assert f.waste is LeanWaste.INVENTORY and f.kind is FindingKind.WASTE
        assert not f.uncertain and f.confidence == 0.72  # judged 10 of 10
        assert reads[6].id in f.waste_events and reads[5].id not in f.waste_events
        assert edit.id in f.evidence and f.id.endswith(edit.id)

    def test_context_within_the_band_is_fine(self) -> None:
        s = Script()
        s.prompt("fix the bug in src/a.py")
        for i in range(6):
            s.read(f"src/m{i}.py")
        s.edit("src/a.py")
        assert found(s, "excessive_context") == []

    def test_band_grows_with_the_files_the_change_touches(self) -> None:
        s = Script()
        s.prompt("fix the bug")
        for i in range(8):
            s.read(f"src/m{i}.py")
        s.edit("src/a.py")
        s.edit("src/b.py")  # two files: band 3 x 2 + 3 = 9
        assert found(s, "excessive_context") == []

    def test_reads_of_changed_files_are_never_claimed(self) -> None:
        s = Script()
        s.prompt("fix the bug")
        for i in range(6):
            s.read(f"src/m{i}.py")
        own, _ = s.read("src/a.py")
        s.edit("src/a.py")
        [f] = found(s, "excessive_context")
        assert own.id in f.evidence and f.waste_events == ()

    def test_rereads_and_ranges_count_once(self) -> None:
        s = Script()
        s.prompt("fix the bug")
        for i in range(6):
            s.read(f"src/m{i}.py")
        s.read("src/m0.py", offset=200)
        s.edit("src/a.py")
        assert found(s, "excessive_context") == []

    def test_band_is_configurable(self) -> None:
        s = Script()
        s.prompt("fix the bug")
        s.read("src/m0.py")
        s.read("src/m1.py")
        s.edit("src/a.py")
        tight = DetectorRegistry(
            [ExcessiveContext(band=ContextBand(per_file=1, slack=0))]
        )
        [f] = run(s, tight).findings
        assert f.detector == "excessive_context"
        assert found(s, "excessive_context") == []

    def test_each_task_has_its_own_band(self) -> None:
        s = Script()
        s.prompt("look around")
        for i in range(7):
            s.read(f"src/m{i}.py")
        s.say("done")
        s.prompt("now fix src/a.py")
        s.read("src/a.py")
        s.edit("src/a.py")
        assert found(s, "excessive_context") == []


# --- insufficient_context (TER-DET-003, point 22) ---------------------------


@pytest.mark.req("TER-DET-003")
class TestInsufficientContext:
    def test_edit_with_no_context_in_the_task(self) -> None:
        s = Script()
        prompt = s.prompt("fix the bug in src/a.py")
        edit, _ = s.edit("src/a.py")
        [f] = found(s, "insufficient_context")
        assert f.kind is FindingKind.RISK and f.waste is LeanWaste.DEFECTS
        assert f.confidence == 0.7 and f.tokens == 0 and f.waste_events == ()
        assert f.evidence[0] == prompt.id and edit.id in f.evidence

    def test_one_item_per_file_is_enough(self) -> None:
        s = Script()
        s.prompt("fix both")
        s.read("src/a.py")
        s.edit("src/a.py")
        s.search("helper")
        s.edit("src/b.py")
        assert found(s, "insufficient_context") == []

    def test_boundary_second_file_without_its_own_item(self) -> None:
        s = Script()
        s.prompt("fix both")
        s.read("src/a.py")
        s.edit("src/a.py")
        second, _ = s.edit("src/b.py")
        [f] = found(s, "insufficient_context")
        assert f.subject == "src/b.py" and f.id.endswith(second.id)

    def test_repeated_edits_of_one_file_need_no_more_context(self) -> None:
        s = Script()
        s.prompt("fix a")
        s.read("src/a.py")
        for i in range(4):
            s.edit("src/a.py", str(i))
        assert found(s, "insufficient_context") == []

    def test_new_files_are_left_to_premature_implementation(self) -> None:
        s = Script()
        s.prompt("add a module")
        s.write("src/new.py", "x = 1\n")
        assert found(s, "insufficient_context") == []

    def test_context_carried_from_an_earlier_task_is_uncertain(self) -> None:
        s = Script()
        s.prompt("explain src/a.py")
        s.read("src/a.py")
        s.say("It parses arguments.")
        s.prompt("now rename the flag")
        s.edit("src/a.py")
        [f] = found(s, "insufficient_context")
        assert f.uncertain and f.confidence == 0.55


# --- unused_traversal (TER-DET-007, point 28) -------------------------------


@pytest.mark.req("TER-DET-007")
class TestUnusedTraversal:
    def test_search_whose_files_are_never_used_is_motion(self) -> None:
        s = Script()
        s.prompt("add retries")
        search, result = s.search("retry", "src/old.py:1:retry\nsrc/legacy.py:9:retry")
        s.read("src/net.py")
        s.edit("src/net.py")
        s.say("Added retries to net.py.")
        [f] = found(s, "unused_traversal")
        assert f.waste is LeanWaste.MOTION and f.kind is FindingKind.WASTE
        assert not f.uncertain and f.confidence == 0.72  # judged 10 of 10
        assert result is not None
        assert f.waste_events == (search.id, result.id)

    def test_traversal_whose_file_is_read_later_is_used(self) -> None:
        s = Script()
        s.search("retry", "src/old.py:1:retry\nsrc/legacy.py:9:retry")
        s.read("src/old.py")
        s.say("done")
        assert found(s, "unused_traversal") == []

    def test_naming_a_listed_file_in_reasoning_counts_as_use(self) -> None:
        s = Script()
        s.bash("ls src", "net.py\nlegacy.py")
        s.think("legacy.py is unrelated; the change belongs in net.py")
        s.say("done")
        assert found(s, "unused_traversal") == []

    def test_shell_traversal_is_judged_too(self) -> None:
        s = Script()
        s.bash("find . -name '*.cfg'", "./setup.cfg\n./tox.cfg")
        s.say("done")
        [f] = found(s, "unused_traversal")
        assert f.subject.startswith("find")

    def test_empty_result_rules_something_out(self) -> None:
        s = Script()
        s.search("retry", "No matches found")
        s.say("done")
        assert found(s, "unused_traversal") == []

    def test_not_judged_before_a_response(self) -> None:
        s = Script()
        s.search("retry", "src/old.py:1:retry")
        assert found(s, "unused_traversal") == []

    def test_non_traversing_shell_is_not_judged(self) -> None:
        s = Script()
        s.bash("git status", "modified: src/a.py")
        s.say("done")
        assert found(s, "unused_traversal") == []

    def test_a_repeat_is_left_to_repeated_exploration(self) -> None:
        s = Script()
        s.search("retry", "src/old.py:1:retry")
        s.search("retry", "src/old.py:1:retry")
        s.say("done")
        assert len(found(s, "unused_traversal")) == 1
        assert len(found(s, "repeated_exploration")) == 1


# --- failed_route (TER-DET-008, points 29, 30) ------------------------------


@pytest.mark.req("TER-DET-008")
class TestFailedRoute:
    def test_failover_then_work_on_another_route_is_waiting(self) -> None:
        s = Script()
        s.prompt("research the API")
        failed = s.failover("research-1: flaky/large failed (PROVIDER_UNAVAILABLE)")
        done = s.say("research-1: mock/smart")
        [f] = found(s, "failed_route")
        assert f.waste is LeanWaste.WAITING and f.kind is FindingKind.WASTE
        assert f.confidence == 0.8 and not f.uncertain
        assert f.evidence == (failed.id, done.id) and f.waste_events == (failed.id,)
        assert f.tokens == 0 and f.seconds == 1.0

    def test_the_wait_lands_in_the_waiting_flow_state(self) -> None:
        s = Script()
        s.prompt("research the API")
        failed = s.failover("research-1: flaky/large failed")
        s.say("research-1: mock/smart")
        a = run(s)
        assert dict(a.scorecard.flow_seconds)[FlowState.WAITING] == 1.0
        [c] = [c for c in a.classifications if c.event_id == failed.id]
        assert (
            c.flow is FlowState.WAITING and c.activity_class is ActivityClass.AVOIDABLE
        )

    def test_no_failover_no_finding(self) -> None:
        s = Script()
        s.prompt("research the API")
        s.say("research-1: mock/smart")
        assert found(s, "failed_route") == []

    def test_boundary_failover_never_recovered_is_uncertain(self) -> None:
        s = Script()
        s.prompt("research the API")
        s.failover("research-1: flaky/large failed")
        [f] = found(s, "failed_route")
        assert f.uncertain and f.confidence == 0.55 and len(f.evidence) == 1

    def test_the_response_for_the_same_task_is_cited(self) -> None:
        s = Script()
        s.prompt("two tasks")
        s.failover("coder-2: flaky/large failed")
        s.say("research-1: mock/smart")
        same = s.say("coder-2: mock/smart")
        [f] = found(s, "failed_route")
        assert f.evidence[-1] == same.id

    def test_other_routing_markers_are_not_steps(self) -> None:
        s = Script()
        s.prompt("research the API")
        s.lifecycle(EventKind.ROUTE_SELECTED, "research-1: mock/smart")
        s.lifecycle(EventKind.ATTEMPT_STARTED, "attempt 1 research-1")
        s.say("research-1: mock/smart")
        a = run(s)
        assert len(a.steps) == 2 and found(s, "failed_route") == []

    def test_gare_failover_run(self) -> None:
        trace = GareRunSource().read(FAILOVER_RUN)
        a = explain(trace.events, RegexTokenizer())
        routes = [f for f in a.findings if f.detector == "failed_route"]
        assert len(routes) == 3 and all(not f.uncertain for f in routes)
        assert all("flaky/flaky-large failed" in f.subject for f in routes)
        assert dict(a.scorecard.flow_seconds)[FlowState.WAITING] > 0
        [cm] = [
            c
            for c in build_countermeasures(a.findings, a.steps)
            if c.detector == "failed_route"
        ]
        assert "flaky/flaky-large" in cm.actions[0].text


# --- exploration drivers (TER-DET-009, point 38) ----------------------------


@pytest.mark.req("TER-DET-009")
class TestExplorationDrivers:
    def label_of(
        self, s: Script, event_id: str
    ) -> tuple[ExplorationDriver, str | None]:
        [label] = [e for e in run(s).exploration if e.event_id == event_id]
        return label.driver, label.motive

    def test_exploration_addressing_an_open_question(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag to cli.py")
        question = s.think("Where is the argument parser defined?")
        search, _ = s.search("parser")
        assert self.label_of(s, search.id) == (
            ExplorationDriver.UNCERTAINTY_DRIVEN,
            question.id,
        )

    def test_exploration_of_the_stated_task_is_intent_directed(self) -> None:
        s = Script()
        prompt = s.prompt("Add a verbose flag to cli.py")
        s.think("Where is the argument parser defined?")
        read, _ = s.read("src/cli.py")
        assert self.label_of(s, read.id) == (
            ExplorationDriver.INTENT_DIRECTED,
            prompt.id,
        )

    def test_unlinked_exploration_is_aimless(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag to cli.py")
        read, _ = s.read("docs/history.md")
        assert self.label_of(s, read.id) == (ExplorationDriver.AIMLESS, None)

    def test_a_question_asked_later_does_not_drive_earlier_exploration(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        search, _ = s.search("parser")
        s.think("Where is the parser defined?")
        assert (
            self.label_of(s, search.id)[0] is not ExplorationDriver.UNCERTAINTY_DRIVEN
        )

    def test_boundary_questions_close_at_the_next_prompt(self) -> None:
        s = Script()
        s.prompt("Where is the parser defined?")
        s.say("In cli.py.")
        s.prompt("Add a verbose flag")
        search, _ = s.search("parser")
        assert self.label_of(s, search.id)[0] is ExplorationDriver.AIMLESS

    def test_a_statement_is_not_a_question(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        s.think("The parser is defined somewhere.")
        search, _ = s.search("parser")
        assert self.label_of(s, search.id)[0] is ExplorationDriver.AIMLESS

    def test_labels_cover_every_exploration_request_and_are_exported(self) -> None:
        s = Script()
        s.prompt("Why does the build fail?")
        s.bash("ls build", "out.log")
        s.read("build/out.log")
        s.bash("pytest -q", PASS)
        s.edit("src/a.py")
        a = run(s)
        explored = [
            x.event_id for x in a.steps if x.is_request and x.stage.value == "explore"
        ]
        assert [e.event_id for e in a.exploration] == explored
        exported = a.as_dict()["exploration"]
        assert isinstance(exported, list) and len(exported) == len(explored)

    def test_labels_never_change_the_activity_class(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        read, _ = s.read("docs/history.md")
        [c] = [c for c in run(s).classifications if c.event_id == read.id]
        assert c.activity_class is ActivityClass.NECESSARY_NON_VALUE_ADDING


# --- countermeasures and follow-up ------------------------------------------


@pytest.mark.req("TER-RPT-005")
def test_every_new_detector_has_a_countermeasure_and_a_follow_up() -> None:
    s = Script()
    s.prompt("fix a")
    s.failover("t: flaky/large failed")
    s.search("retry", "src/old.py:1:retry")
    for i in range(7):
        s.read(f"src/m{i}.py")
    s.edit("src/a.py")
    s.say("t: mock/smart")
    s.prompt("now fix b")
    s.edit("src/b.py")
    s.say("done")
    a = run(s)
    fired = {f.detector for f in a.findings}
    new = {
        "excessive_context",
        "insufficient_context",
        "unused_traversal",
        "failed_route",
    }
    assert new <= fired
    measures = {c.detector: c for c in build_countermeasures(a.findings, a.steps)}
    assert new <= set(measures) and all(measures[d].actions for d in new)
    follow = follow_ups(a.findings, flow_efficiency=None, avoidable_share=0.0)
    assert {f.how for f in follow} >= {f"findings[detector={d}]" for d in new}
