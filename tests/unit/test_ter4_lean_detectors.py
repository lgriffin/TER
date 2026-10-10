"""Unit tests for every L2 waste detector: positive, negative and boundary cases."""

from __future__ import annotations

import pytest
from ter4_lean_builder import FAIL, FAIL_OTHER, PASS, Script

from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import ToolKind
from ter.domain.lean import (
    CycleVerdict,
    Finding,
    FindingKind,
    LeanAnalysis,
    LeanWaste,
    Outcome,
    explain,
)


def run(script: Script) -> LeanAnalysis:
    return explain(script.events, RegexTokenizer())


def found(script: Script, detector: str) -> list[Finding]:
    return [f for f in run(script).findings if f.detector == detector]


# --- repeated_tool_call ------------------------------------------------------


@pytest.mark.req("TER-DET-002", "TER-DET-010")
class TestRepeatedToolCall:
    def test_identical_call_and_output_is_waste(self) -> None:
        s = Script()
        s.prompt("list things")
        s.bash("ls src", "a.py")
        _, first = s.bash("ls src", "a.py")
        [f] = found(s, "repeated_tool_call")
        assert f.confidence == 0.9 and not f.uncertain
        assert f.waste is LeanWaste.OVER_PROCESSING and f.kind is FindingKind.WASTE
        assert first is not None and first.id in f.waste_events
        assert len(f.evidence) == 4

    def test_different_output_is_not_waste(self) -> None:
        s = Script()
        s.bash("ls src", "a.py")
        s.bash("ls src", "a.py\nb.py")
        assert found(s, "repeated_tool_call") == []

    def test_timing_noise_does_not_make_output_different(self) -> None:
        s = Script()
        s.bash("pytest -q", "1 failed in 0.21s")
        s.bash("pytest -q", "1 failed in 0.19s")
        [f] = found(s, "repeated_tool_call")
        assert f.confidence == 0.85 and "validation" in f.title

    def test_validation_rerun_after_edit_is_left_to_rework(self) -> None:
        s = Script()
        s.bash("pytest -q", FAIL)
        s.edit("src/a.py")
        s.bash("pytest -q", FAIL)
        assert found(s, "repeated_tool_call") == []

    def test_non_validation_repeat_after_edit_is_less_certain(self) -> None:
        s = Script()
        s.bash("ls src", "a.py")
        s.edit("src/a.py")
        s.bash("ls src", "a.py")
        [f] = found(s, "repeated_tool_call")
        assert f.confidence == 0.75

    def test_unobserved_output_is_uncertain(self) -> None:
        s = Script()
        s.bash("ls src", "a.py")
        s.bash("ls src", None)
        [f] = found(s, "repeated_tool_call")
        assert f.uncertain and f.confidence == 0.5

    def test_reads_are_left_to_repeated_exploration(self) -> None:
        s = Script()
        s.read("a.py")
        s.read("a.py")
        assert found(s, "repeated_tool_call") == []

    # Calibrated on real transcripts: a turn-ending call (a "no reply needed"
    # tool) repeated once per turn was the whole of the confident findings,
    # and 17 of 17 repeats across a prompt judged on real sessions were not
    # waste. A new prompt is new information.
    def test_turn_ending_call_repeated_across_prompts_is_not_a_finding(self) -> None:
        s = Script()
        for message in ("first wake", "second wake", "third wake"):
            s.prompt(message)
            s.call("end_turn", ToolKind.OTHER, {"reason": "other"}, "ok")
        assert found(s, "repeated_tool_call") == []

    def test_repeat_within_one_turn_after_a_response_is_still_waste(self) -> None:
        s = Script()
        s.prompt("check devices")
        s.call("list_devices", ToolKind.OTHER, {}, "laptop")
        s.say("One device.")
        s.call("list_devices", ToolKind.OTHER, {}, "laptop")
        [f] = found(s, "repeated_tool_call")
        assert f.confidence == 0.9 and not f.uncertain

    def test_boundary_validation_rerun_across_a_prompt_is_not_a_finding(
        self,
    ) -> None:
        s = Script()
        s.prompt("run the tests")
        s.bash("pytest -q", PASS)
        s.prompt("run them again")
        s.bash("pytest -q", PASS)
        assert found(s, "repeated_tool_call") == []
        # The same re-run inside one turn is still a finding.
        s.bash("pytest -q", PASS)
        [f] = found(s, "repeated_tool_call")
        assert f.confidence == 0.85


# --- repeated_exploration ----------------------------------------------------


@pytest.mark.req("TER-DET-002")
class TestRepeatedExploration:
    def test_rereading_unchanged_file(self) -> None:
        s = Script()
        s.read("src/a.py", "def f(): pass")
        s.read("src/a.py", "def f(): pass")
        [f] = found(s, "repeated_exploration")
        assert f.waste is LeanWaste.MOTION and f.confidence == 0.85
        assert f.context_tokens > 0

    def test_reread_after_editing_that_file_is_legitimate(self) -> None:
        s = Script()
        s.read("src/a.py")
        s.edit("src/a.py")
        s.read("src/a.py")
        assert found(s, "repeated_exploration") == []

    def test_reread_after_editing_another_file_still_counts(self) -> None:
        s = Script()
        s.read("src/a.py")
        s.edit("src/b.py")
        s.read("src/a.py")
        assert len(found(s, "repeated_exploration")) == 1

    def test_different_range_is_not_a_repeat(self) -> None:
        s = Script()
        s.read("src/a.py", offset=1)
        s.read("src/a.py", offset=200)
        assert found(s, "repeated_exploration") == []

    def test_changed_output_is_not_a_repeat(self) -> None:
        s = Script()
        s.read("src/a.py", "v1")
        s.read("src/a.py", "v2")
        assert found(s, "repeated_exploration") == []

    def test_repeated_search_and_after_any_edit(self) -> None:
        s = Script()
        s.search("argparse")
        s.search("argparse")
        assert len(found(s, "repeated_exploration")) == 1
        s.edit("x.py")
        s.search("argparse")
        assert len(found(s, "repeated_exploration")) == 1

    def test_unobserved_output_is_uncertain(self) -> None:
        s = Script()
        s.read("src/a.py")
        s.call("Read", ToolKind.FS_READ, {"file_path": "src/a.py"}, None)
        [f] = found(s, "repeated_exploration")
        assert f.uncertain


# --- rework_cycle and productive iteration (point 37) -----------------------


@pytest.mark.req("TER-DET-006")
class TestReworkVersusIteration:
    def test_identical_failure_after_fix_is_rework(self) -> None:
        s = Script()
        s.bash("pytest -q", FAIL)
        fix, _ = s.edit("src/a.py")
        s.bash("pytest -q", FAIL.replace("0.12", "0.30"))
        a = run(s)
        [f] = [f for f in a.findings if f.detector == "rework_cycle"]
        assert f.waste is LeanWaste.REWORK and f.confidence == 0.8
        assert fix.id in f.waste_events
        assert [c.verdict for c in a.cycles] == [CycleVerdict.REWORK]

    def test_fail_fix_pass_is_iteration_not_waste(self) -> None:
        s = Script()
        s.bash("pytest -q", FAIL)
        fix, _ = s.edit("src/a.py")
        s.bash("pytest -q", PASS)
        a = run(s)
        assert [f for f in a.findings if f.detector == "rework_cycle"] == []
        [cycle] = a.cycles
        assert cycle.verdict is CycleVerdict.ITERATION and "passed" in cycle.reason
        by_id = {c.event_id: c for c in a.classifications}
        assert by_id[fix.id].flow.value == "recovering"
        assert by_id[fix.id].avoidable_share == 0

    def test_failure_that_moves_is_iteration(self) -> None:
        s = Script()
        s.bash("pytest -q", FAIL)
        s.edit("src/a.py")
        s.bash("pytest -q", FAIL_OTHER)
        a = run(s)
        assert a.cycles[0].verdict is CycleVerdict.ITERATION
        assert "differently" in a.cycles[0].reason

    def test_second_identical_failure_raises_confidence(self) -> None:
        s = Script()
        s.bash("pytest -q", FAIL)
        s.edit("src/a.py", "1")
        s.bash("pytest -q", FAIL)
        s.edit("src/a.py", "2")
        s.bash("pytest -q", FAIL)
        confidences = sorted(f.confidence for f in found(s, "rework_cycle"))
        assert confidences == [0.8, 0.9]

    def test_iteration_resets_the_streak(self) -> None:
        s = Script()
        s.bash("pytest -q", FAIL)
        s.edit("src/a.py", "1")
        s.bash("pytest -q", FAIL_OTHER)
        s.edit("src/a.py", "2")
        s.bash("pytest -q", FAIL_OTHER)
        [f] = found(s, "rework_cycle")
        assert f.confidence == 0.8

    def test_rerun_without_edit_or_unknown_outcome_is_no_cycle(self) -> None:
        s = Script()
        s.bash("pytest -q", FAIL)
        s.bash("pytest -q", FAIL)
        s.bash("pytest -q", FAIL)
        s.edit("a.py")
        s.bash("pytest -q", "collected 0 items")
        assert run(s).cycles == ()

    def test_different_command_is_not_paired(self) -> None:
        s = Script()
        s.bash("pytest tests/a.py -q", FAIL)
        s.edit("a.py")
        s.bash("pytest tests/b.py -q", FAIL)
        assert run(s).cycles == ()


# --- unvalidated_implementation ---------------------------------------------


@pytest.mark.req("TER-DET-005")
class TestUnvalidatedImplementation:
    def test_edit_then_response_without_any_check(self) -> None:
        s = Script()
        s.prompt("fix it")
        s.read("src/a.py")
        s.edit("src/a.py")
        s.say("Done.")
        [f] = found(s, "unvalidated_implementation")
        assert f.kind is FindingKind.RISK and f.confidence == 0.85
        assert f.waste_events == () and f.tokens == 0

    def test_validated_edit_is_fine(self) -> None:
        s = Script()
        s.edit("src/a.py")
        s.bash("pytest -q", PASS)
        s.say("Done.")
        assert found(s, "unvalidated_implementation") == []

    def test_no_response_yet_is_work_in_progress(self) -> None:
        s = Script()
        s.edit("src/a.py")
        assert found(s, "unvalidated_implementation") == []

    def test_earlier_validation_lowers_confidence(self) -> None:
        s = Script()
        s.bash("pytest -q", PASS)
        s.edit("src/a.py")
        s.say("Done.")
        [f] = found(s, "unvalidated_implementation")
        assert f.confidence == 0.75

    def test_documentation_only_is_uncertain(self) -> None:
        s = Script()
        for _ in range(3):
            s.edit("README.md")
        s.say("Done.")
        [f] = found(s, "unvalidated_implementation")
        assert f.uncertain and f.title.startswith("3 edit(s)")

    # Judged on real sessions: one or two documentation edits reported
    # without a check were never waste.
    def test_boundary_fewer_than_three_documentation_edits_is_not_a_finding(
        self,
    ) -> None:
        s = Script()
        s.edit("README.md")
        s.edit("docs/guide.md")
        s.say("Done.")
        assert found(s, "unvalidated_implementation") == []

    def test_one_code_edit_beside_documentation_still_counts(self) -> None:
        s = Script()
        s.edit("README.md")
        s.edit("src/a.py")
        s.say("Done.")
        [f] = found(s, "unvalidated_implementation")
        assert f.confidence == 0.85

    def test_responding_after_a_failing_check(self) -> None:
        s = Script()
        s.edit("src/a.py")
        s.bash("pytest -q", FAIL)
        s.say("Done, mostly.")
        [f] = found(s, "unvalidated_implementation")
        assert "failing" in f.title and f.confidence == 0.85

    def test_judged_per_prompt(self) -> None:
        s = Script()
        s.prompt("one")
        s.edit("src/a.py")
        s.say("Done.")
        s.prompt("two")
        s.edit("src/b.py")
        s.bash("pytest -q", PASS)
        s.say("Done.")
        [f] = found(s, "unvalidated_implementation")
        assert "src/a.py" in f.subject

    # Calibrated on real transcripts: checks are often chained after a change
    # in one shell line, which the shell intent reads as a change.
    def test_check_chained_after_a_change_validates_the_edits(self) -> None:
        s = Script()
        s.prompt("fix it")
        s.read("src/a.py")
        s.edit("src/a.py")
        _, done = s.bash("sed -i 's/x/y/' src/b.py && pytest -q tests/unit", PASS)
        s.say("Done.")
        a = run(s)
        assert [
            f for f in a.findings if f.detector == "unvalidated_implementation"
        ] == []
        assert done is not None
        [result] = [x for x in a.steps if x.event_id == done.id]
        assert result.outcome is Outcome.PASSED

    def test_failing_check_chained_after_a_change_is_reported(self) -> None:
        s = Script()
        s.prompt("fix it")
        s.read("src/a.py")
        s.edit("src/a.py")
        s.bash("sed -i 's/x/y/' src/b.py && pytest -q tests/unit", FAIL)
        s.say("Done, mostly.")
        [f] = found(s, "unvalidated_implementation")
        assert "failing" in f.title and f.confidence == 0.85

    def test_check_before_a_commit_validates_the_edits(self) -> None:
        s = Script()
        s.edit("src/a.py")
        s.bash("ruff check src && git add -A && git commit -qm fix", "ok")
        s.say("Done.")
        assert found(s, "unvalidated_implementation") == []

    def test_ci_checks_validate_the_edits(self) -> None:
        s = Script()
        s.edit("src/a.py")
        s.bash("gh pr checks 12 --watch", "build pass")
        s.say("Done.")
        assert found(s, "unvalidated_implementation") == []

    def test_change_line_without_a_check_does_not_validate(self) -> None:
        s = Script()
        s.edit("src/a.py")
        s.bash("git add -A && git commit -qm fix", "ok")
        s.say("Done.")
        [f] = found(s, "unvalidated_implementation")
        assert f.confidence == 0.85

    def test_boundary_ad_hoc_script_beside_a_change_is_not_a_check(self) -> None:
        s = Script()
        s.edit("src/a.py")
        s.bash("mkdir -p out && python3 - <<'EOF'\nprint(1)\nEOF", "1")
        s.say("Done.")
        [f] = found(s, "unvalidated_implementation")
        assert f.confidence == 0.85


# --- premature_implementation -----------------------------------------------


@pytest.mark.req("TER-DET-005")
class TestPrematureImplementation:
    def test_edit_of_unread_file(self) -> None:
        s = Script()
        s.prompt("fix a")
        s.edit("src/a.py")
        [f] = found(s, "premature_implementation")
        assert f.kind is FindingKind.RISK and f.confidence == 0.75

    def test_read_before_edit_is_fine(self) -> None:
        s = Script()
        s.read("src/a.py")
        s.edit("src/a.py")
        assert found(s, "premature_implementation") == []

    def test_file_named_in_search_output_counts_as_seen(self) -> None:
        s = Script()
        s.search("thing", "src/a.py:3: thing = 1")
        s.edit("src/a.py")
        assert found(s, "premature_implementation") == []

    def test_new_file_before_any_exploration_is_uncertain(self) -> None:
        s = Script()
        s.write("src/new.py", "x = 1")
        [f] = found(s, "premature_implementation")
        assert f.uncertain

    def test_new_file_after_exploration_is_fine(self) -> None:
        s = Script()
        s.read("src/a.py")
        s.write("src/new.py", "x = 1")
        s.edit("src/new.py")
        assert found(s, "premature_implementation") == []

    # Calibrated on real transcripts: the one confident finding was a file
    # read with ``cat`` in the shell, whose output never names the file.
    def test_file_read_through_the_shell_counts_as_seen(self) -> None:
        s = Script()
        s.prompt("fix the builder")
        s.bash("sed -n 1,40p tests/a.py; cat tests/builder.py", "def f():\n    pass")
        s.edit("/repo/tests/builder.py")
        assert found(s, "premature_implementation") == []

    def test_shell_command_naming_another_file_does_not_count(self) -> None:
        s = Script()
        s.bash("cat tests/other.py", "def f():\n    pass")
        s.edit("/repo/tests/builder.py")
        [f] = found(s, "premature_implementation")
        assert f.confidence == 0.75

    def test_boundary_shell_read_after_the_edit_does_not_count(self) -> None:
        s = Script()
        s.edit("/repo/tests/builder.py")
        s.bash("cat tests/builder.py", "def f():\n    pass")
        [f] = found(s, "premature_implementation")
        assert f.confidence == 0.75


# --- excessive_planning ------------------------------------------------------


@pytest.mark.req("TER-DET-005")
class TestExcessivePlanning:
    def test_four_planning_steps_without_action(self) -> None:
        s = Script()
        s.think("plan the retry decorator carefully")
        s.todo("design")
        restated = s.think("plan the retry decorator, carefully")
        again, again_result = s.todo("design")
        s.read("a.py")
        [f] = found(s, "excessive_planning")
        assert f.confidence == 0.75
        assert len(f.evidence) == 6  # steps and todo results
        assert again_result is not None
        assert set(f.waste_events) == {restated.id, again.id, again_result.id}

    def test_three_is_below_threshold(self) -> None:
        s = Script()
        s.think("a b c")
        s.think("d e f")
        s.todo("x")
        s.read("a.py")
        assert found(s, "excessive_planning") == []

    def test_run_at_end_of_session_counts(self) -> None:
        s = Script()
        for _ in range(5):
            s.think("thought about the retry plan")
        assert found(s, "excessive_planning")[0].confidence == 0.8

    @pytest.mark.req("TER-LEN-004")
    def test_planning_steps_that_add_decisions_are_never_waste(self) -> None:
        s = Script()
        s.prompt("Add a retry decorator to src/net.py")
        s.think("I need a retry decorator with exponential backoff.")
        s.think("It needs attempts, a base delay and a multiplier.")
        s.think("Plan: write decorator, apply to fetch_json, run tests.")
        s.todo("write decorator")
        s.todo("apply decorator to fetch_json")
        s.read("src/net.py")
        assert found(s, "excessive_planning") == []

    @pytest.mark.req("TER-LEN-004")
    def test_only_the_restating_step_of_a_run_is_waste(self) -> None:
        s = Script()
        s.prompt("Add a retry decorator to src/net.py")
        s.think("I need a retry decorator with exponential backoff.")
        s.think("It needs attempts, a base delay and a multiplier.")
        decision = s.think("Plan: write decorator, apply to fetch_json, run tests.")
        restated = s.think("So: a retry decorator with exponential backoff.")
        s.read("src/net.py")
        [f] = found(s, "excessive_planning")
        assert f.waste_events == (restated.id,)
        assert decision.id in f.evidence and decision.id not in f.waste_events

    @pytest.mark.req("TER-LEN-004")
    def test_boundary_a_quarter_new_words_is_still_a_restatement(self) -> None:
        s = Script()
        s.think("alpha beta gamma delta")
        s.think("epsilon zeta theta iota")
        s.think("kappa lambda sigma omega")
        # 1 of 4 content words new (25%): no decision added.
        low = s.think("alpha beta gamma rho")
        # 2 of 4 new (50%): a decision.
        s.think("alpha beta upsilon chi")
        s.read("a.py")
        [f] = found(s, "excessive_planning")
        assert f.waste_events == (low.id,)


# --- fragmented_edits --------------------------------------------------------


@pytest.mark.req("TER-DET-007")
class TestFragmentedEdits:
    def test_three_edits_in_a_row(self) -> None:
        s = Script()
        s.read("a.py")
        s.edit("a.py", "1")
        s.think("next hunk")
        s.edit("a.py", "2")
        s.edit("a.py", "3")
        s.bash("pytest -q", PASS)
        [f] = found(s, "fragmented_edits")
        assert f.confidence == 0.7 and f.tokens == 0 and f.context_tokens > 0
        assert f.waste is LeanWaste.MOTION

    def test_two_edits_are_fine(self) -> None:
        s = Script()
        s.edit("a.py", "1")
        s.edit("a.py", "2")
        assert found(s, "fragmented_edits") == []

    def test_interleaved_files_break_the_run(self) -> None:
        s = Script()
        s.edit("a.py", "1")
        s.edit("b.py", "1")
        s.edit("a.py", "2")
        s.edit("a.py", "3")
        assert found(s, "fragmented_edits") == []

    def test_other_tools_break_the_run(self) -> None:
        s = Script()
        s.edit("a.py", "1")
        s.edit("a.py", "2")
        s.read("b.py")
        s.edit("a.py", "3")
        assert found(s, "fragmented_edits") == []

    def test_longer_runs_are_more_certain(self) -> None:
        s = Script()
        for i in range(5):
            s.edit("a.py", str(i))
        assert found(s, "fragmented_edits")[0].confidence == 0.8

    def test_edits_sent_in_one_turn_are_one_round_trip(self) -> None:
        # Parallel Edit calls: all requested before the first result arrives.
        s = Script()
        s.read("a.py")
        calls = [s.edit("a.py", str(i), output=None)[0] for i in range(4)]
        for call in calls:
            s.complete(call, "updated")
        assert found(s, "fragmented_edits") == []

    def test_two_round_trips_are_fine_however_many_calls(self) -> None:
        s = Script()
        first = [s.edit("a.py", str(i), output=None)[0] for i in range(3)]
        for call in first:
            s.complete(call, "updated")
        s.think("one more hunk")
        s.edit("a.py", "3")
        assert found(s, "fragmented_edits") == []

    def test_round_trips_not_calls_are_counted(self) -> None:
        s = Script()
        first = [s.edit("a.py", str(i), output=None)[0] for i in range(3)]
        first_results = [s.complete(call, "updated") for call in first]
        _, second = s.edit("a.py", "3")
        _, third = s.edit("a.py", "4")
        [f] = found(s, "fragmented_edits")
        assert f.confidence == 0.7
        assert f.title.startswith("5 edits to a.py over 3 round trips")
        assert "parallel Edit calls" in f.explanation
        assert "multi-edit" not in f.explanation.lower()
        # The first round trip is the change; only later results are overhead.
        assert second is not None and third is not None
        assert f.waste_events == (second.id, third.id)
        assert not set(f.waste_events) & {r.id for r in first_results}


# --- unused_context ----------------------------------------------------------


class TestUnusedContext:
    def test_read_never_mentioned_is_uncertain_inventory(self) -> None:
        s = Script()
        s.prompt("add a flag to the cli")
        s.read("src/cli.py", "def main(): pass")
        s.read("src/utils.py", "def slugify(s): return s")
        s.edit("src/cli.py")
        s.say("Added the flag in cli.py.")
        [f] = found(s, "unused_context")
        assert f.subject == "src/utils.py" and f.waste is LeanWaste.INVENTORY
        assert f.uncertain and f.confidence == 0.65

    def test_identifier_use_counts(self) -> None:
        s = Script()
        s.read("src/utils.py", "def slugify(s): return s")
        s.think("I can reuse slugify here")
        s.say("done")
        assert found(s, "unused_context") == []

    def test_file_without_definitions_is_less_certain(self) -> None:
        s = Script()
        s.read("notes.txt", "just some notes")
        s.say("done")
        [f] = found(s, "unused_context")
        assert f.confidence == 0.55

    def test_not_judged_before_a_response(self) -> None:
        s = Script()
        s.read("src/utils.py", "def slugify(s): return s")
        assert found(s, "unused_context") == []


# --- unnecessary_handoff -----------------------------------------------------


@pytest.mark.req("TER-DET-008")
class TestUnnecessaryHandoff:
    def test_handoff_then_same_work_directly(self) -> None:
        s = Script()
        task, _ = s.task(
            "Research requests changelog", "Find breaking changes in requests"
        )
        s.call(
            "WebFetch",
            ToolKind.NET_FETCH,
            {
                "url": "https://x.invalid/requests/changelog",
                "prompt": "breaking changes",
            },
            "notes",
        )
        [f] = found(s, "unnecessary_handoff")
        assert f.waste is LeanWaste.HANDOFFS and task.id in f.waste_events
        assert 0.7 <= f.confidence <= 0.85

    def test_unrelated_follow_up_is_fine(self) -> None:
        s = Script()
        s.task("Research requests changelog", "Find breaking changes in requests")
        s.read("pyproject.toml")
        assert found(s, "unnecessary_handoff") == []

    # Calibrated on real transcripts: an orchestrator that hands long briefs
    # to parallel workers, then reviews and merges their output, shares a few
    # words of each brief with every short command it runs.
    def test_reviewing_a_long_brief_with_a_short_command_is_fine(self) -> None:
        brief = (
            "Implement the scorecard in src/lean/analysis.py: add flow efficiency, "
            "peak work in progress and waiting time per stage; extend the golden "
            "snapshots, document the scorecard section, run pytest, ruff and mypy, "
            "commit on your branch and report the commit ids and check results."
        )
        s = Script()
        s.prompt("Build the scorecard with a worker")
        s.task("Lean scorecard", brief, "Async agent launched")
        s.bash("sed -n 1,80p src/lean/analysis.py", "def scorecard(): ...")
        s.bash("git merge --no-edit worker-scorecard", "Merge made")
        assert found(s, "unnecessary_handoff") == []

    def test_boundary_half_of_the_task_covered_is_uncertain(self) -> None:
        s = Script()
        s.task("Fetch release notes", "alpha beta gamma")
        s.bash("curl https://x.invalid/notes/release/fetch", "notes")
        [f] = found(s, "unnecessary_handoff")
        assert f.uncertain and f.confidence == 0.65

    def test_boundary_less_than_half_of_the_task_is_fine(self) -> None:
        s = Script()
        s.task("Fetch release notes", "alpha beta gamma delta")
        s.bash("curl https://x.invalid/notes/release/fetch", "notes")
        assert found(s, "unnecessary_handoff") == []


# --- repeated_reasoning ------------------------------------------------------


@pytest.mark.req("TER-DET-002")
class TestRepeatedReasoning:
    def test_restated_reasoning(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        s.think("I should find where the parser handles arguments.")
        s.search("argparse")
        s.think("I need to find where the parser handles arguments.")
        [f] = found(s, "repeated_reasoning")
        assert f.confidence >= 0.7 and 0 < f.share <= 1

    @pytest.mark.req("TER-LEN-004")
    def test_new_decision_is_not_a_restatement(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        s.think("I should find where the parser handles arguments.")
        s.think(
            "The parser handles arguments in main via argparse; add store_true option printing names."
        )
        assert found(s, "repeated_reasoning") == []

    def test_edit_in_between_resets(self) -> None:
        s = Script()
        s.think("I should find where the parser handles arguments.")
        s.edit("a.py")
        s.think("I should find where the parser handles arguments.")
        assert found(s, "repeated_reasoning") == []

    @pytest.mark.req("TER-LEN-004")
    def test_reasoning_that_uses_new_evidence_is_not_waste(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        s.think("I should find where the parser handles arguments.")
        s.search("parser", "src/cli.py:3: parser = build_parser()")
        # One new word in five (below the 25% decision bound), but it is what
        # the search just showed: new evidence.
        s.think("I need to find where the parser handles arguments in build_parser.")
        assert found(s, "repeated_reasoning") == []

    # Judged on real sessions: blocks repeating 80% or more of earlier key
    # words were waste, 76% to 79% were not.
    @pytest.mark.req("TER-LEN-004")
    def test_boundary_a_quarter_new_words_is_not_a_restatement(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        s.think("I should find where the parser handles arguments.")
        s.think("Find parser arguments quickly.")  # 1 of 4 key words new
        assert found(s, "repeated_reasoning") == []

    @pytest.mark.req("TER-LEN-004")
    def test_boundary_new_word_not_from_evidence_is_still_restated(self) -> None:
        s = Script()
        s.prompt("Add a verbose flag")
        s.think("I should find where the parser handles arguments.")
        s.search("parser", "src/cli.py:3: parser = build_parser()")
        s.think("I need to find where the parser handles arguments in parse_cli.")
        [f] = found(s, "repeated_reasoning")
        assert f.share == pytest.approx(0.8)


# --- regeneration ------------------------------------------------------------


@pytest.mark.req("TER-DET-002")
class TestRegeneration:
    CONTENT = "import time\n\ndef retry():\n    pass\n\ndef other():\n    return 1\n"

    def test_rewriting_own_file(self) -> None:
        s = Script()
        s.read("a.py")
        s.write("src/r.py", self.CONTENT)
        s.write("src/r.py", self.CONTENT + "# note\n")
        [f] = found(s, "regeneration")
        assert f.confidence == 0.8 and f.waste is LeanWaste.OVERPRODUCTION
        assert 0.8 < f.share < 1.0

    def test_rewriting_a_read_file_is_uncertain(self) -> None:
        s = Script()
        s.read("src/r.py", "   1\timport time\n   2\tdef retry():\n   3\t    pass\n")
        s.write(
            "src/r.py",
            "import time\ndef retry():\n    pass\ndef login():\n    return 1\n",
        )
        [f] = found(s, "regeneration")
        assert f.uncertain and f.confidence == 0.6

    # Judged on real sessions: a rewrite whose new file is mostly new content
    # (14% repeated) was real change, 35% and more repeated was waste.
    def test_boundary_rewrite_that_is_mostly_new_content_is_not_a_finding(
        self,
    ) -> None:
        old = "".join(f"line {i}\n" for i in range(3))
        s = Script()
        s.write("src/r.py", old)
        s.write("src/r.py", old + "".join(f"new {i}\n" for i in range(8)))
        assert found(s, "regeneration") == []  # 3 of 11 lines: 27% repeated
        s = Script()
        s.write("src/r.py", old)
        s.write("src/r.py", old + "".join(f"more {i}\n" for i in range(7)))
        [f] = found(s, "regeneration")  # 3 of 10 lines: 30% repeated
        assert f.share == pytest.approx(0.3) and f.confidence == 0.8

    def test_substantial_rewrite_is_fine(self) -> None:
        s = Script()
        s.write("src/r.py", self.CONTENT)
        s.write("src/r.py", "totally\ndifferent\ncontent\n")
        assert found(s, "regeneration") == []
