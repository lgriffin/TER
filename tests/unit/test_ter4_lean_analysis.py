"""Unit tests for the L2 model: facts, classification, graph, scorecard, A3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from ter4_lean_builder import FAIL, PASS, Script

from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import AnalysisEngine, EventKind, explain_batch
from ter.domain.lean import (
    DEFAULT_REGISTRY,
    ActivityClass,
    DetectorRegistry,
    EdgeType,
    Finding,
    FindingKind,
    FlowState,
    LeanWaste,
    Outcome,
    SessionView,
    ShellIntent,
    Stage,
    TerMeasure,
    build_a3,
    explain,
)
from ter.domain.lean.countermeasures import ActionKind
from ter.domain.lean.facts import (
    content_words,
    defined_identifiers,
    failure_signature,
    overlap,
    runs_check,
    shell_intent,
    source_lines,
    tool_paths,
    validation_outcome,
)
from ter.domain.lean.model import UNCERTAIN_BELOW


def analysis_of(script: Script, **kw: Any) -> Any:
    return explain(script.events, RegexTokenizer(), **kw)


# --- facts -------------------------------------------------------------------


@pytest.mark.req("TER-LEN-007")
@pytest.mark.parametrize(
    ("command", "intent"),
    [
        ("pytest tests/test_a.py -q", ShellIntent.VALIDATE),
        ("cd app && npm run test", ShellIntent.VALIDATE),
        ("python -m mypy src", ShellIntent.VALIDATE),
        ("go test ./...", ShellIntent.VALIDATE),
        ("python -c 'import a; print(a.f())'", ShellIntent.VALIDATE),
        ("ruff check src && pytest", ShellIntent.VALIDATE),
        ("pip install requests", ShellIntent.CHANGE),
        ("git commit -m x && pytest", ShellIntent.CHANGE),
        ("ls -la", ShellIntent.EXPLORE),
        ("git status", ShellIntent.EXPLORE),
        ("cd src && grep -r foo .", ShellIntent.EXPLORE),
        ("docker compose up", ShellIntent.OTHER),
        ("", ShellIntent.OTHER),
    ],
)
def test_shell_intent(command: str, intent: ShellIntent) -> None:
    assert shell_intent(command) is intent


@pytest.mark.req("TER-DET-005")
@pytest.mark.parametrize(
    ("command", "checks"),
    [
        ("pytest -q", True),
        ("sed -i 's/a/b/' x.py && pytest -q tests/unit", True),
        ("git checkout -- docs/x.md; mypy src/ | tail -1", True),
        ("ruff check src && git add -A && git commit -qm x", True),
        ("source .venv/bin/activate && python -m pytest -q", True),
        ("gh pr checks 12 --watch", True),
        ("pre-commit run --all-files", True),
        ("git add -A && git commit -qm x", False),
        ("mkdir -p out && python3 - <<'EOF'\nprint(1)\nEOF", False),
        ("echo pytest && rm -f x", False),
        ("grep -rn pytest docs", False),
    ],
)
def test_runs_check_sees_checks_beside_changes(command: str, checks: bool) -> None:
    assert runs_check(command) is checks


@pytest.mark.req("TER-LEN-007")
def test_ci_and_pre_commit_are_validation() -> None:
    assert shell_intent("gh run watch 42") is ShellIntent.VALIDATE
    assert shell_intent("pre-commit run --all-files") is ShellIntent.VALIDATE
    assert shell_intent("gh pr view 12") is ShellIntent.OTHER


@pytest.mark.req("TER-DET-006")
@pytest.mark.parametrize(
    ("output", "outcome"),
    [
        (FAIL, Outcome.FAILED),
        ("Traceback (most recent call last):\n  File x", Outcome.FAILED),
        ("error[E0308]: mismatched types", Outcome.FAILED),
        ("Found 3 errors in 2 files", Outcome.FAILED),
        ("Process exited with exit code 2", Outcome.FAILED),
        (PASS, Outcome.PASSED),
        ("All checks passed!", Outcome.PASSED),
        ("Success: no issues found in 3 source files", Outcome.PASSED),
        ("0 errors, 0 warnings", Outcome.PASSED),
        ("a.txt\nb.txt", Outcome.UNKNOWN),
        ("", Outcome.UNKNOWN),
    ],
)
def test_validation_outcome(output: str, outcome: Outcome) -> None:
    assert validation_outcome(output) is outcome


@pytest.mark.req("TER-DET-006")
def test_failure_signature_ignores_timing_but_not_the_failure() -> None:
    assert failure_signature(FAIL) == failure_signature(FAIL.replace("0.12s", "9.9s"))
    assert failure_signature(FAIL) != failure_signature(
        FAIL.replace("test_x", "test_z")
    )
    assert failure_signature("weird") == failure_signature("weird")


@pytest.mark.req("TER-DET-002")
def test_small_facts() -> None:
    assert tool_paths({"file_path": "a.py"}) == ("a.py",)
    assert tool_paths({"pattern": "x"}) == ()
    assert "slugify" in defined_identifiers("def slugify(s):\n    pass\nMAX_SIZE = 3")
    assert "MAX_SIZE" in defined_identifiers("MAX_SIZE = 3")
    assert source_lines("   1\tx = 1\n   2\t\n   3→y") == frozenset({"x = 1", "y"})
    assert "the" not in content_words("the parser handles arguments")
    assert overlap(frozenset(), frozenset({"a"})) == 0.0
    assert overlap(frozenset({"a", "b"}), frozenset({"a"})) == 1.0


# --- stages and classes ------------------------------------------------------


@pytest.mark.req("TER-LEN-001", "TER-LEN-007")
def test_every_event_gets_a_stage_and_classified_basis() -> None:
    s = Script()
    s.prompt("fix")
    s.think("look first")
    s.read("a.py")
    s.todo("plan")
    s.edit("a.py")
    s.bash("pip install x")
    s.bash("pytest -q", PASS)
    s.say("working")
    s.say("done")
    a = analysis_of(s)
    stages = [c.stage for c in a.classifications]
    assert stages == [
        Stage.INTENT,
        Stage.PLAN,
        Stage.EXPLORE,
        Stage.EXPLORE,
        Stage.PLAN,
        Stage.PLAN,
        Stage.IMPLEMENT,
        Stage.IMPLEMENT,
        Stage.IMPLEMENT,
        Stage.IMPLEMENT,
        Stage.VALIDATE,
        Stage.VALIDATE,
        Stage.RESPOND,
        Stage.RESPOND,
    ]
    classes = {c.event_id: c for c in a.classifications}
    events = s.events
    assert classes[events[0].id].activity_class is None
    assert classes[events[6].id].activity_class is ActivityClass.VALUE_ADDING
    assert (
        classes[events[8].id].activity_class is ActivityClass.NECESSARY_NON_VALUE_ADDING
    )
    assert (
        classes[events[12].id].activity_class
        is ActivityClass.NECESSARY_NON_VALUE_ADDING
    )
    assert classes[events[13].id].activity_class is ActivityClass.VALUE_ADDING
    assert all(c.basis for c in a.classifications)


@pytest.mark.req("TER-ANL-021", "TER-LEN-002")
def test_confident_waste_is_avoidable_and_uncertain_is_kept_apart() -> None:
    s = Script()
    s.prompt("x")
    s.read("src/a.py", "def a(): pass")
    repeat, _ = s.read("src/a.py", "def a(): pass")
    s.read("src/b.py", None)
    unsure, _ = s.read("src/b.py", None)
    s.say("used a.py and b.py")
    a = analysis_of(s)
    by_id = {c.event_id: c for c in a.classifications}
    assert by_id[repeat.id].activity_class is ActivityClass.AVOIDABLE
    assert by_id[repeat.id].flow is FlowState.REPEATING
    assert by_id[repeat.id].basis.startswith("repeated_exploration:")
    # An unobserved repeat is uncertain: its own bucket, never avoidable.
    assert by_id[unsure.id].activity_class is ActivityClass.NECESSARY_NON_VALUE_ADDING
    assert by_id[unsure.id].uncertain_share == 1.0
    assert "uncertain repeated_exploration:" in by_id[unsure.id].basis
    sc = a.scorecard
    assert sc.uncertain_findings == 1
    assert dict(sc.activity_tokens)["uncertain"] > 0
    assert sum(n for _, n in sc.activity_tokens) == sc.generated_tokens
    assert sum(n for _, n in sc.flow_tokens) == sc.generated_tokens
    assert UNCERTAIN_BELOW == 0.7


@pytest.mark.req("TER-ANL-022")
def test_uncertain_waste_counts_until_verified() -> None:
    s = Script()
    s.prompt("x")
    s.read("src/b.py", None)
    unsure, _ = s.read("src/b.py", None)
    s.say("done")
    a = analysis_of(s)
    [f] = [f for f in a.findings if f.detector == "repeated_exploration"]
    assert f.uncertain
    c = next(c for c in a.classifications if c.event_id == unsure.id)
    assert c.flow is FlowState.REPEATING and c.uncertain_basis == f.id
    sc = a.scorecard
    # Counted: in the waste totals, the finding count and flow efficiency ...
    assert sc.findings == 1 and sc.uncertain_findings == 1
    assert sc.waste_tokens == sc.uncertain_waste_tokens > 0
    assert dict(sc.flow_tokens)[FlowState.REPEATING] == sc.waste_tokens
    assert sc.flow_efficiency_tokens is not None and sc.flow_efficiency_tokens < 1.0
    # ... and charged to the finding, so the A3 Pareto reconciles.
    assert round(a.allocated_waste_tokens()[f.id]) == sc.waste_tokens


@pytest.mark.req("TER-SCR-002", "TER-FLW-001")
def test_scorecard_dimensions_and_explained_composite() -> None:
    s = Script()
    s.prompt("fix")
    s.read("a.py")
    s.bash("pytest -q", FAIL)
    s.edit("a.py", "1")
    s.bash("pytest -q", FAIL)
    s.edit("a.py", "2")
    s.bash("pytest -q", PASS)
    s.say("done")
    sc = analysis_of(s, ter=TerMeasure(0.5, "test")).scorecard
    flows = dict(sc.flow_tokens)
    assert flows[FlowState.REWORKING] > 0 and flows[FlowState.RECOVERING] > 0
    assert sc.flow_efficiency_tokens == pytest.approx(
        (flows[FlowState.PROGRESSING] + flows[FlowState.RECOVERING])
        / sc.generated_tokens
    )
    assert sc.flow_efficiency_time is not None and 0 < sc.flow_efficiency_time < 1
    assert sc.rework_cycles == 1 and sc.iterations == 1
    assert sc.composite is not None
    names = [n for n, _, _ in sc.composite.components]
    assert names == ["Flow efficiency (tokens)", "Flow efficiency (time)", "TER"]
    assert sc.composite.value == pytest.approx(
        sum(v * w for _, v, w in sc.composite.components)
    )
    assert "mean" in sc.composite.formula
    assert sc.as_dict()["ter"] == {"value": 0.5, "method": "test"}
    assert sc.developer_seconds == 0


@pytest.mark.req("TER-SCR-002")
def test_scorecard_without_time_or_activity() -> None:
    empty = analysis_of(Script())
    assert empty.scorecard.flow_efficiency_tokens is None
    assert empty.scorecard.composite is None
    assert empty.scorecard.activity_share(ActivityClass.AVOIDABLE) == 0.0
    s = Script(timed=False)
    s.prompt("x")
    s.say("y")
    sc = analysis_of(s).scorecard
    assert sc.flow_efficiency_time is None
    assert sc.composite is not None and len(sc.composite.components) == 1


@pytest.mark.req("TER-LEN-007")
def test_value_stream_has_every_stage_in_order() -> None:
    s = Script()
    s.prompt("x")
    s.read("a.py")
    s.read("a.py")
    s.say("y")
    vs = analysis_of(s).value_stream
    assert [v.stage for v in vs] == list(Stage)
    explore = vs[1]
    assert explore.steps == 2 and explore.avoidable_tokens > 0 and explore.findings
    assert vs[0].steps == 1 and vs[0].seconds == 0


@pytest.mark.req("TER-LEN-002")
def test_time_attribution_gives_tool_time_to_the_request() -> None:
    s = Script()
    s.prompt("x")
    request, _ = s.read("a.py")
    a = analysis_of(s)
    steps = {st.event_id: st for st in a.steps}
    assert steps[request.id].seconds == 2.0  # its own gap plus the tool's running time


# --- evidence graph ----------------------------------------------------------


@pytest.mark.req("TER-GRF-002", "TER-GRF-003")
def test_evidence_graph_edges_and_export() -> None:
    s = Script()
    prompt = s.prompt("fix a")
    read, read_done = s.read("a.py")
    edit1, _ = s.edit("a.py", "1")
    run1, fail = s.bash("pytest -q", FAIL)
    edit2, _ = s.edit("a.py", "2")
    run2, _ = s.bash("pytest -q", PASS)
    s.read("b.py")
    again, _ = s.read("b.py")
    s.say("done")
    g = analysis_of(s).graph
    edges = {(e.source, e.target, e.type) for e in g.edges}
    assert read_done is not None and fail is not None
    assert (read_done.id, read.id, EdgeType.COMPLETES) in edges
    assert (edit1.id, read_done.id, EdgeType.MOTIVATED_BY) in edges
    assert (edit1.id, prompt.id, EdgeType.MOTIVATED_BY) in edges
    assert (run1.id, edit1.id, EdgeType.VALIDATES) in edges
    assert (edit2.id, fail.id, EdgeType.CORRECTS) in edges
    assert (run2.id, edit2.id, EdgeType.VALIDATES) in edges
    assert (again.id, s.events[-5].id, EdgeType.REPEATS) in edges
    assert prompt.id in g.ancestors(edit2.id)
    assert g.edges_of(EdgeType.CORRECTS)
    exported = json.loads(json.dumps(g.as_dict()))
    assert exported["schema"] == "ter.evidence/0.1"
    ids = {n["id"] for n in exported["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in exported["edges"])
    assert all(e.target != e.source for e in g.edges)


# --- registry ----------------------------------------------------------------


@dataclass(frozen=True)
class _EveryPrompt:
    id: str = "every_prompt"
    waste: LeanWaste = LeanWaste.WAITING
    kind: FindingKind = FindingKind.RISK
    summary: str = "test plugin"
    confidence_rule: str = "always 0.5"

    def detect(self, view: SessionView) -> list[Finding]:
        return [
            Finding(
                id=f"{self.id}:{s.event_id}",
                detector=self.id,
                waste=self.waste,
                kind=self.kind,
                activity_class=ActivityClass.AVOIDABLE,
                confidence=0.5,
                title="prompt",
                explanation="a prompt",
                evidence=(s.event_id,),
                waste_events=(),
                share=0.0,
                subject="",
                tokens=0,
                context_tokens=0,
                seconds=0.0,
            )
            for s in view.steps
            if s.kind is EventKind.PROMPT
        ]


def test_detectors_are_plugins() -> None:
    registry = DetectorRegistry([_EveryPrompt()])
    assert "every_prompt" in registry and len(registry) == 1
    assert registry.get("every_prompt").id == "every_prompt"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_EveryPrompt())
    s = Script()
    s.prompt("a")
    s.prompt("b")
    a = analysis_of(s, registry=registry)
    assert [f.detector for f in a.findings] == ["every_prompt", "every_prompt"]
    assert a.detectors == (("every_prompt", "waiting", "risk", "always 0.5"),)
    assert len(DEFAULT_REGISTRY) == 17
    assert all(d.confidence_rule and d.summary for d in DEFAULT_REGISTRY)


# --- countermeasures and A3 --------------------------------------------------


def _messy() -> Script:
    s = Script()
    s.prompt("Fix parse in src/a.py and keep tests green")
    s.read("src/a.py", "def parse(): pass")
    s.read("src/a.py", "def parse(): pass")
    s.bash("pytest tests/test_a.py -q", FAIL)
    s.edit("src/a.py", "1")
    s.bash("pytest tests/test_a.py -q", FAIL)
    s.edit("src/b.py", "1")
    s.say("Done")
    return s


@pytest.mark.req("TER-RPT-005")
def test_countermeasures_are_derived_from_findings() -> None:
    a = analysis_of(_messy())
    report = build_a3(a, ["Fix parse in src/a.py"])
    detectors = {c.detector for c in report.countermeasures}
    assert detectors == {f.detector for f in a.findings}
    rework = next(c for c in report.countermeasures if c.detector == "rework_cycle")
    assert any("pytest tests/test_a.py -q" in (x.snippet or "") for x in rework.actions)
    kinds = {x.kind for c in report.countermeasures for x in c.actions}
    assert {ActionKind.CLAUDE_MD, ActionKind.HOOK} <= kinds
    for c in report.countermeasures:
        assert set(c.addresses) <= {f.id for f in a.findings}
        for action in c.actions:
            if action.language == "json" or action.language == "json+bash":
                json.loads((action.snippet or "").partition("\n\n")[0])
    reread = next(
        c for c in report.countermeasures if c.detector == "repeated_exploration"
    )
    assert "src/a.py" in json.dumps(reread.as_dict())


@pytest.mark.req("TER-RPT-005")
def test_no_findings_means_no_countermeasures() -> None:
    s = Script()
    s.prompt("x")
    s.read("a.py", "def a(): pass")
    s.edit("a.py")
    s.bash("pytest -q", PASS)
    s.say("a.py done")
    report = build_a3(analysis_of(s), ["x"])
    assert report.countermeasures == ()
    assert [f.metric for f in report.follow_up] == [
        "Agentic flow efficiency (tokens)",
        "Avoidable share of generated tokens",
    ]


@pytest.mark.req("TER-RPT-005")
def test_every_detector_has_a_countermeasure_and_a_follow_up() -> None:
    from ter.domain.lean import countermeasures as cm
    from ter.domain.lean.drift import EVIDENCE_DETECTORS
    from ter.domain.lean.surface import GROUNDED_DETECTORS

    every = {
        d.id for d in (*DEFAULT_REGISTRY, *GROUNDED_DETECTORS, *EVIDENCE_DETECTORS)
    }
    assert set(cm._CATALOGUE) == every
    assert set(cm._MEASURES) == every


@pytest.mark.req("TER-RPT-003")
def test_a3_structure() -> None:
    a = analysis_of(_messy(), ter=TerMeasure(0.4, "t"))
    report = build_a3(a, ["Fix parse in src/a.py " + "x" * 200])
    assert report.title.endswith("…") and len(report.title) <= 120
    d = report.as_dict()
    assert list(d) == [
        "schema",
        "title",
        "session_id",
        "background",
        "problem",
        "current_state",
        "analysis",
        "root_causes",
        "findings",
        "countermeasures",
        "follow_up",
        "detectors",
        "lean_concepts",
        "context_inventory",
    ]
    assert d["schema"] == "ter.a3/0.1"
    assert "avoidable" in report.problem and "risk" in report.problem
    assert report.pareto and report.pareto[0].tokens >= report.pareto[-1].tokens
    assert all(not f.uncertain for f in report.root_causes[:1])
    json.dumps(d)


@pytest.mark.req("TER-RPT-003")
def test_a3_edge_cases() -> None:
    empty = build_a3(analysis_of(Script()))
    assert empty.title == "Agent session"
    assert "no agent activity" in empty.problem


@pytest.mark.req("TER-ANL-010")
def test_engine_explain_equals_batch_explain() -> None:
    s = _messy()
    engine = AnalysisEngine(RegexTokenizer())
    for event in s.events:
        engine.apply(event)
        engine.apply(event)
    assert engine.explain() == explain_batch(s.events, RegexTokenizer())
    assert engine.explain() == analysis_of(s)


# --- review fixes --------------------------------------------------------------


@pytest.mark.req("TER-LEN-007")
@pytest.mark.parametrize(
    ("command", "intent"),
    [
        # A runner's name as an argument is not a run.
        ("echo pytest", ShellIntent.EXPLORE),
        ("echo 'run pytest before merging'", ShellIntent.EXPLORE),
        ("grep -rn pytest .", ShellIntent.EXPLORE),
        ("cat <<EOF\npytest -q\nrm -rf build\nEOF", ShellIntent.EXPLORE),
        ("echo rm -rf build", ShellIntent.EXPLORE),
        ('git log --grep "mypy"', ShellIntent.EXPLORE),
        # A runner in command position is, wherever the segment is.
        ("echo start; pytest -q", ShellIntent.VALIDATE),
        ("cd app\nnpm test", ShellIntent.VALIDATE),
        ("pytest -q 2>&1 | tail -5", ShellIntent.VALIDATE),
        (".venv/bin/pytest -q", ShellIntent.VALIDATE),
        ("PYTHONPATH=src uv run pytest", ShellIntent.VALIDATE),
        ("timeout 60 sudo -E pytest -x", ShellIntent.VALIDATE),
        ("python -m pytest", ShellIntent.VALIDATE),
        ("python - <<'EOF'\nimport os; os.remove('x')\nEOF", ShellIntent.VALIDATE),
        ("python -m pip install requests", ShellIntent.CHANGE),
        ("ls | tee listing.txt", ShellIntent.CHANGE),
        ('echo "unbalanced', ShellIntent.EXPLORE),
    ],
)
def test_shell_intent_reads_command_positions(
    command: str, intent: ShellIntent
) -> None:
    assert shell_intent(command) is intent


@pytest.mark.req("TER-LEN-007")
def test_printing_a_runner_name_does_not_validate_an_edit() -> None:
    s = Script()
    s.prompt("Fix a.py")
    s.read("src/a.py", "def a(): pass")
    s.edit("src/a.py")
    s.bash("echo pytest", "pytest")
    s.say("Fixed src/a.py")
    a = analysis_of(s)
    assert [f.detector for f in a.findings] == ["unvalidated_implementation"]


def _countermeasure(command: str | None) -> Any:
    s = Script()
    s.prompt("Fix a.py")
    s.read("src/a.py", "def a(): pass")
    if command is not None:
        s.bash(command, PASS)
    s.edit("src/a.py")
    s.say("Fixed src/a.py")
    report = build_a3(analysis_of(s), ["Fix a.py"])
    return next(
        c for c in report.countermeasures if c.detector == "unvalidated_implementation"
    )


@pytest.mark.req("TER-RPT-005")
def test_no_observed_check_means_no_executable_edit_hook() -> None:
    cm = _countermeasure(None)
    assert ActionKind.HOOK not in {a.kind for a in cm.actions}
    assert all("<your test command" not in (a.snippet or "") for a in cm.actions)
    assert any("no edit hook is generated" in a.text for a in cm.actions)
    observed = _countermeasure("pytest -q")
    [hook] = [a for a in observed.actions if a.kind is ActionKind.HOOK]
    assert "&& pytest -q >/dev/null" in (hook.snippet or "")


@pytest.mark.req("TER-RPT-005")
def test_edit_hook_quotes_the_observed_command() -> None:
    import shlex
    import subprocess

    def hook_command(command: str) -> str:
        cm = _countermeasure(command)
        [hook] = [a for a in cm.actions if a.kind is ActionKind.HOOK]
        settings = json.loads(hook.snippet or "")
        check: str = settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
        return check

    # An observed check that fails, with an apostrophe, quotes and a comment.
    command = 'pytest -k "it\'s" -q 2>/dev/null; false # isn\'t "done" `yet`'
    check = hook_command(command)
    diagnostic = check.split("|| { echo ", 1)[1].rsplit(" >&2; exit 2; }", 1)[0]
    assert shlex.split(diagnostic) == [f"{command} fails after this edit"]
    ran = subprocess.run(
        ["bash", "-c", check],
        env={"CLAUDE_PROJECT_DIR": ".", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 2
    assert ran.stderr == f"{command} fails after this edit\n"
    assert hook_command("pytest tests/test_a.py -q").startswith(
        'cd "$CLAUDE_PROJECT_DIR" && pytest tests/test_a.py -q >/dev/null'
    )


@pytest.mark.req("TER-LEN-007")
def test_results_without_call_ids_pair_only_when_unambiguous() -> None:
    from ter.domain import Actor, Event, Provenance, ToolCall, ToolKind, make_event_id
    from ter.domain.lean.steps import StepLog

    def event(n: int, kind: EventKind, tool: ToolCall, text: str = "") -> Event:
        return Event(
            id=make_event_id("s", n, kind.value),
            session_id="s",
            sequence=n,
            kind=kind,
            actor=Actor.TOOL if kind is EventKind.TOOL_COMPLETED else Actor.ASSISTANT,
            text=text,
            provenance=Provenance(source="t", record_id=f"r{n}"),
            tool=tool,
        )

    bash = ToolCall("Bash", ToolKind.EXEC_SHELL, None, {"command": "pytest -q"})
    read = ToolCall("Read", ToolKind.FS_READ, None, {"file_path": "a.py"})
    result = ToolCall("", ToolKind.OTHER, None)
    log = StepLog()
    for e in (
        event(0, EventKind.TOOL_REQUESTED, bash),
        event(1, EventKind.TOOL_COMPLETED, result, FAIL),
        # Two id-less requests in flight: their results cannot be told apart.
        event(2, EventKind.TOOL_REQUESTED, read),
        event(3, EventKind.TOOL_REQUESTED, bash),
        event(4, EventKind.TOOL_COMPLETED, result, "x"),
    ):
        log.add(e, 1)
    steps = log.steps()
    paired = steps[1]
    assert paired.request_index == 0
    assert paired.command == "pytest -q"
    assert paired.shell is ShellIntent.VALIDATE
    assert paired.outcome is Outcome.FAILED
    assert steps[4].request_index is None
    assert steps[4].stage is Stage.EXPLORE


@pytest.mark.req("TER-DET-001")
def test_files_changed_unread_under_one_prompt_get_distinct_ids() -> None:
    s = Script()
    s.prompt("Add login")
    s.write("src/login.py", "def login(): pass")
    s.write("templates/login.html", "<form></form>")
    s.say("Added login")
    a = analysis_of(s)
    premature = [f for f in a.findings if f.detector == "premature_implementation"]
    assert len(premature) == 2
    assert len({f.id for f in premature}) == 2
    for f in premature:
        assert a.finding(f.id) is f
