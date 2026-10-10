"""Model routing (L3): task classification (TER-RTE-005), roles from the
routing profile (TER-RTE-001), escalation with one route.escalated event
(TER-RTE-002) and keeping the profile without an evidenced signal
(TER-RTE-003). The router is advisory: it returns decisions and events."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from ter4_lean_builder import FAIL, FAIL_OTHER, PASS, Script
from ter4_shop_repo import SHOP, at, shop_repo

from ter import bootstrap
from ter.adapters.driven.import_linter import ImportLinterContracts
from ter.adapters.driven.in_memory import InMemoryRoutingProfiles
from ter.adapters.driven.routing_profiles import default_routing_profiles
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.adapters.driving.cli import main
from ter.application.ground import ground_session
from ter.application.route import RouteSession
from ter.bootstrap.capabilities import repository_evidence
from ter.domain import EventKind, TokenUsage
from ter.domain.lean import LeanAnalysis, explain
from ter.domain.lean.detectors import SessionView
from ter.domain.lean.grounding import RepositoryGrounding
from ter.domain.routing import (
    CLASSIFICATION_RULES,
    Dimension,
    Level,
    ModelBinding,
    RouteEscalation,
    RoutingPlan,
    RoutingProfile,
    RoutingProfileError,
    Scope,
    TaskClassification,
    TaskKind,
    UnknownRoleError,
    ValidationNeed,
    classify_tasks,
    route_session,
)

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "sample_sessions" / "example_session.jsonl"
PRICING = "src/app/domain/pricing.py"
MODEL = "src/app/domain/model.py"
SUMMARY = "src/app/reports/summary.py"


def profile(**changes: object) -> RoutingProfile:
    base = RoutingProfile(
        name="test",
        roles={
            "explore": ModelBinding("p", "small"),
            "implement": ModelBinding("p", "mid"),
            "review": ModelBinding("p", "mid"),
            "escalate": ModelBinding("p", "large"),
        },
        task_roles={
            TaskKind.READ_ONLY: "explore",
            TaskKind.VALIDATE: "review",
            TaskKind.CHANGE: "implement",
        },
        escalation={"explore": "implement", "implement": "escalate"},
        escalate_on=frozenset({"rework_cycle"}),
        source="test",
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def analyse(s: Script, g: RepositoryGrounding | None = None) -> LeanAnalysis:
    if g is None:
        return explain(s.events, RegexTokenizer())
    return explain(s.events, RegexTokenizer(), repository=g)


def classify(
    s: Script, g: RepositoryGrounding | None = None
) -> tuple[TaskClassification, ...]:
    a = analyse(s, g)
    return classify_tasks(SessionView.of(a.steps, a.intent, a.repository), a.findings)


def plan(s: Script, p: RoutingProfile | None = None) -> RoutingPlan:
    a = analyse(s)
    tasks = classify_tasks(SessionView.of(a.steps, a.intent), a.findings)
    return route_session(
        "s", a.steps, tasks, a.findings, p or profile(), first_sequence=len(s.events)
    )


def rework(s: Script) -> None:
    """A task whose check fails the same way after a fix: rework_cycle fires."""
    s.prompt("fix the failing test in src/a.py")
    s.read("src/a.py")
    s.bash("pytest -q", FAIL)
    s.edit("src/a.py")
    s.bash("pytest -q", FAIL.replace("0.12", "0.30"))


def ground(s: Script, root: Path) -> RepositoryGrounding:
    return ground_session(
        s.events, repository_evidence(root, "python-ast"), ImportLinterContracts()
    )


# --- classification (TER-RTE-005) ------------------------------------------


@pytest.mark.req("TER-RTE-005")
class TestClassification:
    def test_every_task_gets_all_five_classes_with_rule_and_evidence(self) -> None:
        s = Script()
        prompt = s.prompt("fix the rounding in src/a.py")
        read, _ = s.read("src/a.py")
        edit, _ = s.edit("src/a.py")
        s.bash("pytest -q", PASS)
        s.say("done")
        [t] = classify(s)
        assert t.prompt == prompt.id and t.kind is TaskKind.CHANGE
        assert [e.dimension for e in t.evidence] == list(Dimension)
        assert all(e.rule == CLASSIFICATION_RULES[e.dimension] for e in t.evidence)
        assert (t.complexity, t.ambiguity, t.risk) == (
            Level.LOW,
            Level.LOW,
            Level.MEDIUM,
        )
        assert (t.scope, t.validation) == (Scope.FILE, ValidationNeed.TESTS)
        assert t.of(Dimension.AMBIGUITY).events == (prompt.id,)
        assert t.of(Dimension.RISK).events == (edit.id,)
        assert t.of(Dimension.SCOPE).reason == "edits src/a.py"
        d = t.as_dict()
        assert d["classes"] == {
            "complexity": "low",
            "ambiguity": "low",
            "risk": "medium",
            "repository_scope": "file",
            "validation_needs": "tests",
        }
        assert read.id not in t.of(Dimension.SCOPE).events  # edits win over reads

    def test_tasks_split_at_prompts_and_the_preamble_has_no_prompt(self) -> None:
        s = Script()
        s.read("README.md")
        s.prompt("what does src/a.py do?")
        s.read("src/a.py")
        s.say("it adds")
        s.prompt("run the tests")
        s.bash("pytest -q", PASS)
        tasks = classify(s)
        assert [t.kind for t in tasks] == [
            TaskKind.READ_ONLY,
            TaskKind.READ_ONLY,
            TaskKind.VALIDATE,
        ]
        assert tasks[0].prompt is None and tasks[0].ambiguity is Level.HIGH
        assert [t.first for t in tasks] == [0, 2, 6]

    @pytest.mark.parametrize(
        ("files", "level"),
        [
            (0, Level.LOW),
            (1, Level.LOW),
            (2, Level.MEDIUM),
            (3, Level.MEDIUM),
            (4, Level.HIGH),
        ],
    )
    def test_complexity_bands_by_files_edited(self, files: int, level: Level) -> None:
        s = Script()
        s.prompt("tidy src/")
        for i in range(files):
            s.edit(f"src/m{i}.py")
        [t] = classify(s)
        assert t.complexity is level

    def test_distinct_failures_raise_complexity_one_level(self) -> None:
        s = Script()
        s.prompt("fix src/a.py")
        s.edit("src/a.py")
        failing, _ = s.bash("pytest -q", FAIL)
        s.bash("pytest -q tests/test_b.py", FAIL_OTHER)
        [t] = classify(s)
        assert t.complexity is Level.MEDIUM
        assert "2 distinct signatures" in t.of(Dimension.COMPLEXITY).reason

    @pytest.mark.parametrize(
        ("prompt", "level"),
        [
            ("fix the bug in src/a.py", Level.LOW),
            ("make it faster", Level.MEDIUM),
            ("why is it slow? can you look", Level.HIGH),
            ("e.g. speed it up", Level.MEDIUM),  # an abbreviation names no file
        ],
    )
    def test_ambiguity_from_the_prompt(self, prompt: str, level: Level) -> None:
        s = Script()
        s.prompt(prompt)
        s.say("ok")
        [t] = classify(s)
        assert t.ambiguity is level

    def test_a_refinement_of_a_named_intent_is_medium(self) -> None:
        s = Script()
        s.prompt("fix the rounding bug in src/pricing.py so totals round to cents")
        s.say("ok")
        s.prompt("also make the rounding of totals use bankers rounding")
        s.say("ok")
        tasks = classify(s)
        assert tasks[1].ambiguity in (Level.MEDIUM, Level.LOW)
        assert tasks[1].ambiguity is not Level.HIGH

    @pytest.mark.parametrize(
        ("path", "level"),
        [
            ("pyproject.toml", Level.HIGH),
            (".github/workflows/ci.yml", Level.HIGH),
            ("src/a.py", Level.MEDIUM),
            ("tests/test_a.py", Level.LOW),
            ("docs/guide.md", Level.LOW),
        ],
    )
    def test_risk_by_what_is_edited(self, path: str, level: Level) -> None:
        s = Script()
        s.prompt("change it")
        s.edit(path)
        [t] = classify(s)
        assert t.risk is level

    def test_a_destructive_command_is_high_risk(self) -> None:
        s = Script()
        s.prompt("clean up")
        rm, _ = s.bash("rm -rf build/")
        [t] = classify(s)
        assert t.risk is Level.HIGH and rm.id in t.of(Dimension.RISK).events

    @pytest.mark.parametrize(
        ("paths", "scope"),
        [
            ((), Scope.NONE),
            (("src/a.py",), Scope.FILE),
            (("src/a.py", "src/b.py"), Scope.DIRECTORY),
            (("src/a.py", "tests/test_a.py"), Scope.REPOSITORY),
        ],
    )
    def test_scope_by_files_and_directories(
        self, paths: tuple[str, ...], scope: Scope
    ) -> None:
        s = Script()
        s.prompt("change it")
        for p in paths:
            s.edit(p)
        [t] = classify(s)
        assert t.scope is scope

    def test_a_read_only_task_is_scoped_by_what_it_reads(self) -> None:
        s = Script()
        s.prompt("explain the module")
        s.read("src/a.py")
        s.read("lib/b.py")
        [t] = classify(s)
        assert (
            t.scope is Scope.REPOSITORY
            and "reads 2 files" in t.of(Dimension.SCOPE).reason
        )

    @pytest.mark.parametrize(
        ("path", "need"),
        [
            (None, ValidationNeed.NONE),
            ("README.md", ValidationNeed.CHECK),
            ("config.yaml", ValidationNeed.CHECK),
            ("src/a.py", ValidationNeed.TESTS),
        ],
    )
    def test_validation_needs(self, path: str | None, need: ValidationNeed) -> None:
        s = Script()
        s.prompt("change it")
        if path:
            s.edit(path)
        [t] = classify(s)
        assert t.validation is need

    def test_classes_never_depend_on_token_counts(self) -> None:
        short, long = Script(), Script()
        for s, words in ((short, "x"), (long, "x " * 2000)):
            s.prompt("fix src/a.py " + words)
            s.edit("src/a.py", "a", "b" + words)
            s.say("done " + words)
        assert [e.value for e in classify(short)[0].evidence] == [
            e.value for e in classify(long)[0].evidence
        ]


@pytest.mark.req("TER-RTE-005")
class TestGroundedClassification:
    @pytest.fixture
    def shop(self, tmp_path: Path) -> Path:
        return shop_repo(tmp_path / "shop")

    def test_a_named_seed_lowers_ambiguity_and_the_surface_names_the_tests(
        self, shop: Path
    ) -> None:
        s = Script()
        s.prompt("Fix the rounding in price_with_tax")
        s.read(at(PRICING), SHOP[PRICING])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        [t] = classify(s, ground(s, shop))
        assert t.grounded and t.ambiguity is Level.LOW
        assert PRICING in t.of(Dimension.AMBIGUITY).reason
        assert (
            t.scope is Scope.FILE and t.of(Dimension.SCOPE).reason == f"edits {PRICING}"
        )
        assert "tests/test_pricing.py" in t.of(Dimension.VALIDATION).reason

    def test_an_unrelated_edit_is_high_risk(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        stray, _ = s.edit(at(SUMMARY), "len(rows)", "len(list(rows))")
        [t] = classify(s, ground(s, shop))
        assert t.risk is Level.HIGH and stray.id in t.of(Dimension.RISK).events
        assert t.scope is Scope.REPOSITORY

    def test_a_contract_break_is_high_risk(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.edit(
            at(PRICING), "def price_with_tax", "import requests\n\n\ndef price_with_tax"
        )
        [t] = classify(s, ground(s, shop))
        assert t.risk is Level.HIGH
        assert "architecture contract" in t.of(Dimension.RISK).reason


# --- profiles and roles (TER-RTE-001) --------------------------------------


@pytest.mark.req("TER-RTE-001")
class TestProfiles:
    def test_a_profile_must_define_every_role_it_names(self) -> None:
        with pytest.raises(RoutingProfileError, match="undefined role"):
            profile(escalation={"implement": "frontier"})
        with pytest.raises(RoutingProfileError, match="no role to task kind"):
            profile(task_roles={TaskKind.CHANGE: "implement"})
        with pytest.raises(RoutingProfileError, match="cycle"):
            profile(escalation={"implement": "escalate", "escalate": "implement"})
        with pytest.raises(UnknownRoleError):
            profile().binding("frontier")

    def test_decisions_name_roles_only_and_survive_a_model_swap(self) -> None:
        s = Script()
        rework(s)
        s.prompt("explain src/b.py")
        s.read("src/b.py")
        first = plan(s)
        swapped = profile(
            roles={
                r: ModelBinding("q", f"{b.model}-v2")
                for r, b in profile().roles.items()
            }
        )
        second = plan(s, swapped)
        roles = set(profile().roles)
        assert {d.role for d in first.decisions} | {
            d.final_role for d in first.decisions
        } <= roles
        assert [d.as_dict() for d in first.decisions] == [
            d.as_dict() for d in second.decisions
        ]
        assert [e.text for e in first.events] == [e.text for e in second.events]
        assert not any("mid" in e.text or "large" in e.text for e in first.events)

    def test_the_shipped_default_profile_routes_the_sample_session(self) -> None:
        profiles = default_routing_profiles()
        routed = RouteSession(
            bootstrap.session_source_for(SAMPLE), RegexTokenizer(), profiles
        )(SAMPLE)
        assert routed.plan.profile.name == profiles.default() == "default"
        assert all(d.role in routed.plan.profile.roles for d in routed.plan.decisions)


# --- escalation (TER-RTE-002, TER-RTE-003) ----------------------------------


@pytest.mark.req("TER-RTE-002")
class TestEscalation:
    def test_an_evidenced_signal_escalates_with_one_event(self) -> None:
        s = Script()
        rework(s)
        p = plan(s)
        [d] = p.decisions
        assert d.escalated and (d.role, d.final_role) == ("implement", "escalate")
        assert d.signal is not None and d.signal.detector == "rework_cycle"
        [event] = p.events
        assert event is d.event and event.kind is EventKind.ROUTE_ESCALATED
        assert event.sequence == len(s.events) and event.usage is None
        e = RouteEscalation.parse(event.text)
        assert e == d.escalation
        assert e is not None
        assert (e.signal, e.source_role, e.target_role) == (
            "rework_cycle",
            "implement",
            "escalate",
        )
        # Latency: the task's wall time up to the signal (one second a step).
        assert e.latency_ms == 1000 * (d.signal.at - d.task.first + 1) - 1000

    def test_the_event_records_the_token_usage_spent_before_the_signal(self) -> None:
        s = Script()
        s.prompt("fix the failing test in src/a.py")
        answer = s.say("Looking.")
        i = s.events.index(answer)
        s.events[i] = replace(
            answer,
            usage=TokenUsage(input_tokens=700, output_tokens=40, cache_read_tokens=5),
        )
        s.bash("pytest -q", FAIL)
        s.edit("src/a.py")
        s.bash("pytest -q", FAIL.replace("0.12", "0.30"))
        [event] = plan(s).events
        e = RouteEscalation.parse(event.text)
        assert e is not None
        assert (e.input_tokens, e.output_tokens, e.cache_read_tokens) == (700, 40, 5)

    def test_one_event_per_task_even_with_several_signals(self) -> None:
        s = Script()
        rework(s)
        s.edit("src/a.py", "b", "c")
        s.bash("pytest -q", FAIL.replace("0.12", "0.40"))
        p = plan(s)
        assert len(p.events) == 1
        rework(s)
        assert len(plan(s).events) == 2
        assert len({e.id for e in plan(s).events}) == 2

    def test_the_events_are_stable_across_runs(self) -> None:
        s = Script()
        rework(s)
        assert plan(s).events == plan(s).events

    def test_the_escalation_codec_round_trips_and_refuses_other_text(self) -> None:
        e = RouteEscalation(
            "rework_cycle", "f:1", "implement", "escalate", 12, 1, 2, 3, 4
        )
        assert RouteEscalation.parse(e.text()) == e
        assert RouteEscalation.parse("research-1: flaky/large failed") is None

    def test_an_escalation_event_read_back_is_judged_by_unearned_escalation(
        self,
    ) -> None:
        # The router's own event, appended after a completed answer, is what
        # TER-DET-011 judges.
        s = Script()
        rework(s)
        s.say("The test still fails the same way.")
        [event] = plan(s).events
        s.events.append(replace(event, timestamp=s.events[-1].timestamp))
        s.say("It still fails the same way.")
        a = analyse(s)
        assert [f.detector for f in a.findings if f.detector == "unearned_escalation"]


@pytest.mark.req("TER-RTE-003")
class TestKeepTheProfile:
    def test_no_signal_keeps_the_role(self) -> None:
        s = Script()
        s.prompt("fix src/a.py")
        s.edit("src/a.py")
        s.bash("pytest -q", PASS)
        [d] = plan(s).decisions
        assert not d.escalated and d.role == d.final_role == "implement"
        assert "no detector signal" in d.reason and plan(s).events == ()

    def test_a_signal_from_a_detector_outside_escalate_on_keeps_the_role(self) -> None:
        s = Script()
        rework(s)
        p = plan(s, profile(escalate_on=frozenset({"regeneration"})))
        assert not p.decisions[0].escalated and p.events == ()

    def test_an_uncertain_signal_keeps_the_role(self) -> None:
        # Uncertain waste counts until verified, but never triggers an
        # intervention (ADR 0006): an unobserved re-read is uncertain.
        s = Script()
        s.prompt("explain src/a.py")
        s.read("src/a.py", None)
        s.read("src/a.py", None)
        s.say("ok")
        a = analyse(s)
        assert any(
            f.detector == "repeated_exploration" and f.uncertain for f in a.findings
        )
        p = plan(s, profile(escalate_on=frozenset({"repeated_exploration"})))
        assert not p.decisions[0].escalated and p.events == ()

    def test_a_signal_in_another_task_keeps_this_one(self) -> None:
        s = Script()
        rework(s)
        s.prompt("now explain src/b.py")
        s.read("src/b.py")
        p = plan(s)
        assert [d.escalated for d in p.decisions] == [True, False]

    def test_no_role_above_keeps_the_role_and_says_why(self) -> None:
        s = Script()
        rework(s)
        [d] = plan(s, profile(escalation={})).decisions
        assert not d.escalated and d.signal is not None
        assert "names no role above implement" in d.reason


# --- the command line --------------------------------------------------------


@pytest.mark.req("TER-RTE-005")
def test_cli_route_prints_classes_and_decisions() -> None:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["route", str(SAMPLE)], bootstrap.cli_services(), stdout=out, stderr=err
    )
    text = out.getvalue()
    assert code == 0, err.getvalue()
    assert "profile default" in text and "advisory" in text
    for label in ("complexity", "ambiguity", "risk", "scope", "validation", "decision"):
        assert label in text


@pytest.mark.req("TER-RTE-001")
def test_cli_route_json_and_profile_errors(tmp_path: Path) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["route", str(SAMPLE), "--json", "--profile", "local-code"],
        bootstrap.cli_services(),
        stdout=out,
        stderr=err,
    )
    assert code == 0
    data = json.loads(out.getvalue())
    assert data["schema"] == "ter.route/0.1" and data["profile"]["name"] == "local-code"
    assert set(data["rules"]) == {d.value for d in Dimension}
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["route", str(SAMPLE), "--profile", "nope"],
        bootstrap.cli_services(),
        stdout=out,
        stderr=err,
    )
    assert code == 2 and "Unknown routing profile 'nope'" in err.getvalue()
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    code = main(
        ["route", str(SAMPLE), "--profiles", str(tmp_path)],
        bootstrap.cli_services(),
        stdout=io.StringIO(),
        stderr=(err := io.StringIO()),
    )
    assert code == 2 and "schema" in err.getvalue()


@pytest.mark.req("TER-RTE-002")
def test_routing_with_an_in_memory_profile_set() -> None:
    profiles = InMemoryRoutingProfiles([profile()])
    s = Script()
    rework(s)

    class _Source:
        format_name = "script"

        def read(self, ref: object) -> object:
            from ter.domain import SessionTrace

            return SessionTrace("s", "script", tuple(s.events))

    routed = RouteSession(_Source(), RegexTokenizer(), profiles)("x")  # type: ignore[arg-type]
    assert routed.plan.profile.name == "test" and len(routed.plan.events) == 1
