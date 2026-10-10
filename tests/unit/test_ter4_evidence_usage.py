"""L3 evidence usage (TER-EVD-008), the grounded evidence graph (TER-GRF-001),
outcome value (TER-LEN-009) and exploration drift (TER-ITN-006), on the
synthetic shop repository (``ter4_shop_repo``) and synthetic sessions:
positive, negative and boundary cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ter4_lean_builder import FAIL, PASS, Script
from ter4_shop_repo import SHOP, at, shop_repo
from test_ter4_grounded_cli import run, session

from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.application.ground import ground_session
from ter.bootstrap.capabilities import repository_evidence
from ter.domain.lean import LeanAnalysis, build_countermeasures, explain
from ter.domain.lean.analysis import LeanAnalyser
from ter.domain.lean.countermeasures import follow_ups
from ter.domain.lean.drift import EVIDENCE_DETECTORS, package_of_path
from ter.domain.lean.evidence_graph import (
    EDGE_RULES,
    GraphEdgeType,
    NodeRole,
    build_grounded_graph,
)
from ter.domain.lean.model import UNCERTAIN_BELOW, Finding
from ter.domain.lean.usage import (
    USAGE_RULES,
    FileNames,
    FileRole,
    ReadUsage,
    UsageStatus,
    UseKind,
    file_role,
)
from ter.domain.lean.value import VALUE_RULES, ValueClass, ValueJudgement

PRICING = "src/app/domain/pricing.py"
MODEL = "src/app/domain/model.py"
CHECKOUT = "src/app/service/checkout.py"
WEB = "src/app/adapters/web.py"
SUMMARY = "src/app/reports/summary.py"
SEED = "scripts/seed_data.py"

#: The shop plus a script in another top-level package that nothing imports.
SHOP_WITH_SCRIPTS = {
    **SHOP,
    SEED: "def load_seed_rows():\n    return [1, 2, 3]\n",
    ".github/workflows/ci.yml": "on: push\n",
}


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    return shop_repo(tmp_path / "shop", SHOP_WITH_SCRIPTS)


def analyse(script: Script, root: Path) -> LeanAnalysis:
    g = ground_session(script.events, repository_evidence(root, "python-ast"), None)
    return explain(script.events, RegexTokenizer(), repository=g)


def read_of(analysis: LeanAnalysis, path: str, nth: int = 0) -> ReadUsage:
    assert analysis.usage is not None
    return [r for r in analysis.usage.reads if r.path == path][nth]


def judged(analysis: LeanAnalysis, event_id: str) -> ValueJudgement:
    assert analysis.value is not None
    return next(j for j in analysis.value.judgements if j.event_id == event_id)


def drift(analysis: LeanAnalysis) -> list[Finding]:
    return [f for f in analysis.findings if f.detector == "exploration_drift"]


def kinds(read: ReadUsage) -> list[UseKind]:
    return [u.kind for u in read.uses]


# --- TER-EVD-008: evidence usage --------------------------------------------


@pytest.mark.req("TER-EVD-008")
class TestEvidenceUsage:
    def test_a_read_file_that_is_edited_later_was_used(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        read, _ = s.read(at(PRICING), SHOP[PRICING])
        edit, _ = s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        r = read_of(analyse(s, shop), PRICING)
        assert r.event_id == read.id and r.status is UsageStatus.USED
        assert r.material and kinds(r) == [UseKind.EDITED]
        assert r.uses[0].event_id == edit.id
        assert r.evidence[0] == read.id and edit.id in r.evidence

    def test_a_dependency_of_a_later_edit_was_used(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.read(at(MODEL), SHOP[MODEL])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        r = read_of(analyse(s, shop), MODEL)
        assert UseKind.IMPORTED_BY_EDIT in kinds(r) and r.material

    def test_a_symbol_named_in_a_later_edit_uses_the_read(self, shop: Path) -> None:
        s = Script()
        s.prompt("Show the monthly summary on the web page")
        s.read(at(SUMMARY), SHOP[SUMMARY])
        s.edit(at(WEB), "return checkout(request)", "return monthly_summary(request)")
        s.say("Done.")
        assert kinds(read_of(analyse(s, shop), SUMMARY)) == [UseKind.NAMED_IN_EDIT]

    def test_a_check_naming_a_test_of_the_read_file_tested_it(self, shop: Path) -> None:
        s = Script()
        s.prompt("Check pricing.py")
        s.read(at(PRICING), SHOP[PRICING])
        s.read(at(MODEL), SHOP[MODEL])
        s.bash("pytest tests/test_pricing.py", PASS)
        s.say("All good.")
        a = analyse(s, shop)
        assert kinds(read_of(a, PRICING)) == [UseKind.TESTED]
        # Boundary: the test imports pricing, not model: links are direct.
        assert read_of(a, MODEL).status is UsageStatus.UNUSED

    def test_a_command_acting_on_the_file_used_it(self, shop: Path) -> None:
        s = Script()
        s.prompt("Back up the summary module")
        s.read(at(SUMMARY), SHOP[SUMMARY])
        s.bash(f"cp {SUMMARY} /tmp/summary_backup.py")
        s.say("Copied.")
        assert kinds(read_of(analyse(s, shop), SUMMARY)) == [UseKind.NAMED_IN_COMMAND]

    def test_reading_again_is_not_a_use(self, shop: Path) -> None:
        s = Script()
        s.prompt("Look at the shop")
        s.read(at(SUMMARY), SHOP[SUMMARY])
        s.bash(f"cat {SUMMARY}", SHOP[SUMMARY])
        s.say("Seen.")
        a = analyse(s, shop)
        first, again = read_of(a, SUMMARY), read_of(a, SUMMARY, 1)
        assert first.status is UsageStatus.UNUSED and again.via == "shell"

    def test_a_decision_naming_the_file_is_a_use_but_not_material(
        self, shop: Path
    ) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.read(at(SUMMARY), SHOP[SUMMARY])
        s.think("monthly_summary only counts rows, so it is not involved.")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        r = read_of(analyse(s, shop), SUMMARY)
        assert kinds(r) == [UseKind.NAMED_IN_DECISION]
        assert r.status is UsageStatus.USED and not r.material

    def test_unused_reads_are_costed_and_pending_reads_wait(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.read(at(SUMMARY), SHOP[SUMMARY])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        s.read(at(CHECKOUT), SHOP[CHECKOUT])
        a = analyse(s, shop)
        unused = read_of(a, SUMMARY)
        assert unused.status is UsageStatus.UNUSED and unused.context_tokens > 0
        assert read_of(a, CHECKOUT).status is UsageStatus.PENDING
        assert a.usage is not None
        assert a.usage.unused_tokens == unused.context_tokens
        summary = a.usage.as_dict()["summary"]
        assert isinstance(summary, dict)
        assert summary["unused"] == 1 and summary["pending"] == 1
        # Files explored against files changed (point 67).
        assert a.usage.explored == (SUMMARY, CHECKOUT)
        assert a.usage.changed == (PRICING,)
        assert summary["explored_and_changed"] == 0

    def test_shell_reads_of_several_files_are_one_read_each(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.bash(f"sed -n 1,5p {PRICING}; head {at(MODEL)}", SHOP[PRICING])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        a = analyse(s, shop)
        assert {r.path for r in a.usage.reads} == {PRICING, MODEL}  # type: ignore[union-attr]
        assert {r.via for r in a.usage.reads} == {"shell"}  # type: ignore[union-attr]

    def test_reads_outside_the_repository_are_not_judged(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.read("/tmp/notes.txt", "x")
        s.read(at(PRICING), SHOP[PRICING])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        a = analyse(s, shop)
        assert [r.path for r in a.usage.reads] == [PRICING]  # type: ignore[union-attr]

    def test_names_must_name_one_file(self, shop: Path) -> None:
        s = Script()
        s.prompt("x")
        g = ground_session(s.events, repository_evidence(shop, "python-ast"), None)
        names = FileNames(g)
        assert "__init__.py" not in names.path_of
        assert names.path_of["pricing.py"] == PRICING
        assert names.path_of["app.domain.pricing"] == PRICING
        assert names.path_of["price_with_tax"] == PRICING
        assert names.named({"pricing.py", "__init__.py", "checkout"}) == {PRICING}

    def test_every_use_kind_has_a_published_rule(self) -> None:
        assert set(USAGE_RULES) == set(UseKind)
        assert [k for k in UseKind if not k.material] == [UseKind.NAMED_IN_DECISION]

    def test_file_roles(self) -> None:
        assert file_role("tests/test_a.py") is FileRole.TEST
        assert file_role("web/src/a.test.ts") is FileRole.TEST
        assert file_role(".github/workflows/ci.yml") is FileRole.CI
        assert file_role("docs/x.md") is FileRole.DOC
        assert file_role("README.md") is FileRole.DOC
        assert file_role("pyproject.toml") is FileRole.CONFIG
        assert file_role("Makefile") is FileRole.CONFIG
        assert file_role("src/app/core.py") is FileRole.SOURCE

    def test_without_a_repository_nothing_is_added(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        s.read(at(PRICING), SHOP[PRICING])
        s.say("Done.")
        a = explain(s.events, RegexTokenizer())
        assert a.usage is None and a.value is None and a.grounded_graph is None
        out = a.as_dict()
        assert not {"evidence_usage", "outcome_value", "grounded_evidence_graph"} & set(
            out
        )


# --- TER-GRF-001: the grounded evidence graph ----------------------------------


def _graph_script() -> Script:
    s = Script()
    s.prompt("Fix the rounding in pricing.py")
    s.think("Read pricing.py first.")
    s.read(at(PRICING), SHOP[PRICING])
    s.think("price_with_tax multiplies by 1.2; use an exact fraction.")
    s.edit(at(PRICING), "* 1.2", "* 12 / 10")
    s.bash("pytest tests/test_pricing.py", FAIL)
    s.edit(at(PRICING), "* 12 / 10", "* 6 / 5")
    s.bash("pytest tests/test_pricing.py", PASS)
    s.say("Fixed.")
    return s


@pytest.mark.req("TER-GRF-001")
class TestEvidenceGraph:
    def test_nodes_have_roles_and_every_edge_type_has_a_rule(self, shop: Path) -> None:
        s = _graph_script()
        g = analyse(s, shop).grounded_graph
        assert g is not None
        assert set(EDGE_RULES) == set(GraphEdgeType)
        roles = [n.role for n in g.nodes]
        assert roles == [
            NodeRole.INTENT,
            NodeRole.DECISION,
            NodeRole.ACTION,
            NodeRole.OBSERVATION,
            NodeRole.DECISION,
            NodeRole.CHANGE,
            NodeRole.OBSERVATION,
            NodeRole.VALIDATION,
            NodeRole.OBSERVATION,
            NodeRole.CHANGE,
            NodeRole.OBSERVATION,
            NodeRole.VALIDATION,
            NodeRole.OBSERVATION,
            NodeRole.RESPONSE,
        ]

    def test_edges_link_intent_observation_decision_change_and_validation(
        self, shop: Path
    ) -> None:
        s = _graph_script()
        e = s.events
        g = analyse(s, shop).grounded_graph
        assert g is not None
        has = {(x.source, x.target, x.type) for x in g.edges}
        prompt, think1, read, result, think2, edit1 = (x.id for x in e[:6])
        run1, fail, edit2, run2 = e[7].id, e[8].id, e[9].id, e[11].id
        assert (result, read, GraphEdgeType.OBSERVES) in has
        assert (edit1, prompt, GraphEdgeType.PURSUES) in has
        assert (think2, result, GraphEdgeType.BASED_ON) in has
        assert (edit1, think2, GraphEdgeType.DECIDED_BY) in has
        assert (edit1, result, GraphEdgeType.USES) in has  # repository evidence
        assert (run1, edit1, GraphEdgeType.VALIDATES) in has
        assert (edit2, fail, GraphEdgeType.CORRECTS) in has
        assert (run2, edit2, GraphEdgeType.VALIDATES) in has
        assert (think1, prompt, GraphEdgeType.PURSUES) in has

    def test_every_decision_rests_on_evidence(self, shop: Path) -> None:
        g = analyse(_graph_script(), shop).grounded_graph
        assert g is not None
        for node in g.nodes:
            if node.role is NodeRole.DECISION:
                assert g.out(node.event_id), node

    def test_following_edges_back_reconstructs_the_change(self, shop: Path) -> None:
        s = _graph_script()
        g = analyse(s, shop).grounded_graph
        assert g is not None
        chain = [n.event_id for n in g.reconstruct(s.events[9].id)]
        assert chain[0] == s.events[0].id and chain[-1] == s.events[9].id
        # The read, the decision and the failure it corrects are all in it.
        assert {s.events[3].id, s.events[4].id, s.events[8].id} <= set(chain)

    def test_the_graph_is_deterministic_and_batch_equals_incremental(
        self, shop: Path
    ) -> None:
        s = _graph_script()
        g = ground_session(s.events, repository_evidence(shop, "python-ast"), None)
        batch = explain(s.events, RegexTokenizer(), repository=g)
        live = LeanAnalyser()
        counter = RegexTokenizer()
        for event in s.events:
            live.add(event, counter.count(event.text))
            live.add(event, counter.count(event.text))  # redelivered: ignored
        incremental = live.analysis(repository=g)
        assert batch.grounded_graph is not None
        assert batch.grounded_graph.as_dict() == incremental.grounded_graph.as_dict()  # type: ignore[union-attr]
        assert batch.usage == incremental.usage and batch.value == incremental.value
        again = explain(s.events, RegexTokenizer(), repository=g)
        assert json.dumps(again.as_dict()) == json.dumps(batch.as_dict())

    def test_without_usage_the_graph_has_no_uses_edges(self, shop: Path) -> None:
        a = analyse(_graph_script(), shop)
        bare = build_grounded_graph(a.session_id, a.steps)
        assert not bare.edges_of(GraphEdgeType.USES)
        assert bare.edges_of(GraphEdgeType.DECIDED_BY)


# --- TER-LEN-009: outcome value -------------------------------------------------


@pytest.mark.req("TER-LEN-009")
class TestOutcomeValue:
    def test_reads_on_and_off_the_surface(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        on, _ = s.read(at(PRICING), SHOP[PRICING])
        neighbour, _ = s.read(at(MODEL), SHOP[MODEL])
        off_used, _ = s.read(at(SUMMARY), SHOP[SUMMARY])
        off_unused, _ = s.read(at(WEB), SHOP[WEB])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10  # see monthly_summary")
        s.say("Done.")
        a = analyse(s, shop)
        assert judged(a, on.id).rule == "explore.surface_used"
        assert judged(a, neighbour.id).value is ValueClass.REQUIRED
        assert judged(a, off_used.id).rule == "explore.off_surface_used"
        j = judged(a, off_unused.id)
        assert j.value is ValueClass.NO_VALUE and j.uncertain
        assert j.confidence < UNCERTAIN_BELOW

    def test_the_same_read_is_valued_against_the_current_intent(
        self, shop: Path
    ) -> None:
        def session(prompt: str, edited: str) -> tuple[Script, str]:
            s = Script()
            s.prompt(prompt)
            read, _ = s.read(at(SUMMARY), SHOP[SUMMARY])
            s.edit(at(edited), "a", "b")
            s.say("Done.")
            return s, read.id

        s1, r1 = session("Fix the rounding in pricing.py", PRICING)
        s2, r2 = session(
            "Make monthly_summary count unique rows in summary.py", SUMMARY
        )
        assert judged(analyse(s1, shop), r1).value is ValueClass.NO_VALUE
        assert judged(analyse(s2, shop), r2).value is ValueClass.REQUIRED

    def test_reasoning(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        surface = s.think("price_with_tax multiplies by 1.2.")
        off = s.think("monthly_summary looks unrelated.")
        prose = s.think("Let me fix it.")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        a = analyse(s, shop)
        assert judged(a, surface.id).rule == "reasoning.surface_then_change"
        assert judged(a, off.id).rule == "reasoning.off_surface_unused"
        assert judged(a, prose.id).value is ValueClass.UNJUDGED

    def test_validation(self, shop: Path) -> None:
        s = Script()
        s.prompt("Fix the rounding in pricing.py")
        baseline, _ = s.bash("pytest tests/test_pricing.py", PASS)
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        covers, _ = s.bash("pytest tests/test_pricing.py", PASS)
        again, _ = s.bash("pytest tests/test_pricing.py", PASS)
        s.edit(at(PRICING), "* 12 / 10", "* 6 / 5")
        suite, _ = s.bash("ruff check", "All checks passed!")
        s.edit(at(PRICING), "* 6 / 5", "* 12 / 10")
        other, _ = s.bash("pytest tests/test_summary.py", PASS)
        s.say("Done.")
        a = analyse(s, shop)
        assert judged(a, baseline.id).rule == "validation.baseline"
        assert judged(a, covers.id).rule == "validation.covers_surface"
        assert judged(a, again.id).rule == "validation.no_new_change"
        assert judged(a, suite.id).rule == "validation.suite_after_change"
        assert judged(a, other.id).rule == "validation.off_surface"

    def test_a_task_that_changes_nothing_has_no_surface(self, shop: Path) -> None:
        s = Script()
        s.prompt("What does pricing.py do?")
        read, _ = s.read(at(PRICING), SHOP[PRICING])
        s.say("price_with_tax adds tax to the order_total.")
        j = judged(analyse(s, shop), read.id)
        assert j.rule == "explore.no_change_used" and j.value is ValueClass.SUPPORTING

    def test_every_judgement_cites_evidence_and_a_published_rule(
        self, shop: Path
    ) -> None:
        a = analyse(_graph_script(), shop)
        assert a.value is not None and a.value.judgements
        ids = {st.event_id for st in a.steps}
        for j in a.value.judgements:
            assert j.rule in VALUE_RULES and j.event_id in j.evidence
            assert set(j.evidence) <= ids
            assert j.uncertain == (j.confidence < UNCERTAIN_BELOW)
        out = a.value.as_dict()
        assert set(out["classes"]) == {v.value for v in ValueClass}  # type: ignore[arg-type]
        for rule in VALUE_RULES.values():
            if rule.value is ValueClass.NO_VALUE:
                assert rule.confidence < UNCERTAIN_BELOW


# --- TER-ITN-006: exploration drift ------------------------------------------


def _fix_pricing(s: Script, prompt: str = "Fix the rounding in pricing.py") -> None:
    s.prompt(prompt)
    s.read(at(PRICING), SHOP[PRICING])


@pytest.mark.req("TER-ITN-006")
class TestExplorationDrift:
    def test_a_read_in_another_package_nothing_used_is_confident_drift(
        self, shop: Path
    ) -> None:
        s = Script()
        _fix_pricing(s)
        read, result = s.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        [f] = drift(analyse(s, shop))
        assert f.confidence == 0.75 and not f.uncertain
        assert f.subject == SEED and read.id in f.evidence
        assert s.events[0].id in f.evidence  # the prompt
        assert result is not None and f.waste_events == (read.id, result.id)

    def test_a_read_in_the_same_package_is_uncertain(self, shop: Path) -> None:
        s = Script()
        _fix_pricing(s)
        s.read(at(SUMMARY), SHOP[SUMMARY])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        [f] = drift(analyse(s, shop))
        assert f.confidence == 0.6 and f.uncertain

    def test_files_the_change_depends_on_are_never_drift(self, shop: Path) -> None:
        s = Script()
        _fix_pricing(s)
        s.read(at(MODEL), SHOP[MODEL])  # import neighbour
        s.read(at(WEB), SHOP[WEB])  # one import link beyond the surface
        s.read(at("tests/test_summary.py"), SHOP["tests/test_summary.py"])  # test
        s.read(at("README.md"), SHOP["README.md"])  # doc
        s.read(at("pyproject.toml"), SHOP["pyproject.toml"])  # config
        s.read(at(".github/workflows/ci.yml"), "on: push\n")  # CI
        s.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10  # like load_seed_rows")  # used
        s.say("Done.")
        assert drift(analyse(s, shop)) == []

    def test_a_recorded_intent_change_is_not_drift(self, shop: Path) -> None:
        s = Script()
        _fix_pricing(s)
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        s.prompt("Now change load_seed_rows in seed_data.py to return four rows")
        s.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        s.edit(at(SEED), "[1, 2, 3]", "[1, 2, 3, 4]")
        s.say("Done.")
        assert drift(analyse(s, shop)) == []

    def test_a_decision_naming_it_later_keeps_it_uncertain(self, shop: Path) -> None:
        s = Script()
        _fix_pricing(s)
        s.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        s.think("load_seed_rows is not relevant here.")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        read, thought = sorted(drift(analyse(s, shop)), key=lambda f: f.title)
        assert read.title.startswith("Exploration") and read.confidence == 0.6
        # The reasoning itself names only the drifted file: uncertain drift.
        assert thought.title.startswith("Reasoning") and thought.confidence == 0.5

    def test_surfaces_the_prompt_did_not_name_cap_confidence(self, shop: Path) -> None:
        first_edit = Script()
        first_edit.prompt("Tidy things up please")
        first_edit.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        first_edit.edit(at(PRICING), "* 1.2", "* 12 / 10")
        first_edit.say("Done.")
        [f] = drift(analyse(first_edit, shop))
        assert f.confidence == 0.5

        inherited = Script()
        _fix_pricing(inherited)
        inherited.edit(at(PRICING), "* 1.2", "* 12 / 10")
        inherited.say("Done.")
        inherited.prompt("Also make the rounding exact for the tax part too")
        inherited.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        inherited.edit(at(PRICING), "* 12 / 10", "* 6 / 5")
        inherited.say("Done.")
        found = drift(analyse(inherited, shop))
        assert [f.confidence for f in found] == [0.6]

    def test_reasoning_drift_is_always_uncertain(self, shop: Path) -> None:
        s = Script()
        _fix_pricing(s)
        thought = s.think("Maybe seed_data.py needs a look too.")
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        [f] = drift(analyse(s, shop))
        assert f.confidence == 0.5 and f.waste_events == (thought.id,)

    def test_no_finding_before_a_response_or_without_a_change(self, shop: Path) -> None:
        live = Script()
        _fix_pricing(live)
        live.edit(at(PRICING), "* 1.2", "* 12 / 10")
        live.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        assert drift(analyse(live, shop)) == []
        question = Script()
        _fix_pricing(question, "What does pricing.py do?")
        question.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        question.say("It adds tax.")
        assert drift(analyse(question, shop)) == []

    def test_the_detector_is_grounded_only_and_has_a_countermeasure(
        self, shop: Path
    ) -> None:
        s = Script()
        _fix_pricing(s)
        s.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        s.edit(at(PRICING), "* 1.2", "* 12 / 10")
        s.say("Done.")
        l2 = explain(s.events, RegexTokenizer())
        assert "exploration_drift" not in {d[0] for d in l2.detectors}
        a = analyse(s, shop)
        assert "exploration_drift" in {d[0] for d in a.detectors}
        [cm] = [
            c
            for c in build_countermeasures(a.findings, a.steps)
            if c.detector == "exploration_drift"
        ]
        assert cm.actions
        ups = follow_ups(a.findings, flow_efficiency=None, avoidable_share=0.0)
        assert any(u.how == "findings[detector=exploration_drift]" for u in ups)
        assert EVIDENCE_DETECTORS[0].confidence_rule

    def test_top_level_packages(self) -> None:
        assert package_of_path("src/app/x.py") == "src/app"
        assert package_of_path("scripts/x.py") == "scripts"
        assert package_of_path("packages/web/src/a.ts") == "packages/web"
        assert package_of_path("README.md") == ""


# --- end to end ---------------------------------------------------------------


@pytest.mark.req("TER-EVD-008")
@pytest.mark.req("TER-GRF-001")
@pytest.mark.req("TER-LEN-009")
def test_explain_and_a3_json_carry_usage_value_and_graph(tmp_path: Path) -> None:
    repo = shop_repo(tmp_path / "shop")
    path = str(session(tmp_path))
    code, out, err = run(["explain", path, "--repo", str(repo), "--json"])
    assert code == 0, err
    analysis = json.loads(out)
    [read] = analysis["evidence_usage"]["reads"]
    assert read["path"] == PRICING and read["status"] == "used"
    assert analysis["outcome_value"]["judgements"]
    assert analysis["grounded_evidence_graph"]["schema"] == "ter.evidence-graph/1"
    code, out, _ = run(["explain", path, "--repo", str(repo)])
    assert "evidence usage   1 read(s): 1 used" in out
    assert "outcome value    " in out
    code, out, _ = run(["a3", path, "--repo", str(repo), "--ter", "off", "--json"])
    a3 = json.loads(out)["analysis"]
    assert {"evidence_usage", "outcome_value", "evidence_graph"} <= set(a3)
    code, out, _ = run(["a3", path, "--ter", "off", "--json"])
    assert "evidence_usage" not in json.loads(out)["analysis"]


@pytest.mark.req("TER-RPT-007")
def test_a_grounded_a3_page_shows_reads_used_and_files_explored_against_changed(
    shop: Path,
) -> None:
    from ter.adapters.driving.reports import render_a3_html
    from ter.domain.lean import build_a3

    s = Script()
    s.prompt("Fix the rounding in pricing.py")
    s.read(at(PRICING), SHOP[PRICING])
    s.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
    s.edit(at(PRICING), "* 12 / 10")
    s.bash("pytest -q tests/test_pricing.py", PASS)
    s.say("Fixed.")
    analysis = analyse(s, shop)
    assert analysis.usage is not None
    page = render_a3_html(build_a3(analysis, ["Fix the rounding"]))
    section = page[page.index('id="s-repository"') : page.index('id="s-5"')]
    assert "Reads later used" in section and "50%" in section
    assert "Explored → changed" in section and "1 / 2" in section
    assert (
        '<span class="mark yes" aria-label="changed after it was read">✓</span>'
        f"<code>{PRICING}</code>" in section
    )
    # The unused read is listed with its context tokens and its read event.
    unused = read_of(analysis, SEED)
    assert unused.status is UsageStatus.UNUSED
    assert f"<code>{SEED}</code>" in section and unused.event_id in section
    assert f'<td class="num">{unused.context_tokens:,}</td>' in section
    assert "Outcome value" in section
    assert '<li class="level">L3 Grounded</li>' in page
    assert 'href="#s-repository"' in page


@pytest.mark.req("TER-RPT-007")
def test_an_ungrounded_a3_page_has_no_repository_section(shop: Path) -> None:
    from ter.adapters.driving.reports import render_a3_html
    from ter.domain.lean import build_a3

    s = Script()
    s.prompt("Fix the rounding in pricing.py")
    s.read(at(PRICING), SHOP[PRICING])
    s.say("Fixed.")
    page = render_a3_html(build_a3(explain(s.events, RegexTokenizer()), ["x"]))
    assert "s-repository" not in page
    assert '<li class="level">L2 Explained</li>' in page


@pytest.mark.req("TER-RPT-007")
def test_the_changed_list_orders_reads_against_edits_and_skips_failed_edits(
    shop: Path,
) -> None:
    from ter.adapters.driving.reports import render_a3_html
    from ter.domain.lean import build_a3

    failed = "<tool_use_error>String to replace not found in file.</tool_use_error>"
    s = Script()
    s.prompt("Fix the rounding in pricing.py")
    s.read(at(PRICING), SHOP[PRICING])
    s.edit(at(PRICING), "* 1.2", "* 12 / 10")  # read first
    s.edit(at(CHECKOUT), "a", "b")  # edited, then read
    s.read(at(CHECKOUT), SHOP[CHECKOUT])
    s.write(at("src/app/new.py"), "X = 1\n")  # created
    s.read(at(WEB), SHOP[WEB])
    s.edit(at(WEB), "a", "b", output=failed)  # failed: not a change
    s.bash("pytest -q tests/test_pricing.py", PASS)
    s.say("Fixed.")
    page = render_a3_html(build_a3(analyse(s, shop), ["Fix the rounding"]))
    section = page[page.index('id="s-repository"') : page.index('id="s-5"')]
    changed = section[section.index("Files changed") :]
    changed = changed[: changed.index("</ul>")]
    assert (
        f'aria-label="read before its first edit">✓</span><code>{PRICING}<' in changed
    )
    assert (
        f'aria-label="edited before it was read">!</span><code>{CHECKOUT}<' in changed
    )
    assert (
        'aria-label="created by the session">+</span><code>src/app/new.py<' in changed
    )
    assert WEB not in changed
    # Explored then changed counts only reads followed by a successful edit.
    assert "1 / 3" in section


@pytest.mark.req("TER-RPT-007")
def test_every_unused_read_is_listed(shop: Path) -> None:
    from ter.adapters.driving.reports import render_a3_html
    from ter.domain.lean import build_a3

    s = Script()
    s.prompt("Fix the rounding in pricing.py")
    for _ in range(5):
        s.read(at(SEED), SHOP_WITH_SCRIPTS[SEED])
        s.read(at(WEB), SHOP[WEB])
    s.read(at(PRICING), SHOP[PRICING])
    s.edit(at(PRICING), "* 1.2", "* 12 / 10")
    s.say("Fixed.")
    analysis = analyse(s, shop)
    assert analysis.usage is not None
    unused = [r for r in analysis.usage.reads if r.status is UsageStatus.UNUSED]
    assert len(unused) > 8
    page = render_a3_html(build_a3(analysis, ["Fix"]))
    assert f"Show the other {len(unused) - 8} unused read(s)" in page
    for r in unused:
        assert r.event_id in page
