"""Golden snapshots of the L2 explanation for every corpus session.

Freezes the findings, classifications, value stream, scorecard and evidence
graph (``<name>.lean.json``), the A3 view-model (``<name>.a3.json``) and the
rendered A3 page (``report/<name>.a3.html``). TER inside the A3 is the
offline, pinned TER 3 score, so it matches ``<name>.default.json``. A diff is
a behaviour change: regenerate with ``TER_UPDATE_GOLDEN=1`` and review it.
"""

from __future__ import annotations

import json

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.embedders import HashingEmbedder
from ter.adapters.driven.ter3 import Ter3Scorer
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.adapters.driving.reports import render_a3_html
from ter.application import ExplainedSession, ExplainSession
from ter.domain.lean import FindingKind

from .conftest import (
    CORPUS,
    SNAPSHOT_DIR,
    assert_matches_snapshot,
    assert_matches_text_snapshot,
)


def _explained(name: str) -> ExplainedSession:
    use_case = ExplainSession(
        ClaudeCodeJsonlSource(),
        RegexTokenizer(),
        Ter3Scorer(RegexTokenizer(), HashingEmbedder()),
    )
    return use_case(CORPUS[name])


@pytest.mark.req("TER-LEN-008")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_lean_analysis_matches_golden_snapshot(name: str) -> None:
    analysis = _explained(name).analysis
    assert_matches_snapshot(f"{name}.lean", json.loads(json.dumps(analysis.as_dict())))


@pytest.mark.req("TER-LEN-008", "TER-RPT-003")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_a3_view_model_matches_golden_snapshot(name: str) -> None:
    a3 = _explained(name).a3
    assert_matches_snapshot(f"{name}.a3", json.loads(json.dumps(a3.as_dict())))


@pytest.mark.req("TER-LEN-008", "TER-RPT-004")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_a3_html_matches_golden_snapshot(name: str) -> None:
    assert_matches_text_snapshot(
        f"report/{name}.a3.html", render_a3_html(_explained(name).a3)
    )


@pytest.mark.req("TER-ANL-012")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_a3_ter_is_the_frozen_ter3_score(name: str) -> None:
    ter = _explained(name).analysis.scorecard.ter
    frozen = json.loads(
        (SNAPSHOT_DIR / f"{name}.default.json").read_text(encoding="utf-8")
    )
    assert ter is not None
    assert ter.value == pytest.approx(frozen["aggregate_ter"], abs=1e-6)


@pytest.mark.req("TER-DET-001")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_finding_ids_are_unique(name: str) -> None:
    analysis = _explained(name).analysis
    ids = [f.id for f in analysis.findings]
    assert len(ids) == len(set(ids))
    for f in analysis.findings:
        assert analysis.finding(f.id) is f


@pytest.mark.req("TER-RPT-003")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_a3_pareto_and_costs_reconcile_with_the_scorecard(name: str) -> None:
    a3 = _explained(name).a3
    sc = a3.analysis.scorecard
    assert sum(bar.tokens for bar in a3.pareto) == (sc.waste_tokens if a3.pareto else 0)
    allocated = a3.analysis.allocated_waste_tokens()
    assert sum(allocated.values()) == pytest.approx(sc.waste_tokens, abs=1)
    # Uncertain waste counts until verified (ADR 0006), so every waste
    # finding's allocation adds up to the scorecard.
    waste = {f.id for f in a3.analysis.findings if f.kind is FindingKind.WASTE}
    waste_cost = sum(
        allocated.get(i, 0.0)
        for c in a3.countermeasures
        for i in c.addresses
        if i in waste
    )
    assert waste_cost == pytest.approx(sc.waste_tokens, abs=1)
