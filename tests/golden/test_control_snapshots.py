"""Golden snapshots of control charts over the golden corpus (point 201).

The eight corpus sessions are the baseline: their measures, the limits
computed from them and the rendered chart page are frozen under
``snapshots/control/``. A diff means a measure, the limit arithmetic, a rule
or the renderer changed; recalibrated detectors move the measures too.
Regenerate with ``TER_UPDATE_GOLDEN=1`` and review the diff.
"""

from __future__ import annotations

import json
from functools import cache

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.embedders import HashingEmbedder
from ter.adapters.driven.ter3 import Ter3Scorer
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.adapters.driving.reports.control import render_control_html
from ter.application import ExplainSession
from ter.application.control import measure_sessions
from ter.domain.lean.control import (
    ControlLimits,
    ControlReport,
    MeasuresDocument,
    compute_limits,
    control_report,
)

from .conftest import CORPUS, assert_matches_text_snapshot

pytestmark = pytest.mark.usefixtures("pinned_models")


@cache
def _measures() -> MeasuresDocument:
    use_case = ExplainSession(
        ClaudeCodeJsonlSource(),
        RegexTokenizer(),
        Ter3Scorer(RegexTokenizer(), HashingEmbedder()),
    )
    measured = measure_sessions([CORPUS[n] for n in sorted(CORPUS)], use_case)
    assert not measured.failures
    return MeasuresDocument(measured.rows)


def _limits() -> ControlLimits:
    return compute_limits(_measures().rows, computed_on="2026-10-10")


def _report() -> ControlReport:
    return control_report(_measures().rows, _limits())


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@pytest.mark.req("TER-SPC-009")
def test_corpus_measures_match_golden_snapshot() -> None:
    assert_matches_text_snapshot(
        "control/corpus.measures.json", _json(_measures().as_dict())
    )


@pytest.mark.req("TER-SPC-001", "TER-SPC-002")
def test_corpus_limits_match_golden_snapshot() -> None:
    assert_matches_text_snapshot(
        "control/corpus.limits.json", _json(_limits().as_dict())
    )


@pytest.mark.req("TER-SPC-003", "TER-SPC-006")
def test_corpus_signals_match_golden_snapshot() -> None:
    assert_matches_text_snapshot(
        "control/corpus.report.json", _json(_report().as_dict())
    )


@pytest.mark.req("TER-SPC-010")
def test_control_page_matches_golden_snapshot() -> None:
    assert_matches_text_snapshot(
        "control/corpus.html", render_control_html(_report(), title="Golden corpus")
    )


@pytest.mark.req("TER-SPC-011")
def test_a3_placed_against_corpus_limits_matches_golden_snapshot() -> None:
    from ter.adapters.driving.reports.a3 import render_a3_html

    use_case = ExplainSession(
        ClaudeCodeJsonlSource(),
        RegexTokenizer(),
        Ter3Scorer(RegexTokenizer(), HashingEmbedder()),
    )
    a3 = use_case(CORPUS["rework_loop"]).a3.placed(_limits())
    assert a3.process_control is not None and a3.process_control.firing
    assert_matches_text_snapshot(
        "control/rework_loop.a3.json", _json(a3.as_dict()["process_control"])
    )
    assert_matches_text_snapshot("control/rework_loop.a3.html", render_a3_html(a3))
