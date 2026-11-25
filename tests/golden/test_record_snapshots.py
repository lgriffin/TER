"""Golden TER Records (``ter.record/0.1``, ADR 0007) for every corpus session.

Each session is explained with the offline, pinned TER 3 score and priced
at the shipped book's ``sonnet`` rates (the corpus names a demo model, see
``test_cost_snapshots.py``), and its Record is frozen. A diff is a behaviour
change: regenerate with ``TER_UPDATE_GOLDEN=1`` and review it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ter.adapters.driven.calibration import default_calibration
from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.embedders import HashingEmbedder
from ter.adapters.driven.ter3 import Ter3Scorer
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.application import ExplainSession
from ter.domain.lean.control import detector_fingerprint
from ter.domain.ter_record import build_record

from .conftest import CORPUS, assert_matches_snapshot
from .test_cost_snapshots import _book


def record(name: str) -> dict[str, Any]:
    """The Record of a corpus session, as JSON would carry it."""
    use_case = ExplainSession(
        ClaudeCodeJsonlSource(),
        RegexTokenizer(),
        Ter3Scorer(RegexTokenizer(), HashingEmbedder()),
        prices=_book(),
    )
    explained = use_case(CORPUS[name])
    trace = explained.trace
    doc = build_record(
        explained.a3,
        trace.events,
        source=trace.source_format,
        coverage=trace.coverage,
        event_schema=trace.schema_version,
        tokenizer="regex",
        detectors=detector_fingerprint(explained.analysis),
        calibration=default_calibration().calibration(),
    )
    loaded: dict[str, Any] = json.loads(json.dumps(doc))
    return loaded


@pytest.mark.req("TER-REC-001")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_record_matches_golden_snapshot(name: str) -> None:
    assert_matches_snapshot(f"record/{name}.record", record(name))
