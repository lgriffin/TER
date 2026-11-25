"""The TER Record: a portable, content-free efficiency record of one session.

ADR 0007. One JSON object (``ter.record/0.1``) per session says what was
consumed, what it did, what was verified and how sure TER is, so a billing
or telemetry pipeline can join efficiency without reading the session. It
is built here, by a pure function of an A3 report (the analysis, its cost
and its verdict), the session's events and the judged samples behind the
detectors; it adds no measure of its own beyond the cost split
(TER-ANL-042) and the observed shape (TER-SHP-001) the A3 already carries.

**No content** (TER-REC-002). A finding appears as its detector and its
numbers: its title, subject, explanation and evidence ids stay in the A3,
because they name files and quote the session. The only free strings are
identifiers TER defines, model names the provider reported, dates and
digests. The session id is hashed. A price book named by a path is called
``custom``. The outcome keeps its verdict and counts, never a check's name.

**Uncertainty travels with the number** (TER-REC-003, TER-REC-004).
Tokens and cost a finding below 0.70 claims are ``uncertain``, never
``avoidable``, and every detector the record reports carries the judged
samples behind it.

**Recomputable** (TER-REC-005). ``integrity`` holds a digest of the event
stream and of the evidence graph; :func:`record_differences` lists every
figure two records disagree on, so a verifier can rebuild the record from
the session it names and report each difference.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .calibration import Calibration
from .costing import COST_CLASSES, SessionCost, session_date
from .events import Event, TokenUsage
from .lean.a3 import A3Report
from .lean.model import FindingKind
from .outcome import CheckStatus

__all__ = [
    "RECORD_SCHEMA",
    "build_record",
    "events_digest",
    "record_differences",
]

RECORD_SCHEMA = "ter.record/0.1"

#: How the cost was split among activity classes (ADR 0007, TER-ANL-042).
ALLOCATION = "whole-turn-by-generated-tokens"

#: A price book name kept as is: the shipped book (``ter:data/...``) or a
#: name with no path in it. Anything else may be a local path and is
#: reported as ``custom`` (TER-REC-002).
_SAFE_BOOK = re.compile(r"^(ter:[A-Za-z0-9_./-]+|[A-Za-z0-9_.-]+)$")


def _digest(payload: object) -> str:
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def events_digest(events: Iterable[Event]) -> str:
    """A digest of the event stream: what each event was, in order.

    It covers each event's id, kind, actor, text, tool call, usage and
    timestamp, so a record can be tied to the log it was built from. The
    digest reveals none of it.
    """
    return _digest(
        [
            [
                e.id,
                e.kind.value,
                e.actor.value,
                e.text,
                None
                if e.tool is None
                else [
                    e.tool.kind.value,
                    e.tool.native_name,
                    e.tool.call_id,
                    dict(e.tool.arguments),
                ],
                None
                if e.usage is None
                else [
                    e.usage.input_tokens,
                    e.usage.output_tokens,
                    e.usage.cache_read_tokens,
                    e.usage.cache_creation_tokens,
                    e.usage.model,
                ],
                None if e.timestamp is None else e.timestamp.isoformat(),
            ]
            for e in events
            if not e.kind.is_record
        ]
    )


def _ratio(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _tokens(turns: Sequence[TokenUsage]) -> dict[str, int]:
    return {
        "model_turns": len(turns),
        "input": sum(u.input_tokens for u in turns),
        "cache_read": sum(u.cache_read_tokens for u in turns),
        "cache_write": sum(u.cache_creation_tokens for u in turns),
        "output": sum(u.output_tokens for u in turns),
    }


def _usage(a3: A3Report) -> dict[str, object]:
    """Provider-reported usage, in all and by the model that served each turn
    (``by_model``), so a FOCUS exporter can join each model's share to that
    model's billing rows (ADR 0008). A turn whose source names no model is
    under ``model: null``."""
    turns = [s.usage for s in a3.analysis.steps if s.usage is not None]
    models = sorted({u.model for u in turns}, key=lambda m: (m is None, m or ""))
    return {
        **_tokens(turns),
        "by_model": [
            {"model": model, **_tokens([u for u in turns if u.model == model])}
            for model in models
        ],
    }


def _work(a3: A3Report) -> dict[str, object]:
    sc = a3.analysis.scorecard
    activity = dict(sc.activity_tokens)
    return {
        "generated_tokens": sc.generated_tokens,
        "context_tokens": sc.context_tokens,
        "activity_tokens": {k: int(activity.get(k, 0)) for k in COST_CLASSES},
        "waste_tokens": sc.waste_tokens,
        "uncertain_waste_tokens": sc.uncertain_waste_tokens,
        "waste_context_tokens": sc.waste_context_tokens,
        "flow_efficiency_tokens": _ratio(sc.flow_efficiency_tokens),
        "flow_efficiency_time": _ratio(sc.flow_efficiency_time),
        "ter": None if sc.ter is None else round(sc.ter.value, 4),
    }


def _findings(a3: A3Report, calibration: Calibration) -> list[dict[str, object]]:
    analysis = a3.analysis
    allocated = analysis.allocated_waste_tokens()
    by_detector: dict[str, list[Any]] = {}
    for f in analysis.findings:
        by_detector.setdefault(f.detector, []).append(f)
    out: list[dict[str, object]] = []
    for detector in sorted(by_detector):
        found = by_detector[detector]
        first = found[0]
        sure = [f for f in found if not f.uncertain]
        unsure = [f for f in found if f.uncertain]
        waste = first.kind is FindingKind.WASTE
        out.append(
            {
                "detector": detector,
                "waste": first.waste.value,
                "kind": first.kind.value,
                "findings": len(found),
                "uncertain_findings": len(unsure),
                "confidence_min": round(min(f.confidence for f in found), 4),
                "confidence_max": round(max(f.confidence for f in found), 4),
                # Uncertain findings never claim avoidable tokens (TER-REC-003).
                "avoidable_tokens": round(sum(allocated.get(f.id, 0.0) for f in sure))
                if waste
                else 0,
                "uncertain_tokens": round(sum(allocated.get(f.id, 0.0) for f in unsure))
                if waste
                else 0,
                "context_tokens": sum(f.context_tokens for f in found),
                "evidence_events": len({e for f in found for e in f.evidence}),
                "calibration": calibration.of(detector).as_dict(),
            }
        )
    return out


def _outcome(a3: A3Report) -> dict[str, object]:
    # No evidence, no verdict and no per-outcome figure (TER-REC-006).
    if a3.outcome is None:
        return {
            "status": "unknown",
            "required_checks": None,
            "passed_checks": None,
            "failed_checks": None,
            "generated_tokens_per_verified_outcome": None,
        }
    verdict = a3.outcome
    per = a3.tokens_per_verified_outcome
    return {
        "status": verdict.verdict.value,
        "required_checks": sum(1 for r in verdict.results if r.check.required),
        "passed_checks": verdict.count(CheckStatus.PASSED),
        "failed_checks": verdict.count(CheckStatus.FAILED)
        + verdict.count(CheckStatus.ERROR),
        "generated_tokens_per_verified_outcome": None if per is None else round(per, 1),
    }


def _book_name(name: str) -> str:
    return name if _SAFE_BOOK.match(name) else "custom"


def _cost(cost: SessionCost | None) -> dict[str, object] | None:
    if cost is None:
        return None
    return {
        "currency": "USD",
        "price_book": _book_name(cost.price_book),
        "priced_on": None if cost.priced_on is None else cost.priced_on.isoformat(),
        "usd": round(cost.usd, 6),
        "estimated": cost.estimated,
        "estimate_reasons": list(cost.estimate_reasons),
        "unpriced_turns": cost.unpriced_turns,
        "unpriced_models": list(cost.unpriced_models),
        "by_token_type": {k: round(v, 6) for k, v in cost.by_token_type},
        "by_activity_class": {k: round(v, 6) for k, v in cost.by_activity_class},
        "output_by_activity_class": {
            k: round(v, 6) for k, v in cost.output_by_activity_class
        },
        "allocation": ALLOCATION,
    }


def build_record(
    a3: A3Report,
    events: Sequence[Event],
    *,
    source: str,
    coverage: float,
    event_schema: str,
    tokenizer: str,
    detectors: str,
    calibration: Calibration,
) -> dict[str, object]:
    """The ``ter.record/0.1`` object of one analysed session (TER-REC-001).

    ``events`` are the session's events as read (for the digest), ``source``
    and ``coverage`` the source's format and record coverage, ``tokenizer``
    the tokenizer's name and ``detectors`` the detector fingerprint, so a
    verifier knows what to rebuild the record with.
    """
    analysis = a3.analysis
    steps = analysis.steps
    models = sorted({s.usage.model for s in steps if s.usage and s.usage.model})
    day = session_date(s.timestamp for s in steps)
    sc = analysis.scorecard
    return {
        "schema": RECORD_SCHEMA,
        "produced_by": {
            "implementation": "ter",
            "event_schema": event_schema,
            "tokenizer": tokenizer,
            "detectors": detectors,
            "ter_method": None if sc.ter is None else sc.ter.method,
        },
        "session": {
            "id": _digest(analysis.session_id or "")[:23],
            "source": source,
            "events": analysis.events,
            "coverage": round(coverage, 4),
            "usage_limits": list(a3.usage_limits),
        },
        "join": {
            "models": models,
            "charge_period": None if day is None else day.isoformat(),
            "principal": None,
            "credential": None,
            "conversation": None,
        },
        "usage": _usage(a3),
        "work": _work(a3),
        "findings": _findings(a3, calibration),
        "outcome": _outcome(a3),
        "shape": a3.shape.as_dict(),
        "cost": _cost(a3.cost),
        "integrity": {
            "events_digest": events_digest(events),
            "evidence_digest": _digest(analysis.graph.as_dict()),
        },
    }


def record_differences(
    claimed: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> list[tuple[str, object, object]]:
    """Every figure two records disagree on, as (path, claimed, rebuilt).

    Paths are dotted keys with list indexes (``findings[0].tokens``); a key
    only one side has is a difference with ``None`` on the other side.
    """
    out: list[tuple[str, object, object]] = []

    def walk(path: str, a: object, b: object) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                walk(f"{path}.{key}" if path else str(key), a.get(key), b.get(key))
        elif isinstance(a, list) and isinstance(b, list):
            for i in range(max(len(a), len(b))):
                walk(
                    f"{path}[{i}]",
                    a[i] if i < len(a) else None,
                    b[i] if i < len(b) else None,
                )
        elif a != b:
            out.append((path, a, b))

    walk("", claimed, rebuilt)
    return out
