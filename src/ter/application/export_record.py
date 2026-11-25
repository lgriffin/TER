"""Use cases: export a session's TER Record, and verify one (ADR 0007).

:func:`record_of` turns an explained session into its ``ter.record/0.1``
object (TER-REC-001). :func:`check_record` rebuilds the record from the
session a claimed record names and lists every figure that differs
(TER-REC-005): first whether the session is the one the record's events
digest names, then each differing figure by its path.

The record says what to rebuild it with: the tokenizer by name, and how
TER 3 was scored (:func:`ter_mode_of`). The price book and the judged
samples are reference data shared by every session, so a verifier uses
its own; a record built with other ones shows the differences.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..domain.calibration import Calibration
from ..domain.lean.control import detector_fingerprint
from ..domain.ter_record import build_record, record_differences
from .explain import ExplainedSession

__all__ = ["RecordCheck", "check_record", "record_of", "ter_mode_of"]


def record_of(
    explained: ExplainedSession, *, tokenizer: str, calibration: Calibration
) -> dict[str, object]:
    """The ``ter.record/0.1`` object of an explained session."""
    trace = explained.trace
    return build_record(
        explained.a3,
        trace.events,
        source=trace.source_format,
        coverage=trace.coverage,
        event_schema=trace.schema_version,
        tokenizer=tokenizer,
        detectors=detector_fingerprint(explained.analysis),
        calibration=calibration,
    )


@dataclass(frozen=True)
class RecordCheck:
    """A claimed record against the one rebuilt from its session."""

    #: Whether the session's events digest is the one the record names.
    same_session: bool
    #: Every differing figure as (path, claimed, rebuilt).
    differences: tuple[tuple[str, object, object], ...]

    @property
    def ok(self) -> bool:
        return self.same_session and not self.differences


def check_record(
    claimed: Mapping[str, Any],
    explained: ExplainedSession,
    *,
    tokenizer: str,
    calibration: Calibration,
) -> RecordCheck:
    """Rebuild the record from ``explained`` and compare it with ``claimed``."""
    rebuilt = json.loads(
        json.dumps(record_of(explained, tokenizer=tokenizer, calibration=calibration))
    )
    integrity = claimed.get("integrity")
    named = integrity.get("events_digest") if isinstance(integrity, Mapping) else None
    return RecordCheck(
        same_session=named == rebuilt["integrity"]["events_digest"],
        differences=tuple(record_differences(claimed, rebuilt)),
    )


def ter_mode_of(record: Mapping[str, Any]) -> str:
    """How TER 3 was scored for ``record``: ``off``, ``offline`` or ``model``."""
    produced = record.get("produced_by")
    method = produced.get("ter_method") if isinstance(produced, Mapping) else None
    if not isinstance(method, str):
        return "off"
    return "offline" if "offline" in method else "model"
