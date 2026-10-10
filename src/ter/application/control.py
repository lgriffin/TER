"""Control charts over many sessions (point 201).

Measuring runs the same explanation ``ter a3`` runs on every session and
keeps only its control measures, so the slow step can run where the private
sessions are and its content-free output travels. Charting reads a limits
document through the :class:`~ter.ports.driven.ControlLimitsSource` port.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..domain.events import SessionTrace
from ..domain.lean.control import (
    ControlLimits,
    ControlReport,
    LimitMethod,
    SessionMeasures,
    compute_limits,
    control_report,
    measure_session,
)
from ..ports.driven import ControlLimitsSource
from .explain import ExplainedSession

__all__ = [
    "ChartSessions",
    "MeasuredSessions",
    "baseline_limits",
    "measure_sessions",
    "session_start",
]


def session_start(trace: SessionTrace) -> datetime | None:
    """When the session started: its first timestamped event."""
    return next((e.timestamp for e in trace.events if e.timestamp is not None), None)


@dataclass(frozen=True)
class MeasuredSessions:
    rows: tuple[SessionMeasures, ...]
    #: ``(session ref, error)`` for every session that could not be explained.
    failures: tuple[tuple[str, str], ...]


def measure_sessions(
    refs: Iterable[Path], explain: Callable[[Path], ExplainedSession]
) -> MeasuredSessions:
    """Explain each session and keep its control measures. A session that
    cannot be read is recorded as a failure; the others are still measured."""
    rows: list[SessionMeasures] = []
    failures: list[tuple[str, str]] = []
    for ref in refs:
        try:
            explained = explain(ref)
        except Exception as exc:  # noqa: BLE001 - one bad session never stops a corpus
            failures.append((str(ref), f"{type(exc).__name__}: {exc}"))
            continue
        trace = explained.trace
        rows.append(
            measure_session(
                explained.analysis,
                session_start(trace),
                session_id=trace.session_id or ref.stem,
            )
        )
    return MeasuredSessions(tuple(rows), tuple(failures))


def baseline_limits(
    rows: Iterable[SessionMeasures],
    method: LimitMethod = LimitMethod.AVERAGE_MOVING_RANGE,
    keep: ControlLimits | None = None,
    computed_on: str | None = None,
) -> ControlLimits:
    """Natural limits from baseline sessions, keeping earlier tuning."""
    return compute_limits(tuple(rows), method, keep=keep, computed_on=computed_on)


class ChartSessions:
    """Chart measured sessions against a limits document."""

    def __init__(self, source: ControlLimitsSource) -> None:
        self._source = source

    def limits(self, ref: str | Path) -> ControlLimits:
        return self._source.limits(ref)

    def __call__(
        self, rows: Iterable[SessionMeasures], ref: str | Path
    ) -> ControlReport:
        return control_report(tuple(rows), self._source.limits(ref))
