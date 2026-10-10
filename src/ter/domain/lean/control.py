"""Control charts: natural process limits for session measures (point 201).

Lean separates *common-cause* variation, the noise every session of a stable
process shows, from *special causes* worth investigating. TER charts each
session measure on an individuals and moving range (XmR) chart, the standard
chart for one value per unit of work (Wheeler, *Understanding Variation*):

* the centre line is the mean of the baseline sessions;
* the natural process limits are ``mean ± 2.66 × average moving range``
  (or ``± 3.145 × median moving range``, which one wild session cannot
  inflate); the moving range chart's upper limit is ``3.268 × average mR``
  (``3.865 × median mR``);
* sigma, for the zone rules, is ``average mR / 1.128`` (``median mR /
  0.954``).

A limit past a measure's natural boundary (below 0 for a count, outside 0 to
1 for a ratio) is no limit at all and is reported as ``None``; on that side
only a tuned limit can signal, and the zone and run rules are not applied,
since a skewed measure piled against its boundary says nothing there.

**Detection rules** (Western Electric, as Wheeler numbers them). Each is a
published rule a developer can switch on or off per measure:

* ``beyond_limits``: one session outside a limit;
* ``two_of_three``: two of three successive sessions beyond two sigma, on
  the same side;
* ``four_of_five``: four of five successive sessions beyond one sigma, on
  the same side;
* ``run_of_eight``: eight successive sessions on the same side of the centre
  line.

**Tuning.** Natural limits are what the process does; a developer may set
tighter (or looser) *action limits* for ``beyond_limits``, with a reason. The
limits document keeps both, so a tuned limit is never mistaken for the
process's own. The zone and run rules always use the natural centre and
sigma: they describe the process, not a target.

**Firing.** Every rule match is a :class:`ControlSignal`: the measure, the
rule, the sessions that make up the pattern (its evidence), the limit crossed
and whether the movement is unfavourable. Only measures whose basis is
structural (a ratio or a count) can fire; token totals and seconds are
charted but never fire, as no intervention may rest on a token count alone
(TER-INT-007). The signal is the typed value an L4 policy will name.

Measures come from the :class:`~.analysis.LeanAnalysis` only, never from the
outcome verdict (point 5), and limits record the detector set they were
computed with: a recalibrated detector changes what the finding counts mean,
so limits from another detector set are stale (:meth:`ControlLimits.stale`).
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar

from .analysis import UNCERTAIN, LeanAnalysis
from .model import ActivityClass, Finding, FindingKind, LeanWaste
from .wip import WipKind

__all__ = [
    "CONTROL_LIMITS_SCHEMA",
    "CONTROL_MEASURES",
    "CONTROL_MEASURES_SCHEMA",
    "CONTROL_REPORT_SCHEMA",
    "MIN_BASELINE",
    "SESSION_CONTROL_SCHEMA",
    "SETTLED_BASELINE",
    "Basis",
    "ControlChart",
    "ControlLimits",
    "ControlLimitsError",
    "ControlMeasure",
    "ControlPoint",
    "ControlReport",
    "ControlRule",
    "ControlSignal",
    "Direction",
    "LimitMethod",
    "MeasureLimits",
    "MeasuresDocument",
    "NaturalLimits",
    "Placement",
    "SessionControl",
    "SessionMeasures",
    "Side",
    "Tuning",
    "control_chart",
    "control_charts",
    "control_measure",
    "control_report",
    "compute_limits",
    "detector_fingerprint",
    "measure_session",
    "natural_limits",
    "order_sessions",
    "place",
    "pseudonymise",
    "session_control",
]

CONTROL_LIMITS_SCHEMA = "ter.control-limits/1"
CONTROL_MEASURES_SCHEMA = "ter.control-measures/1"
CONTROL_REPORT_SCHEMA = "ter.control-report/1"
SESSION_CONTROL_SCHEMA = "ter.session-control/1"

#: Fewest baseline sessions a measure needs before TER computes its limits.
MIN_BASELINE = 8
#: Below this many baseline sessions the limits are marked provisional:
#: usable, but expected to move as sessions are added.
SETTLED_BASELINE = 20

# XmR scaling factors for moving ranges of two successive values.
_E2_AVERAGE = 2.66  # 3 / d2, d2 = 1.128
_D4_AVERAGE = 3.268
_D2_AVERAGE = 1.128
_E2_MEDIAN = 3.145
_D4_MEDIAN = 3.865
_D2_MEDIAN = 0.954

#: Values within this distance count as equal when testing a limit, so a
#: value that rounds onto a limit is never a signal.
_EPSILON = 1e-9


class ControlLimitsError(ValueError):
    """A limits or measures document is malformed, or a baseline is too small."""


class Basis(StrEnum):
    """What a measure counts; only structural bases may fire (TER-INT-007)."""

    RATIO = "ratio"
    COUNT = "count"
    TOKENS = "tokens"
    SECONDS = "seconds"

    @property
    def fires(self) -> bool:
        return self in (Basis.RATIO, Basis.COUNT)


class Direction(StrEnum):
    """Which way a measure moves when the session got worse."""

    HIGHER_IS_WORSE = "higher_is_worse"
    LOWER_IS_WORSE = "lower_is_worse"


class Side(StrEnum):
    ABOVE = "above"
    BELOW = "below"


class LimitMethod(StrEnum):
    """How sigma is estimated from the moving ranges."""

    AVERAGE_MOVING_RANGE = "average_moving_range"
    MEDIAN_MOVING_RANGE = "median_moving_range"


class ControlRule(StrEnum):
    """The published detection rules (Western Electric, Wheeler's order)."""

    BEYOND_LIMITS = "beyond_limits"
    TWO_OF_THREE = "two_of_three"
    FOUR_OF_FIVE = "four_of_five"
    RUN_OF_EIGHT = "run_of_eight"

    @property
    def description(self) -> str:
        return _RULE_TEXT[self]


_RULE_TEXT: dict[ControlRule, str] = {
    ControlRule.BEYOND_LIMITS: "One session outside a control limit.",
    ControlRule.TWO_OF_THREE: (
        "Two of three successive sessions more than two sigma from the "
        "centre line, on the same side."
    ),
    ControlRule.FOUR_OF_FIVE: (
        "Four of five successive sessions more than one sigma from the "
        "centre line, on the same side."
    ),
    ControlRule.RUN_OF_EIGHT: (
        "Eight successive sessions on the same side of the centre line."
    ),
}

ALL_RULES: frozenset[ControlRule] = frozenset(ControlRule)

_E = TypeVar("_E", bound=StrEnum)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlMeasure:
    """A session measure TER can chart, and how to read it from an analysis."""

    key: str
    label: str
    basis: Basis
    direction: Direction
    read: Callable[[LeanAnalysis], float | None] = field(compare=False, repr=False)

    @property
    def lower_bound(self) -> float:
        return 0.0

    @property
    def upper_bound(self) -> float | None:
        return 1.0 if self.basis is Basis.RATIO else None

    @property
    def fires(self) -> bool:
        return self.basis.fires


def _share(part: float, whole: float) -> float | None:
    return part / whole if whole else None


def _activity(key: ActivityClass) -> Callable[[LeanAnalysis], float | None]:
    def read(a: LeanAnalysis) -> float | None:
        sc = a.scorecard
        return sc.activity_share(key) if sc.generated_tokens else None

    return read


def _unverified_waste(a: LeanAnalysis) -> float | None:
    sc = a.scorecard
    if not sc.generated_tokens:
        return None
    return sc.activity_share(ActivityClass.AVOIDABLE) + sc.activity_share(UNCERTAIN)


def _peak_wip(a: LeanAnalysis) -> float | None:
    return None if a.wip.peak is None else float(a.wip.peak.total)


def _ter(a: LeanAnalysis) -> float | None:
    return None if a.scorecard.ter is None else a.scorecard.ter.value


#: The measures TER charts, in report order. A key the scorecard dimensions
#: also report (TER-SCR-001) has the same value there, so a chart and an A3
#: agree.
CONTROL_MEASURES: tuple[ControlMeasure, ...] = (
    ControlMeasure(
        "avoidable_share",
        "Avoidable share of generated tokens",
        Basis.RATIO,
        Direction.HIGHER_IS_WORSE,
        _activity(ActivityClass.AVOIDABLE),
    ),
    # The owner's rule (10 Oct 2026): an uncertain finding is waste until a
    # judgement says otherwise, so the process is also charted with it.
    ControlMeasure(
        "unverified_waste_share",
        "Avoidable plus unverified share of generated tokens",
        Basis.RATIO,
        Direction.HIGHER_IS_WORSE,
        _unverified_waste,
    ),
    ControlMeasure(
        "value_adding_share",
        "Value-adding share of generated tokens",
        Basis.RATIO,
        Direction.LOWER_IS_WORSE,
        _activity(ActivityClass.VALUE_ADDING),
    ),
    ControlMeasure(
        "flow_efficiency_tokens",
        "Flow efficiency (tokens)",
        Basis.RATIO,
        Direction.LOWER_IS_WORSE,
        lambda a: a.scorecard.flow_efficiency_tokens,
    ),
    ControlMeasure(
        "flow_efficiency_time",
        "Flow efficiency (time)",
        Basis.RATIO,
        Direction.LOWER_IS_WORSE,
        lambda a: a.scorecard.flow_efficiency_time,
    ),
    ControlMeasure(
        "edits_validated_share",
        "Edits covered by a later validation run",
        Basis.RATIO,
        Direction.LOWER_IS_WORSE,
        lambda a: _share(a.wip.edits_validated, a.wip.edits_opened),
    ),
    ControlMeasure(
        "ter",
        "Token Efficiency Ratio",
        Basis.RATIO,
        Direction.LOWER_IS_WORSE,
        _ter,
    ),
    ControlMeasure(
        "confident_findings",
        "Confident waste findings",
        Basis.COUNT,
        Direction.HIGHER_IS_WORSE,
        # Scorecard findings include uncertain ones (ADR 0006).
        lambda a: float(a.scorecard.findings - a.scorecard.uncertain_findings),
    ),
    ControlMeasure(
        "uncertain_findings",
        "Uncertain findings (waste until verified)",
        Basis.COUNT,
        Direction.HIGHER_IS_WORSE,
        lambda a: float(a.scorecard.uncertain_findings),
    ),
    ControlMeasure(
        "rework_cycles",
        "Rework cycles (same failure)",
        Basis.COUNT,
        Direction.HIGHER_IS_WORSE,
        lambda a: float(a.scorecard.rework_cycles),
    ),
    ControlMeasure(
        "risk_findings",
        "Risk findings",
        Basis.COUNT,
        Direction.HIGHER_IS_WORSE,
        lambda a: float(a.scorecard.risks),
    ),
    ControlMeasure(
        "wip_peak",
        "Peak WIP",
        Basis.COUNT,
        Direction.HIGHER_IS_WORSE,
        _peak_wip,
    ),
    ControlMeasure(
        "unvalidated_edits_at_end",
        "Edits no validation run covered",
        Basis.COUNT,
        Direction.HIGHER_IS_WORSE,
        lambda a: float(len(a.wip.still_open(WipKind.EDITS))),
    ),
    ControlMeasure(
        "unresolved_failures_at_end",
        "Failing checks never seen passing again",
        Basis.COUNT,
        Direction.HIGHER_IS_WORSE,
        lambda a: float(len(a.wip.still_open(WipKind.FAILURES))),
    ),
    ControlMeasure(
        "generated_tokens",
        "Generated tokens",
        Basis.TOKENS,
        Direction.HIGHER_IS_WORSE,
        lambda a: float(a.scorecard.generated_tokens),
    ),
    ControlMeasure(
        "agent_seconds",
        "Agent time",
        Basis.SECONDS,
        Direction.HIGHER_IS_WORSE,
        lambda a: a.scorecard.agent_seconds,
    ),
)

_BY_KEY: dict[str, ControlMeasure] = {m.key: m for m in CONTROL_MEASURES}


def control_measure(key: str) -> ControlMeasure:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise ControlLimitsError(
            f"unknown control measure {key!r} (known: {', '.join(_BY_KEY)})"
        ) from None


def detector_fingerprint(analysis: LeanAnalysis) -> str:
    """A short hash of the detectors and confidence rules an analysis ran."""
    payload = json.dumps(sorted(list(d) for d in analysis.detectors))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class SessionMeasures:
    """One session's chartable measures; ``None`` means the measure is unknown."""

    session_id: str
    started_at: datetime | None
    values: Mapping[str, float | None]
    detectors: str

    def value(self, key: str) -> float | None:
        return self.values.get(key)

    def as_dict(self) -> dict[str, object]:
        return {
            "session": self.session_id,
            "started_at": None
            if self.started_at is None
            else self.started_at.isoformat(),
            "values": {
                k: None if v is None else round(v, 6) for k, v in self.values.items()
            },
        }

    @classmethod
    def from_mapping(cls, data: object, detectors: str) -> SessionMeasures:
        if not isinstance(data, Mapping):
            raise ControlLimitsError("each session must be an object")
        session = data.get("session")
        if not isinstance(session, str) or not session:
            raise ControlLimitsError("each session needs a non-empty 'session' id")
        started_raw = data.get("started_at")
        started: datetime | None = None
        if started_raw is not None:
            if not isinstance(started_raw, str):
                raise ControlLimitsError(f"{session}: started_at must be a string")
            try:
                started = datetime.fromisoformat(started_raw)
            except ValueError as exc:
                raise ControlLimitsError(f"{session}: started_at: {exc}") from None
        raw = data.get("values")
        if not isinstance(raw, Mapping):
            raise ControlLimitsError(f"{session}: values must be an object")
        values: dict[str, float | None] = {}
        for key, value in raw.items():
            control_measure(str(key))
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float)
            ):
                raise ControlLimitsError(f"{session}: {key} must be a number or null")
            if value is not None and not math.isfinite(value):
                raise ControlLimitsError(f"{session}: {key} must be finite")
            values[str(key)] = None if value is None else float(value)
        return cls(session, started, values, detectors)


def measure_session(
    analysis: LeanAnalysis,
    started_at: datetime | None = None,
    session_id: str | None = None,
) -> SessionMeasures:
    """Every control measure of one analysed session."""
    return SessionMeasures(
        session_id or analysis.session_id or "session",
        started_at,
        {m.key: m.read(analysis) for m in CONTROL_MEASURES},
        detector_fingerprint(analysis),
    )


def order_sessions(rows: Iterable[SessionMeasures]) -> tuple[SessionMeasures, ...]:
    """Process order: by start time, sessions without one last, then by id.

    Run rules read sessions in the order the process produced them, so the
    order is part of the chart.
    """

    def key(row: SessionMeasures) -> tuple[bool, float, str]:
        started = row.started_at
        stamp = 0.0 if started is None else _timestamp(started)
        return (started is None, stamp, row.session_id)

    return tuple(sorted(rows, key=key))


def _timestamp(moment: datetime) -> float:
    # Naive times are read as UTC so mixed sources still sort.
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaturalLimits:
    """What the process does: XmR limits computed from baseline sessions."""

    centre: float
    ucl: float | None
    lcl: float | None
    sigma: float
    moving_range: float
    mr_ucl: float
    sessions: int
    method: LimitMethod

    @property
    def provisional(self) -> bool:
        return self.sessions < SETTLED_BASELINE

    def as_dict(self) -> dict[str, object]:
        def r(v: float | None) -> float | None:
            return None if v is None else round(v, 6)

        return {
            "centre": r(self.centre),
            "ucl": r(self.ucl),
            "lcl": r(self.lcl),
            "sigma": r(self.sigma),
            "moving_range": r(self.moving_range),
            "mr_ucl": r(self.mr_ucl),
            "sessions": self.sessions,
            "provisional": self.provisional,
            "method": self.method.value,
        }


def natural_limits(
    values: Sequence[float],
    method: LimitMethod = LimitMethod.AVERAGE_MOVING_RANGE,
    *,
    lower_bound: float | None = 0.0,
    upper_bound: float | None = None,
) -> NaturalLimits:
    """XmR natural process limits of ``values`` in process order.

    A median moving range of 0 (most successive sessions equal, as in a
    count that is usually 0) would collapse the limits onto the centre and
    flag every session, so that measure falls back to the average moving
    range and records it (TER-SPC-014).

    Raises:
        ControlLimitsError: With fewer than :data:`MIN_BASELINE` values.
    """
    if len(values) < MIN_BASELINE:
        raise ControlLimitsError(
            f"{len(values)} session(s) is too few for control limits "
            f"(at least {MIN_BASELINE})"
        )
    centre = statistics.fmean(values)
    ranges = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
    if method is LimitMethod.MEDIAN_MOVING_RANGE and statistics.median(ranges) == 0:
        method = LimitMethod.AVERAGE_MOVING_RANGE
    if method is LimitMethod.MEDIAN_MOVING_RANGE:
        mr = statistics.median(ranges)
        e2, d4, d2 = _E2_MEDIAN, _D4_MEDIAN, _D2_MEDIAN
    else:
        mr = statistics.fmean(ranges)
        e2, d4, d2 = _E2_AVERAGE, _D4_AVERAGE, _D2_AVERAGE
    ucl: float | None = centre + e2 * mr
    lcl: float | None = centre - e2 * mr
    if upper_bound is not None and ucl is not None and ucl >= upper_bound:
        ucl = None
    if lower_bound is not None and lcl is not None and lcl <= lower_bound:
        lcl = None
    return NaturalLimits(centre, ucl, lcl, mr / d2, mr, d4 * mr, len(values), method)


@dataclass(frozen=True)
class Tuning:
    """A developer's action limits, which replace the natural ones for
    ``beyond_limits`` only. ``None`` keeps that side's natural limit."""

    ucl: float | None
    lcl: float | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"ucl": self.ucl, "lcl": self.lcl, "reason": self.reason}


@dataclass(frozen=True)
class MeasureLimits:
    """One measure's limits, tuning and the rules switched on for it."""

    measure: ControlMeasure
    natural: NaturalLimits
    tuning: Tuning | None = None
    rules: frozenset[ControlRule] = ALL_RULES
    enabled: bool = True

    @property
    def ucl(self) -> float | None:
        """The action limit above: tuned if set, else natural."""
        if self.tuning is not None and self.tuning.ucl is not None:
            return self.tuning.ucl
        return self.natural.ucl

    @property
    def lcl(self) -> float | None:
        if self.tuning is not None and self.tuning.lcl is not None:
            return self.tuning.lcl
        return self.natural.lcl

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.measure.label,
            "basis": self.measure.basis.value,
            "direction": self.measure.direction.value,
            "fires": self.measure.fires,
            "enabled": self.enabled,
            "rules": [r.value for r in ControlRule if r in self.rules],
            "natural": self.natural.as_dict(),
            "tuned": None if self.tuning is None else self.tuning.as_dict(),
        }


@dataclass(frozen=True)
class ControlLimits:
    """A limits document (``ter.control-limits/1``): one entry per measure."""

    measures: tuple[MeasureLimits, ...]
    method: LimitMethod
    detectors: str
    sessions: int
    computed_on: str | None = None

    def get(self, key: str) -> MeasureLimits | None:
        return next((m for m in self.measures if m.measure.key == key), None)

    def stale(self, detectors: str) -> bool:
        """True when these limits came from another detector set."""
        return detectors != self.detectors

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": CONTROL_LIMITS_SCHEMA,
            "computed_on": self.computed_on,
            "method": self.method.value,
            "detectors": self.detectors,
            "sessions": self.sessions,
            "rules": {r.value: r.description for r in ControlRule},
            "measures": {m.measure.key: m.as_dict() for m in self.measures},
        }

    @classmethod
    def from_mapping(cls, data: object) -> ControlLimits:
        """Validate a decoded limits document.

        Raises:
            ControlLimitsError: On a wrong schema, an unknown measure or rule,
                a tuning without a reason, or a tuned lower limit at or above
                the upper one.
        """
        if not isinstance(data, Mapping):
            raise ControlLimitsError("a limits document must be a JSON object")
        if data.get("schema") != CONTROL_LIMITS_SCHEMA:
            raise ControlLimitsError(
                f"schema must be {CONTROL_LIMITS_SCHEMA!r}, got {data.get('schema')!r}"
            )
        method = _enum(LimitMethod, data.get("method"), "method")
        detectors = data.get("detectors")
        if not isinstance(detectors, str) or not detectors:
            raise ControlLimitsError("detectors must name the detector fingerprint")
        sessions = data.get("sessions")
        if not isinstance(sessions, int) or isinstance(sessions, bool):
            raise ControlLimitsError("sessions must be an integer")
        computed_on = data.get("computed_on")
        if computed_on is not None and not isinstance(computed_on, str):
            raise ControlLimitsError("computed_on must be a date string or null")
        raw = data.get("measures")
        if not isinstance(raw, Mapping):
            raise ControlLimitsError("measures must be an object keyed by measure")
        measures = tuple(
            _measure_limits(str(key), entry, method) for key, entry in raw.items()
        )
        order = {m.key: i for i, m in enumerate(CONTROL_MEASURES)}
        return cls(
            tuple(sorted(measures, key=lambda m: order[m.measure.key])),
            method,
            detectors,
            sessions,
            computed_on,
        )


def _enum(kind: type[_E], value: object, where: str) -> _E:
    try:
        return kind(str(value))
    except ValueError:
        known = ", ".join(e.value for e in kind)
        raise ControlLimitsError(
            f"{where}: unknown value {value!r} (known: {known})"
        ) from None


def _number(value: object, where: str, *, optional: bool = True) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ControlLimitsError(f"{where} must be a number")
    if not math.isfinite(value):
        raise ControlLimitsError(f"{where} must be finite")
    return float(value)


def _measure_limits(key: str, entry: object, method: LimitMethod) -> MeasureLimits:
    measure = control_measure(key)
    if not isinstance(entry, Mapping):
        raise ControlLimitsError(f"{key}: must be an object")
    nat = entry.get("natural")
    if not isinstance(nat, Mapping):
        raise ControlLimitsError(f"{key}: natural limits are required")
    centre = _number(nat.get("centre"), f"{key}.natural.centre", optional=False)
    sigma = _number(nat.get("sigma"), f"{key}.natural.sigma", optional=False)
    mr = _number(nat.get("moving_range"), f"{key}.natural.moving_range", optional=False)
    mr_ucl = _number(nat.get("mr_ucl"), f"{key}.natural.mr_ucl", optional=False)
    count = nat.get("sessions")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ControlLimitsError(f"{key}.natural.sessions must be a positive integer")
    assert centre is not None and sigma is not None and mr is not None
    assert mr_ucl is not None
    natural = NaturalLimits(
        centre,
        _number(nat.get("ucl"), f"{key}.natural.ucl"),
        _number(nat.get("lcl"), f"{key}.natural.lcl"),
        sigma,
        mr,
        mr_ucl,
        count,
        # Older files have no per-measure method: the file's method applies.
        _enum(LimitMethod, nat.get("method", method.value), f"{key}.natural.method"),
    )
    _check_natural(key, natural, measure)
    tuning = _tuning(key, entry.get("tuned"), measure)
    if tuning is not None:
        upper = natural.ucl if tuning.ucl is None else tuning.ucl
        lower = natural.lcl if tuning.lcl is None else tuning.lcl
        if upper is not None and lower is not None and lower >= upper:
            raise ControlLimitsError(
                f"{key}: the effective lower limit {lower} must be below the "
                f"effective upper limit {upper} (tuned and natural combined)"
            )
    rules_raw = entry.get("rules", [r.value for r in ControlRule])
    if not isinstance(rules_raw, list):
        raise ControlLimitsError(f"{key}.rules must be a list of rule names")
    rules = frozenset(_enum(ControlRule, r, f"{key}.rules") for r in rules_raw)
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ControlLimitsError(f"{key}.enabled must be true or false")
    return MeasureLimits(measure, natural, tuning, rules, enabled)


def _check_natural(key: str, natural: NaturalLimits, measure: ControlMeasure) -> None:
    """Natural limits read from a file must still describe a process: in
    order, inside the measure's bounds, with a non-negative spread."""
    upper = measure.upper_bound
    for name, value in (
        ("centre", natural.centre),
        ("ucl", natural.ucl),
        ("lcl", natural.lcl),
    ):
        if value is None:
            continue
        if value < measure.lower_bound or (upper is not None and value > upper):
            bound = f"{measure.lower_bound} to {upper}" if upper else "0 or more"
            raise ControlLimitsError(f"{key}.natural.{name} must be {bound}")
    if min(natural.sigma, natural.moving_range, natural.mr_ucl) < 0:
        raise ControlLimitsError(f"{key}.natural spread values must be 0 or more")
    if natural.ucl is not None and natural.ucl < natural.centre:
        raise ControlLimitsError(f"{key}.natural.ucl must not be below the centre")
    if natural.lcl is not None and natural.lcl > natural.centre:
        raise ControlLimitsError(f"{key}.natural.lcl must not be above the centre")


def _tuning(key: str, raw: object, measure: ControlMeasure) -> Tuning | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ControlLimitsError(f"{key}.tuned must be an object or null")
    ucl = _number(raw.get("ucl"), f"{key}.tuned.ucl")
    lcl = _number(raw.get("lcl"), f"{key}.tuned.lcl")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ControlLimitsError(
            f"{key}.tuned needs a reason: a tuned limit says why it differs "
            "from the process's natural limit"
        )
    if ucl is None and lcl is None:
        raise ControlLimitsError(f"{key}.tuned sets neither ucl nor lcl")
    if ucl is not None and lcl is not None and lcl >= ucl:
        raise ControlLimitsError(f"{key}.tuned: lcl must be below ucl")
    upper = measure.upper_bound
    for name, value in (("ucl", ucl), ("lcl", lcl)):
        if value is None:
            continue
        if value < measure.lower_bound or (upper is not None and value > upper):
            bound = f"{measure.lower_bound} to {upper}" if upper else "0 or more"
            raise ControlLimitsError(f"{key}.tuned.{name} must be {bound}")
    return Tuning(ucl, lcl, reason.strip())


def compute_limits(
    rows: Sequence[SessionMeasures],
    method: LimitMethod = LimitMethod.AVERAGE_MOVING_RANGE,
    *,
    keep: ControlLimits | None = None,
    computed_on: str | None = None,
) -> ControlLimits:
    """Natural limits for every measure with enough baseline sessions.

    ``keep`` carries a developer's tuning, rule choices and switched-off
    measures over from an earlier limits document, so recomputing the
    baseline never discards a decision. A measure with fewer than
    :data:`MIN_BASELINE` known values is left out, unless ``keep`` has it:
    then the earlier entry stands unchanged, natural limits included.

    Raises:
        ControlLimitsError: When the rows mix detector sets, or none are given.
    """
    if not rows:
        raise ControlLimitsError("no sessions to compute limits from")
    fingerprints = {r.detectors for r in rows}
    if len(fingerprints) > 1:
        raise ControlLimitsError(
            "sessions were measured with different detector sets "
            f"({', '.join(sorted(fingerprints))}); re-measure them together"
        )
    ordered = order_sessions(rows)
    out: list[MeasureLimits] = []
    for measure in CONTROL_MEASURES:
        values = [v for r in ordered if (v := r.value(measure.key)) is not None]
        previous = None if keep is None else keep.get(measure.key)
        if len(values) < MIN_BASELINE:
            if previous is not None:
                # Too few sessions to recompute: the earlier entry, with its
                # natural limits and the developer's decisions, stands.
                out.append(previous)
            continue
        natural = natural_limits(
            values,
            method,
            lower_bound=measure.lower_bound,
            upper_bound=measure.upper_bound,
        )
        entry = MeasureLimits(measure, natural)
        if previous is not None:
            entry = replace(
                entry,
                tuning=previous.tuning,
                rules=previous.rules,
                enabled=previous.enabled,
            )
        out.append(entry)
    return ControlLimits(
        tuple(out), method, fingerprints.pop(), len(ordered), computed_on
    )


# ---------------------------------------------------------------------------
# Charts and signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlPoint:
    """One session on a chart."""

    index: int
    session_id: str
    value: float
    #: Absolute difference from the previous session (``None`` for the first).
    moving_range: float | None
    #: Distance from the centre line in natural sigmas (signed).
    sigmas: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "session": self.session_id,
            "value": round(self.value, 6),
            "moving_range": None
            if self.moving_range is None
            else round(self.moving_range, 6),
            "sigmas": _finite(self.sigmas),
        }


def _finite(value: float, digits: int = 3) -> float | None:
    """A JSON-safe number: ``None`` for the infinite sigmas of a zero-spread
    process."""
    return round(value, digits) if math.isfinite(value) else None


@dataclass(frozen=True)
class ControlSignal:
    """A detection rule matched: the typed signal an L4 policy can name."""

    measure: str
    rule: ControlRule
    side: Side
    #: The session at which the rule completed, and its value.
    session_id: str
    value: float
    #: The limit or centre line the rule tested against.
    limit: float
    #: Every session that forms the pattern, in process order: the evidence.
    evidence: tuple[str, ...]
    #: True when the movement is the way the measure gets worse.
    unfavourable: bool
    #: True when this signal may drive an intervention: unfavourable, on a
    #: structural measure (never a token count, TER-INT-007).
    fires: bool
    tuned: bool = False

    @property
    def id(self) -> str:
        return f"control.{self.measure}.{self.rule.value}"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "measure": self.measure,
            "rule": self.rule.value,
            "rule_text": self.rule.description,
            "side": self.side.value,
            "session": self.session_id,
            "value": round(self.value, 6),
            "limit": round(self.limit, 6),
            "evidence": list(self.evidence),
            "unfavourable": self.unfavourable,
            "fires": self.fires,
            "tuned": self.tuned,
        }


@dataclass(frozen=True)
class ControlChart:
    """An XmR chart of one measure over sessions, with its signals."""

    limits: MeasureLimits
    points: tuple[ControlPoint, ...]
    signals: tuple[ControlSignal, ...]

    @property
    def measure(self) -> ControlMeasure:
        return self.limits.measure

    def signalled(self, index: int) -> tuple[ControlSignal, ...]:
        sid = self.points[index].session_id
        return tuple(s for s in self.signals if s.session_id == sid)

    def as_dict(self) -> dict[str, object]:
        return {
            "measure": self.measure.key,
            "limits": self.limits.as_dict(),
            "points": [p.as_dict() for p in self.points],
            "signals": [s.as_dict() for s in self.signals],
        }


def _unfavourable(measure: ControlMeasure, side: Side) -> bool:
    if measure.direction is Direction.HIGHER_IS_WORSE:
        return side is Side.ABOVE
    return side is Side.BELOW


def _signal(
    limits: MeasureLimits,
    rule: ControlRule,
    side: Side,
    points: Sequence[ControlPoint],
    limit: float,
    tuned: bool = False,
) -> ControlSignal:
    last = points[-1]
    unfavourable = _unfavourable(limits.measure, side)
    return ControlSignal(
        limits.measure.key,
        rule,
        side,
        last.session_id,
        last.value,
        limit,
        tuple(p.session_id for p in points),
        unfavourable,
        unfavourable and limits.measure.fires,
        tuned,
    )


def _beyond(limits: MeasureLimits, point: ControlPoint) -> ControlSignal | None:
    ucl, lcl = limits.ucl, limits.lcl
    tuning = limits.tuning
    if ucl is not None and point.value > ucl + _EPSILON:
        tuned = tuning is not None and tuning.ucl is not None
        return _signal(
            limits, ControlRule.BEYOND_LIMITS, Side.ABOVE, (point,), ucl, tuned
        )
    if lcl is not None and point.value < lcl - _EPSILON:
        tuned = tuning is not None and tuning.lcl is not None
        return _signal(
            limits, ControlRule.BEYOND_LIMITS, Side.BELOW, (point,), lcl, tuned
        )
    return None


def _open_sides(limits: MeasureLimits) -> tuple[tuple[Side, float], ...]:
    """The sides whose natural limit lies inside the measure's bounds.

    Where the boundary cuts a limit off (no lower limit for a count that is
    mostly 0), the zones and the centre line on that side say nothing about
    a special cause, so the zone and run rules are not applied there.
    """
    natural = limits.natural
    sides: list[tuple[Side, float]] = []
    if natural.ucl is not None:
        sides.append((Side.ABOVE, 1.0))
    if natural.lcl is not None:
        sides.append((Side.BELOW, -1.0))
    return tuple(sides)


def _zone_rule(
    limits: MeasureLimits,
    points: Sequence[ControlPoint],
    i: int,
    rule: ControlRule,
    window: int,
    needed: int,
    sigmas: float,
) -> ControlSignal | None:
    """``needed`` of the last ``window`` points beyond ``sigmas`` on one side,
    the current point among them."""
    if i + 1 < window or limits.natural.sigma <= 0:
        return None
    current = points[i]
    centre, sigma = limits.natural.centre, limits.natural.sigma
    for side, sign in _open_sides(limits):
        if sign * current.sigmas <= sigmas + _EPSILON:
            continue
        span = points[i + 1 - window : i + 1]
        beyond = [p for p in span if sign * p.sigmas > sigmas + _EPSILON]
        if len(beyond) >= needed:
            return _signal(limits, rule, side, beyond, centre + sign * sigmas * sigma)
    return None


def _run_rule(
    limits: MeasureLimits, points: Sequence[ControlPoint], i: int, length: int = 8
) -> ControlSignal | None:
    if i + 1 < length:
        return None
    centre = limits.natural.centre
    span = points[i + 1 - length : i + 1]
    for side, sign in _open_sides(limits):
        if all(sign * (p.value - centre) > _EPSILON for p in span):
            return _signal(limits, ControlRule.RUN_OF_EIGHT, side, span, centre)
    return None


def _points(
    limits: MeasureLimits, values: Sequence[tuple[str, float]]
) -> tuple[ControlPoint, ...]:
    centre, sigma = limits.natural.centre, limits.natural.sigma
    out: list[ControlPoint] = []
    previous: float | None = None
    for i, (sid, value) in enumerate(values):
        z = 0.0
        if sigma > 0:
            z = (value - centre) / sigma
        elif abs(value - centre) > _EPSILON:
            z = math.copysign(math.inf, value - centre)
        mr = None if previous is None else abs(value - previous)
        out.append(ControlPoint(i, sid, value, mr, z))
        previous = value
    return tuple(out)


def control_chart(
    limits: MeasureLimits, values: Sequence[tuple[str, float]]
) -> ControlChart:
    """Chart ``(session id, value)`` pairs in process order against ``limits``.

    A switched-off measure is still charted but raises no signal; a rule
    not in the measure's rule set is not tested.
    """
    points = _points(limits, values)
    signals: list[ControlSignal] = []
    if limits.enabled:
        on = limits.rules
        for i, point in enumerate(points):
            candidates = (
                _beyond(limits, point) if ControlRule.BEYOND_LIMITS in on else None,
                _zone_rule(limits, points, i, ControlRule.TWO_OF_THREE, 3, 2, 2.0)
                if ControlRule.TWO_OF_THREE in on
                else None,
                _zone_rule(limits, points, i, ControlRule.FOUR_OF_FIVE, 5, 4, 1.0)
                if ControlRule.FOUR_OF_FIVE in on
                else None,
                _run_rule(limits, points, i)
                if ControlRule.RUN_OF_EIGHT in on
                else None,
            )
            signals.extend(s for s in candidates if s is not None)
    return ControlChart(limits, points, tuple(signals))


def control_charts(
    rows: Sequence[SessionMeasures], limits: ControlLimits
) -> tuple[ControlChart, ...]:
    """One chart per measure in ``limits``, over ``rows`` in process order.

    A session where the measure is undefined (no edits for an edit share, no
    generated tokens for a token share) is not part of that measure's
    process, so the chart, its moving ranges and its rules run over the
    sessions where it is defined, exactly as :func:`compute_limits` does.
    """
    ordered = order_sessions(rows)
    charts: list[ControlChart] = []
    for entry in limits.measures:
        key = entry.measure.key
        values = [(r.session_id, v) for r in ordered if (v := r.value(key)) is not None]
        charts.append(control_chart(entry, values))
    return tuple(charts)


@dataclass(frozen=True)
class Placement:
    """Where one session's measure falls against its limits."""

    limits: MeasureLimits
    value: float
    sigmas: float
    signal: ControlSignal | None

    @property
    def checked(self) -> bool:
        """Whether ``beyond_limits`` was applied: the measure is switched on
        and the rule is selected for it. An unchecked measure is neither in
        nor out of its limits."""
        return self.limits.enabled and ControlRule.BEYOND_LIMITS in self.limits.rules

    @property
    def in_control(self) -> bool:
        return self.signal is None

    def as_dict(self) -> dict[str, object]:
        return {
            "measure": self.limits.measure.key,
            "value": round(self.value, 6),
            "centre": round(self.limits.natural.centre, 6),
            "ucl": None if self.limits.ucl is None else round(self.limits.ucl, 6),
            "lcl": None if self.limits.lcl is None else round(self.limits.lcl, 6),
            "sigmas": _finite(self.sigmas),
            "signal": None if self.signal is None else self.signal.as_dict(),
        }


def place(limits: MeasureLimits, session_id: str, value: float) -> Placement:
    """One session against its measure's limits: only ``beyond_limits`` can
    apply to a single session; the other rules need the sessions before it."""
    (point,) = _points(limits, ((session_id, value),))
    signal = (
        _beyond(limits, point)
        if limits.enabled and ControlRule.BEYOND_LIMITS in limits.rules
        else None
    )
    return Placement(limits, value, point.sigmas, signal)


@dataclass(frozen=True)
class ControlReport:
    """Every chart of a set of sessions against one limits document."""

    limits: ControlLimits
    charts: tuple[ControlChart, ...]
    sessions: int
    #: The detector set the charted sessions were measured with.
    detectors: str

    @property
    def stale(self) -> bool:
        """The limits came from another detector set than these sessions."""
        return self.limits.stale(self.detectors)

    @property
    def signals(self) -> tuple[ControlSignal, ...]:
        return tuple(s for c in self.charts for s in c.signals)

    @property
    def firing(self) -> tuple[ControlSignal, ...]:
        return tuple(s for s in self.signals if s.fires)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": CONTROL_REPORT_SCHEMA,
            "sessions": self.sessions,
            "detectors": self.detectors,
            "limits_detectors": self.limits.detectors,
            "stale": self.stale,
            "method": self.limits.method.value,
            "signals": len(self.signals),
            "firing": len(self.firing),
            "charts": [c.as_dict() for c in self.charts],
        }


def control_report(
    rows: Sequence[SessionMeasures], limits: ControlLimits
) -> ControlReport:
    """Chart ``rows`` against ``limits``.

    Raises:
        ControlLimitsError: When the rows mix detector sets, or none are given.
    """
    if not rows:
        raise ControlLimitsError("no sessions to chart")
    fingerprints = {r.detectors for r in rows}
    if len(fingerprints) > 1:
        raise ControlLimitsError(
            "sessions were measured with different detector sets "
            f"({', '.join(sorted(fingerprints))}); re-measure them together"
        )
    return ControlReport(
        limits, control_charts(rows, limits), len(rows), fingerprints.pop()
    )


Claims = Callable[[Finding], bool]


def _waste(f: Finding) -> bool:
    return f.kind is FindingKind.WASTE


def _every(claims: Claims) -> Callable[[LeanAnalysis], Claims]:
    return lambda _a: claims


def _citing_open(kind: WipKind) -> Callable[[LeanAnalysis], Claims]:
    """Findings that cite an item of ``kind`` still open at the end."""

    def claims(a: LeanAnalysis) -> Claims:
        still_open = frozenset(a.wip.still_open(kind))
        return lambda f: not still_open.isdisjoint(f.evidence)

    return claims


def _up_to_peak(a: LeanAnalysis) -> Claims:
    """Findings that cite work done up to the WIP peak, the work that piled up."""
    peak = a.wip.peak
    order = {s.event_id: s.index for s in a.steps}
    last = None if peak is None else order.get(peak.event_id)
    if last is None:
        return lambda _f: False
    return lambda f: any(order.get(e, last + 1) <= last for e in f.evidence)


#: The findings that move each measure, so a session outside a limit points
#: at its root causes (TER-SPC-011). Shares, flow and totals move with every
#: waste finding; counts with the findings they count; the end-of-session WIP
#: counts with the findings that cite the work still open; the WIP peak with
#: the findings that cite the work done up to it.
_BEHIND: Mapping[str, Callable[[LeanAnalysis], Claims]] = {
    "avoidable_share": _every(lambda f: _waste(f) and not f.uncertain),
    "unverified_waste_share": _every(_waste),
    "value_adding_share": _every(_waste),
    "flow_efficiency_tokens": _every(_waste),
    "flow_efficiency_time": _every(_waste),
    "edits_validated_share": _citing_open(WipKind.EDITS),
    "ter": _every(_waste),
    "confident_findings": _every(lambda f: _waste(f) and not f.uncertain),
    "uncertain_findings": _every(lambda f: _waste(f) and f.uncertain),
    "rework_cycles": _every(lambda f: f.waste is LeanWaste.REWORK),
    "risk_findings": _every(lambda f: f.kind is FindingKind.RISK),
    "wip_peak": _up_to_peak,
    "unvalidated_edits_at_end": _citing_open(WipKind.EDITS),
    "unresolved_failures_at_end": _citing_open(WipKind.FAILURES),
    "generated_tokens": _every(_waste),
    "agent_seconds": _every(_waste),
}


@dataclass(frozen=True)
class SessionControl:
    """One session placed against a limits document, for its A3.

    ``behind`` names, for each measure outside a limit in the unfavourable
    direction, the findings that move that measure: the root causes to read
    first. A favourable signal has none: nothing went wrong.
    """

    limits: ControlLimits
    detectors: str
    placements: tuple[Placement, ...]
    behind: Mapping[str, tuple[str, ...]]

    @property
    def stale(self) -> bool:
        return self.limits.stale(self.detectors)

    @property
    def signals(self) -> tuple[ControlSignal, ...]:
        return tuple(p.signal for p in self.placements if p.signal is not None)

    @property
    def firing(self) -> tuple[ControlSignal, ...]:
        return tuple(s for s in self.signals if s.fires)

    def as_dict(self) -> dict[str, object]:
        measures = []
        for p in self.placements:
            entry = p.as_dict()
            entry["label"] = p.limits.measure.label
            entry["fires"] = p.limits.measure.fires
            entry["checked"] = p.checked
            if p.signal is not None and p.signal.unfavourable:
                entry["findings"] = list(self.behind.get(p.limits.measure.key, ()))
            measures.append(entry)
        return {
            "schema": SESSION_CONTROL_SCHEMA,
            "method": self.limits.method.value,
            "computed_on": self.limits.computed_on,
            "detectors": self.detectors,
            "limits_detectors": self.limits.detectors,
            "stale": self.stale,
            "signals": len(self.signals),
            "firing": len(self.firing),
            "measures": measures,
        }


def session_control(analysis: LeanAnalysis, limits: ControlLimits) -> SessionControl:
    """Place every measure of one session against ``limits`` (TER-SPC-011).

    Only ``beyond_limits`` applies to one session; the zone and run rules
    need the sessions around it, which the control chart shows. A measure
    the limits leave out, or that is undefined for this session, is skipped.
    """
    row = measure_session(analysis)
    placements: list[Placement] = []
    behind: dict[str, tuple[str, ...]] = {}
    for measure in CONTROL_MEASURES:
        entry = limits.get(measure.key)
        value = row.value(measure.key)
        if entry is None or value is None:
            continue
        placed = place(entry, row.session_id, value)
        placements.append(placed)
        if placed.signal is not None and placed.signal.unfavourable:
            claims = _BEHIND[measure.key](analysis)
            behind[measure.key] = tuple(f.id for f in analysis.findings if claims(f))
    return SessionControl(limits, row.detectors, tuple(placements), behind)


@dataclass(frozen=True)
class MeasuresDocument:
    """A content-free measures file (``ter.control-measures/1``): one row per
    session, measures only, no prompt, path or code."""

    rows: tuple[SessionMeasures, ...]

    def as_dict(self) -> dict[str, object]:
        fingerprints = sorted({r.detectors for r in self.rows})
        return {
            "schema": CONTROL_MEASURES_SCHEMA,
            "detectors": fingerprints[0] if len(fingerprints) == 1 else fingerprints,
            "measures": {m.key: m.label for m in CONTROL_MEASURES},
            "sessions": [r.as_dict() for r in order_sessions(self.rows)],
        }

    @classmethod
    def from_mapping(cls, data: object) -> MeasuresDocument:
        if not isinstance(data, Mapping):
            raise ControlLimitsError("a measures document must be a JSON object")
        if data.get("schema") != CONTROL_MEASURES_SCHEMA:
            raise ControlLimitsError(
                f"schema must be {CONTROL_MEASURES_SCHEMA!r}, "
                f"got {data.get('schema')!r}"
            )
        detectors = data.get("detectors")
        if not isinstance(detectors, str) or not detectors:
            raise ControlLimitsError(
                "detectors must name one detector fingerprint; re-measure "
                "sessions from different detector sets together"
            )
        sessions = data.get("sessions")
        if not isinstance(sessions, list):
            raise ControlLimitsError("sessions must be a list")
        return cls(tuple(SessionMeasures.from_mapping(s, detectors) for s in sessions))


def pseudonymise(rows: Iterable[SessionMeasures]) -> tuple[SessionMeasures, ...]:
    """Replace each session id with a short hash of it, so a measures file
    can leave a private corpus without naming its sessions."""
    return tuple(
        replace(
            r,
            session_id="s-"
            + hashlib.sha256(r.session_id.encode("utf-8")).hexdigest()[:12],
        )
        for r in rows
    )
