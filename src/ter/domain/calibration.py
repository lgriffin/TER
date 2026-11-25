"""How far each detector has been judged on real sessions (TER-REC-004).

A detector's confidence rule is a claim until people judge its findings on
real sessions. Each judged sample records, per detector, how many findings
were waste, not waste or unsure. The samples are data, not code: they grow
as the owner judges more findings and are read through the
:class:`~ter.ports.driven.CalibrationSource` port, like prices (ADR 0003).

Counts are as judged, under the rule in force on the sample's date. A rule
tightened after a sample (``docs/ter4/l2-explained.md``) may no longer make
the findings that were judged not waste; the counts are kept, so a reader
sees how much judgement any figure rests on, not a precision estimate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

__all__ = [
    "CALIBRATION_SCHEMA",
    "Calibration",
    "CalibrationError",
    "DetectorCalibration",
    "JudgedCounts",
    "JudgedSample",
]

CALIBRATION_SCHEMA = "ter.calibration/1"


class CalibrationError(ValueError):
    """A calibration document that cannot be read as judged samples."""


@dataclass(frozen=True, slots=True)
class JudgedCounts:
    """Verdicts on one detector's findings in one sample."""

    waste: int = 0
    not_waste: int = 0
    unsure: int = 0

    def __post_init__(self) -> None:
        for name in ("waste", "not_waste", "unsure"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CalibrationError(f"{name} must be a count, got {value!r}")

    @property
    def judged(self) -> int:
        return self.waste + self.not_waste + self.unsure


@dataclass(frozen=True, slots=True)
class JudgedSample:
    """One judging session: who judged how many findings of which detectors."""

    id: str
    judged_on: date
    judges: int
    detectors: Mapping[str, JudgedCounts]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise CalibrationError("a judged sample needs an id")
        if isinstance(self.judges, bool) or self.judges < 1:
            raise CalibrationError(f"{self.id}: judges must be at least 1")


@dataclass(frozen=True, slots=True)
class DetectorCalibration:
    """Every judged finding of one detector, summed over the samples."""

    detector: str
    samples: int
    waste: int
    not_waste: int
    unsure: int

    @property
    def judged(self) -> int:
        return self.waste + self.not_waste + self.unsure

    def as_dict(self) -> dict[str, object]:
        return {
            "judged": self.judged,
            "waste": self.waste,
            "not_waste": self.not_waste,
            "unsure": self.unsure,
            "samples": self.samples,
        }


class Calibration:
    """The judged samples behind the detectors, newest last."""

    def __init__(self, samples: Iterable[JudgedSample] = ()) -> None:
        ordered = tuple(sorted(samples, key=lambda s: (s.judged_on, s.id)))
        ids = [s.id for s in ordered]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise CalibrationError(f"duplicate sample ids: {', '.join(duplicates)}")
        self.samples = ordered

    def of(self, detector: str) -> DetectorCalibration:
        """The detector's judged findings; all zero when it was never judged."""
        counts = [
            s.detectors[detector] for s in self.samples if detector in s.detectors
        ]
        return DetectorCalibration(
            detector=detector,
            samples=len(counts),
            waste=sum(c.waste for c in counts),
            not_waste=sum(c.not_waste for c in counts),
            unsure=sum(c.unsure for c in counts),
        )

    def detectors(self) -> tuple[str, ...]:
        """Every detector any sample judged, sorted."""
        return tuple(sorted({d for s in self.samples for d in s.detectors}))
