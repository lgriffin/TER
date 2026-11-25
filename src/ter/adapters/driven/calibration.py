"""Calibration adapters behind the :class:`~ter.ports.driven.CalibrationSource` port.

:class:`JsonCalibration` reads a ``ter.calibration/1`` file of judged
samples. With no path it reads the file shipped inside the ``ter`` package
(``ter/data/calibration.json``), so a new judged sample is a data change.
Every malformed document is refused with
:class:`~ter.domain.calibration.CalibrationError`.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from ...domain.calibration import (
    CALIBRATION_SCHEMA,
    Calibration,
    CalibrationError,
    JudgedCounts,
    JudgedSample,
)

__all__ = ["JsonCalibration", "default_calibration", "parse_calibration"]

_SAMPLE_KEYS = {"id", "judged_on", "judges", "source", "detectors"}
_COUNT_KEYS = {"waste", "not_waste", "unsure"}


def parse_calibration(document: Any) -> Calibration:
    """Validate a decoded calibration document and return its samples."""
    if not isinstance(document, dict):
        raise CalibrationError("a calibration document must be a JSON object")
    if document.get("schema") != CALIBRATION_SCHEMA:
        raise CalibrationError(
            f"unsupported calibration schema {document.get('schema')!r}; "
            f"expected {CALIBRATION_SCHEMA!r}"
        )
    unknown = set(document) - {"schema", "samples"}
    if unknown:
        raise CalibrationError(f"unknown keys: {', '.join(sorted(unknown))}")
    samples = document.get("samples")
    if not isinstance(samples, list):
        raise CalibrationError("'samples' must be a list")
    return Calibration(_sample(raw, i) for i, raw in enumerate(samples))


def _sample(raw: Any, index: int) -> JudgedSample:
    where = f"samples[{index}]"
    if not isinstance(raw, dict):
        raise CalibrationError(f"{where} must be an object")
    if set(raw) != _SAMPLE_KEYS:
        raise CalibrationError(f"{where} must have exactly {sorted(_SAMPLE_KEYS)}")
    sample_id, judged_on, judges = raw["id"], raw["judged_on"], raw["judges"]
    if not isinstance(sample_id, str):
        raise CalibrationError(f"{where}.id must be a string")
    if not isinstance(raw["source"], str) or not raw["source"].strip():
        raise CalibrationError(f"{where}.source must say where the verdicts are")
    if not isinstance(judged_on, str):
        raise CalibrationError(f"{where}.judged_on must be an ISO date")
    try:
        day = date.fromisoformat(judged_on)
    except ValueError as exc:
        raise CalibrationError(f"{where}.judged_on: {exc}") from None
    if isinstance(judges, bool) or not isinstance(judges, int):
        raise CalibrationError(f"{where}.judges must be a whole number")
    detectors = raw["detectors"]
    if not isinstance(detectors, dict):
        raise CalibrationError(f"{where}.detectors must be an object")
    counts: dict[str, JudgedCounts] = {}
    for name, value in detectors.items():
        if not isinstance(value, dict) or set(value) != _COUNT_KEYS:
            raise CalibrationError(
                f"{where}.detectors.{name} must have exactly {sorted(_COUNT_KEYS)}"
            )
        counts[name] = JudgedCounts(**value)
    return JudgedSample(sample_id, day, judges, counts)


class JsonCalibration:
    """Judged samples loaded from a ``ter.calibration/1`` JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            text = (
                resources.files("ter")
                .joinpath("data", "calibration.json")
                .read_text(encoding="utf-8")
            )
            self.name = "ter:data/calibration.json"
        else:
            text = Path(path).read_text(encoding="utf-8")
            self.name = str(path)
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"not JSON: {exc}") from None
        self._calibration = parse_calibration(document)

    def calibration(self) -> Calibration:
        return self._calibration


@lru_cache(maxsize=1)
def default_calibration() -> JsonCalibration:
    """The judged samples shipped with TER, loaded once per process."""
    return JsonCalibration()
