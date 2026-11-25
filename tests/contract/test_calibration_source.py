"""Contract suite for the ``CalibrationSource`` port (TER-REC-004).

The shipped JSON samples and the in-memory fake run the same assertions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from ter.adapters.driven.calibration import JsonCalibration
from ter.adapters.driven.in_memory import InMemoryCalibration
from ter.domain.calibration import CalibrationError, JudgedCounts, JudgedSample
from ter.ports import CalibrationSource

Factory = Callable[[], CalibrationSource]


def _json() -> CalibrationSource:
    return JsonCalibration()


def _memory() -> CalibrationSource:
    return InMemoryCalibration(JsonCalibration().calibration().samples)


@pytest.fixture(params=[_json, _memory], ids=["json", "in-memory"])
def source(request: pytest.FixtureRequest) -> CalibrationSource:
    factory: Factory = request.param
    return factory()


@pytest.mark.req("TER-REC-004")
def test_satisfies_the_port_protocol(source: CalibrationSource) -> None:
    assert isinstance(source, CalibrationSource)
    assert source.name


@pytest.mark.req("TER-REC-004")
def test_returns_the_same_samples_every_call(source: CalibrationSource) -> None:
    first = source.calibration()
    assert source.calibration().samples == first.samples
    assert first.samples, "the shipped samples are not empty"


@pytest.mark.req("TER-REC-004")
def test_a_never_judged_detector_reports_zero(source: CalibrationSource) -> None:
    unjudged = source.calibration().of("no_such_detector")
    assert (unjudged.judged, unjudged.samples) == (0, 0)


@pytest.mark.req("TER-REC-004")
def test_sums_each_detector_over_the_samples(source: CalibrationSource) -> None:
    calibration = source.calibration()
    for detector in calibration.detectors():
        mine = [
            s.detectors[detector]
            for s in calibration.samples
            if detector in s.detectors
        ]
        total = calibration.of(detector)
        assert total.samples == len(mine)
        assert total.judged == sum(c.judged for c in mine)


# --- the JSON adapter refuses bad documents with its own error --------------


def _write(tmp_path: Path, doc: object) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


GOOD_SAMPLE = {
    "id": "s",
    "judged_on": "2026-10-10",
    "judges": 1,
    "source": "a test",
    "detectors": {"d": {"waste": 1, "not_waste": 0, "unsure": 0}},
}


@pytest.mark.req("TER-REC-004")
@pytest.mark.parametrize(
    "doc",
    [
        [],
        {"schema": "ter.calibration/2", "samples": []},
        {"schema": "ter.calibration/1"},
        {"schema": "ter.calibration/1", "samples": {}, "extra": 1},
        {"schema": "ter.calibration/1", "samples": [1]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "id": 3}]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "source": " "}]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "judged_on": 5}]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "judged_on": "x"}]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "judges": True}]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "judges": 0}]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "detectors": []}]},
        {
            "schema": "ter.calibration/1",
            "samples": [
                {
                    **GOOD_SAMPLE,
                    "detectors": {"d": {"waste": -1, "not_waste": 0, "unsure": 0}},
                }
            ],
        },
        {
            "schema": "ter.calibration/1",
            "samples": [{**GOOD_SAMPLE, "detectors": {"d": {"waste": 1}}}],
        },
        {"schema": "ter.calibration/1", "samples": [GOOD_SAMPLE, GOOD_SAMPLE]},
        {"schema": "ter.calibration/1", "samples": [{**GOOD_SAMPLE, "id": " "}]},
    ],
)
def test_json_refuses_a_malformed_document(tmp_path: Path, doc: object) -> None:
    with pytest.raises(CalibrationError):
        JsonCalibration(_write(tmp_path, doc))


@pytest.mark.req("TER-REC-004")
def test_json_refuses_text_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationError):
        JsonCalibration(path)


@pytest.mark.req("TER-REC-004")
def test_json_reads_a_file_and_orders_samples_by_date(tmp_path: Path) -> None:
    later = {**GOOD_SAMPLE, "id": "later", "judged_on": "2026-11-01"}
    source = JsonCalibration(
        _write(
            tmp_path, {"schema": "ter.calibration/1", "samples": [later, GOOD_SAMPLE]}
        )
    )
    assert source.name.endswith("calibration.json")
    assert [s.id for s in source.calibration().samples] == ["s", "later"]
    assert source.calibration().of("d").waste == 2


@pytest.mark.req("TER-REC-004")
def test_in_memory_refuses_duplicate_sample_ids() -> None:
    sample = JudgedSample("a", date(2026, 1, 1), 1, {"d": JudgedCounts(1)})
    with pytest.raises(CalibrationError, match="duplicate"):
        InMemoryCalibration([sample, sample])
