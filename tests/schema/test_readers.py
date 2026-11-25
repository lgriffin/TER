"""Readers accept what their schema allows, and writers keep to it.

Hypothesis generates documents from each JSON Schema (hypothesis-jsonschema)
and hands them to TER's reader for that format (TER-TST-003). A reader either
reads the document, or rejects it with its own documented error for a rule
the schema cannot state (unique policy ids, limits in order); it never fails
with an unhandled ``KeyError``, ``TypeError`` or ``AttributeError``. What a
reader accepts, its writer writes back as a conforming document
(TER-TST-002); for the event and ledger formats, whose rules the schema
states in full, the round trip is exact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ter.adapters.driven.event_log import event_from_record, event_to_record
from ter.domain.events import (
    Actor,
    Event,
    EventId,
    EventKind,
    Provenance,
    TokenUsage,
    ToolCall,
    ToolKind,
)
from ter.domain.lean.control import (
    ControlLimits,
    ControlLimitsError,
    MeasuresDocument,
)
from ter.domain.ledger import LedgerError, LedgerRecord, Observation
from ter.domain.policy_set import PolicyError, PolicySet

from .conftest import assert_conforms, documents

FUZZ = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@pytest.mark.req("TER-TST-003")
@FUZZ
@given(documents("ter.event/0.5"))
def test_any_conforming_event_record_round_trips_exactly(record: Any) -> None:
    event = event_from_record(record)
    again = event_to_record(event)
    assert again == record
    assert event_from_record(again) == event


@pytest.mark.req("TER-TST-003")
@FUZZ
@given(documents("ter.intervention-ledger/1"))
def test_any_conforming_ledger_line_is_read_and_written_back(line: Any) -> None:
    if line["type"] == "record":
        record = LedgerRecord.from_mapping(line)
        assert {"schema": line["schema"], "type": "record", **record.as_dict()} == line
    else:
        observation = Observation.from_mapping(line)
        written = observation.as_dict()
        assert written["response"] == line["response"]
        assert written["tool_events"] == line["tool_events"]
        assert written["before"] == {
            k: float(v) for k, v in sorted(line["before"].items())
        }
        assert datetime.fromisoformat(written["observed_at"]) == datetime.fromisoformat(
            line["observed_at"]
        )


@pytest.mark.req("TER-TST-003", "TER-TST-002")
@FUZZ
@given(documents("ter.intervention-policies/1"))
def test_any_conforming_policy_file_is_read_or_refused_with_a_reason(doc: Any) -> None:
    try:
        policies = PolicySet.from_mapping(doc)
    except PolicyError as exc:
        # The one rule the schema cannot state.
        assert "unique" in str(exc), f"the schema should have refused this: {exc}"
        return
    assert_conforms("ter.intervention-policies/1", policies.as_dict())


@pytest.mark.req("TER-TST-003", "TER-TST-002")
@FUZZ
@given(documents("ter.control-limits/1"))
def test_any_conforming_limits_file_is_read_or_refused_with_a_reason(doc: Any) -> None:
    try:
        limits = ControlLimits.from_mapping(doc)
    except ControlLimitsError:
        return  # limit order and measure bounds are arithmetic, not shape
    assert_conforms("ter.control-limits/1", limits.as_dict())


@pytest.mark.req("TER-TST-003", "TER-TST-002")
@FUZZ
@given(documents("ter.control-measures/1"))
def test_any_conforming_measures_file_is_read_or_refused_with_a_reason(
    doc: Any,
) -> None:
    try:
        measures = MeasuresDocument.from_mapping(doc)
    except ControlLimitsError as exc:
        # Rows from several detector sets are written, but never charted.
        assert isinstance(doc["detectors"], list), str(exc)
        return
    assert_conforms("ter.control-measures/1", measures.as_dict())


# --- writers, from generated domain values ---------------------------------

_text = st.text(max_size=40)
_ids = st.text(alphabet="0123456789abcdef", min_size=16, max_size=16).map(EventId)
_json_scalars = st.none() | st.booleans() | st.integers() | _text
_arguments = st.dictionaries(
    _text,
    st.recursive(_json_scalars, lambda c: st.lists(c, max_size=3), max_leaves=6),
    max_size=4,
)
_events = st.builds(
    Event,
    id=_ids,
    session_id=_text,
    sequence=st.integers(min_value=0, max_value=10**6),
    kind=st.sampled_from(EventKind),
    actor=st.sampled_from(Actor),
    text=st.text(),
    provenance=st.builds(
        Provenance,
        source=_text,
        record_id=_text,
        lines=st.lists(st.integers(min_value=0, max_value=10**6), max_size=3).map(
            tuple
        ),
        block_index=st.none() | st.integers(min_value=0, max_value=50),
        fingerprint=st.none() | _text,
    ),
    timestamp=st.none() | st.datetimes(timezones=st.none() | st.just(UTC)),
    tool=st.none()
    | st.builds(
        ToolCall,
        native_name=_text,
        kind=st.sampled_from(ToolKind),
        call_id=st.none() | _text,
        arguments=_arguments,
    ),
    usage=st.none()
    | st.builds(
        TokenUsage,
        input_tokens=st.integers(min_value=0, max_value=10**7),
        output_tokens=st.integers(min_value=0, max_value=10**7),
        cache_creation_tokens=st.integers(min_value=0, max_value=10**7),
        cache_read_tokens=st.integers(min_value=0, max_value=10**7),
        model=st.none() | _text,
        cache_reported=st.booleans(),
    ),
    parent_id=st.none() | _ids,
)


@pytest.mark.req("TER-TST-002")
@FUZZ
@given(_events)
def test_every_event_the_codec_writes_conforms(event: Event) -> None:
    record = event_to_record(event)
    assert_conforms("ter.event/0.5", record)
    assert event_from_record(record) == event


@pytest.mark.req("TER-TST-003", "TER-REC-004")
@FUZZ
@given(documents("ter.calibration/1"))
def test_any_conforming_calibration_reads_or_is_refused_with_its_own_error(
    doc: Any,
) -> None:
    from ter.adapters.driven.calibration import parse_calibration
    from ter.domain.calibration import CalibrationError

    try:
        calibration = parse_calibration(doc)
    except CalibrationError:
        return  # a rule the schema cannot state: duplicate sample ids
    for detector in calibration.detectors():
        assert calibration.of(detector).judged >= 0
