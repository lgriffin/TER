"""Control charts (point 201): XmR limits, detection rules, tuning and firing.

Every rule has positive, negative and boundary cases. Limits for the rule
tests are set by hand (centre 10, sigma 1, limits 7 and 13) so each case
reads as a picture of the chart.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from ter4_lean_builder import FAIL, PASS, Script

from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain.lean import build_a3, explain
from ter.domain.lean import control as control_module
from ter.domain.lean.control import (
    CONTROL_LIMITS_SCHEMA,
    CONTROL_MEASURES,
    MIN_BASELINE,
    ControlLimits,
    ControlLimitsError,
    ControlRule,
    LimitMethod,
    MeasureLimits,
    MeasuresDocument,
    NaturalLimits,
    SessionMeasures,
    Side,
    Tuning,
    compute_limits,
    control_chart,
    control_measure,
    control_report,
    detector_fingerprint,
    measure_session,
    natural_limits,
    order_sessions,
    place,
    pseudonymise,
    session_control,
)
from ter.domain.lean.model import LeanWaste
from ter.domain.lean.wip import WipKind
from ter.domain.lean.scorecard import scorecard_dimensions, software_value_efficiency

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def hand_limits(
    key: str = "wip_peak",
    *,
    centre: float = 10.0,
    sigma: float = 1.0,
    ucl: float | None = 13.0,
    lcl: float | None = 7.0,
    tuning: Tuning | None = None,
    rules: frozenset[ControlRule] = frozenset(ControlRule),
    enabled: bool = True,
) -> MeasureLimits:
    natural = NaturalLimits(
        centre,
        ucl,
        lcl,
        sigma,
        sigma * 1.128,
        sigma * 1.128 * 3.268,
        30,
        LimitMethod.AVERAGE_MOVING_RANGE,
    )
    return MeasureLimits(control_measure(key), natural, tuning, rules, enabled)


def chart(values: list[float], limits: MeasureLimits | None = None) -> Any:
    limits = limits or hand_limits()
    return control_chart(limits, [(f"s{i}", v) for i, v in enumerate(values)])


def rules_of(values: list[float], limits: MeasureLimits | None = None) -> list[str]:
    return [f"{s.rule.value}@{s.session_id}" for s in chart(values, limits).signals]


def row(
    i: int, detectors: str = "fp", started: datetime | None = None, **values: float
) -> SessionMeasures:
    return SessionMeasures(
        f"s{i:02d}",
        T0 + timedelta(hours=i) if started is None else started,
        dict(values),
        detectors,
    )


# --- natural limits -----------------------------------------------------------


@pytest.mark.req("TER-SPC-001")
class TestNaturalLimits:
    def test_average_moving_range_limits(self) -> None:
        # Wheeler's arithmetic on a hand-checked series: mean 10, moving
        # ranges 2,2,2,2,2,2,2 so mR-bar 2.
        values = [9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0]
        lim = natural_limits(values)
        assert lim.centre == pytest.approx(10.0)
        assert lim.moving_range == pytest.approx(2.0)
        assert lim.ucl == pytest.approx(10 + 2.66 * 2)
        assert lim.lcl == pytest.approx(10 - 2.66 * 2)
        assert lim.sigma == pytest.approx(2 / 1.128)
        assert lim.mr_ucl == pytest.approx(3.268 * 2)
        assert lim.sessions == 8

    def test_median_moving_range_resists_one_wild_session(self) -> None:
        values = [10.0, 11.0, 10.0, 11.0, 10.0, 60.0, 10.0, 11.0, 10.0]
        average = natural_limits(values)
        median = natural_limits(values, LimitMethod.MEDIAN_MOVING_RANGE)
        assert median.moving_range == pytest.approx(1.0)
        assert median.ucl == pytest.approx(median.centre + 3.145 * 1.0)
        assert median.mr_ucl == pytest.approx(3.865)
        assert average.ucl is not None and median.ucl is not None
        assert median.ucl < average.ucl

    @pytest.mark.req("TER-SPC-014")
    def test_median_of_zero_falls_back_to_the_average_moving_range(self) -> None:
        # Mostly 0, as confident findings were on the owner's corpus.
        counts = [0.0] * 10 + [3.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        median = natural_limits(counts, LimitMethod.MEDIAN_MOVING_RANGE)
        average = natural_limits(counts)
        assert median.method is LimitMethod.AVERAGE_MOVING_RANGE
        assert median == average
        assert median.ucl is not None and median.ucl > median.centre
        assert median.as_dict()["method"] == "average_moving_range"

    @pytest.mark.req("TER-SPC-014")
    def test_median_with_spread_keeps_the_median_method(self) -> None:
        values = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0]
        lim = natural_limits(values, LimitMethod.MEDIAN_MOVING_RANGE)
        assert lim.method is LimitMethod.MEDIAN_MOVING_RANGE

    def test_limit_past_the_boundary_is_no_limit(self) -> None:
        counts = [0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0]
        lim = natural_limits(counts, lower_bound=0.0)
        assert lim.lcl is None and lim.ucl is not None
        ratios = [0.9, 1.0, 0.8, 1.0, 0.95, 1.0, 0.85, 1.0]
        lim = natural_limits(ratios, lower_bound=0.0, upper_bound=1.0)
        assert lim.ucl is None and lim.lcl is not None

    def test_constant_process_has_zero_width_limits(self) -> None:
        lim = natural_limits([3.0] * 8)
        assert lim.sigma == 0 and lim.ucl == 3.0
        # Any change from a perfectly constant process is beyond the limits.
        limits = MeasureLimits(control_measure("wip_peak"), lim)
        assert [s.rule for s in control_chart(limits, [("a", 4.0)]).signals] == [
            ControlRule.BEYOND_LIMITS
        ]


@pytest.mark.req("TER-SPC-002")
class TestBaselineSize:
    def test_too_few_sessions_compute_no_limits(self) -> None:
        with pytest.raises(ControlLimitsError, match="too few"):
            natural_limits([1.0] * (MIN_BASELINE - 1))

    def test_measure_with_too_few_known_values_is_left_out(self) -> None:
        rows = [row(i, wip_peak=float(i % 3)) for i in range(10)]
        rows += [row(10 + i, rework_cycles=1.0) for i in range(3)]
        limits = compute_limits(rows)
        assert limits.get("wip_peak") is not None
        assert limits.get("rework_cycles") is None

    @pytest.mark.req("TER-SPC-012")
    def test_boundary_eight_sessions_compute_provisional_limits(self) -> None:
        lim = natural_limits([float(i % 2) for i in range(MIN_BASELINE)])
        assert lim.provisional and lim.as_dict()["provisional"] is True

    @pytest.mark.req("TER-SPC-012")
    def test_twenty_sessions_settle(self) -> None:
        assert not natural_limits([float(i % 2) for i in range(20)]).provisional
        assert natural_limits([float(i % 2) for i in range(19)]).provisional


# --- detection rules ------------------------------------------------------------


@pytest.mark.req("TER-SPC-003")
class TestBeyondLimits:
    def test_point_above_the_upper_limit(self) -> None:
        [signal] = chart([10.0, 13.5]).signals
        assert signal.rule is ControlRule.BEYOND_LIMITS
        assert signal.side is Side.ABOVE and signal.limit == 13.0
        assert signal.evidence == ("s1",)

    def test_point_below_the_lower_limit(self) -> None:
        [signal] = chart([6.5]).signals
        assert signal.side is Side.BELOW and signal.limit == 7.0

    def test_point_on_the_limit_is_inside(self) -> None:
        assert chart([13.0, 7.0]).signals == ()


@pytest.mark.req("TER-SPC-003")
class TestTwoOfThree:
    def test_two_of_three_beyond_two_sigma(self) -> None:
        assert rules_of([12.5, 10.0, 12.5]) == ["two_of_three@s2"]

    def test_the_pattern_cites_only_the_points_beyond(self) -> None:
        [signal] = chart([12.5, 10.0, 12.5]).signals
        assert signal.evidence == ("s0", "s2") and signal.limit == 12.0

    def test_opposite_sides_do_not_count(self) -> None:
        assert rules_of([12.5, 10.0, 7.5]) == []

    def test_boundary_exactly_two_sigma_is_not_beyond(self) -> None:
        assert rules_of([12.0, 10.0, 12.0]) == []

    def test_two_of_four_is_not_two_of_three(self) -> None:
        assert rules_of([12.5, 10.0, 10.0, 12.5]) == []


@pytest.mark.req("TER-SPC-003")
class TestFourOfFive:
    def test_four_of_five_beyond_one_sigma(self) -> None:
        assert rules_of([11.5, 11.5, 10.0, 11.5, 11.5]) == ["four_of_five@s4"]

    def test_three_of_five_is_not_enough(self) -> None:
        assert rules_of([11.5, 10.0, 10.0, 11.5, 11.5]) == []

    def test_boundary_exactly_one_sigma_is_not_beyond(self) -> None:
        assert rules_of([11.0, 11.0, 10.0, 11.0, 11.0]) == []


@pytest.mark.req("TER-SPC-003")
class TestRunOfEight:
    def test_eight_on_one_side(self) -> None:
        found = rules_of([10.2] * 8)
        assert found == ["run_of_eight@s7"]
        [signal] = chart([10.2] * 8).signals
        assert signal.evidence == tuple(f"s{i}" for i in range(8))
        assert signal.limit == 10.0

    def test_seven_is_not_a_run(self) -> None:
        assert rules_of([10.2] * 7) == []

    def test_a_point_on_the_centre_line_breaks_the_run(self) -> None:
        assert rules_of([10.2] * 4 + [10.0] + [10.2] * 4) == []

    def test_every_further_point_of_a_run_signals(self) -> None:
        assert rules_of([9.8] * 9) == ["run_of_eight@s7", "run_of_eight@s8"]


@pytest.mark.req("TER-SPC-003")
class TestRuleSelection:
    def test_switched_off_rule_is_not_tested(self) -> None:
        limits = hand_limits(rules=frozenset({ControlRule.BEYOND_LIMITS}))
        assert rules_of([10.2] * 8, limits) == []
        assert rules_of([14.0], limits) == ["beyond_limits@s0"]

    def test_switched_off_measure_is_charted_without_signals(self) -> None:
        result = chart([20.0, 20.0], hand_limits(enabled=False))
        assert len(result.points) == 2 and result.signals == ()

    def test_zone_and_run_rules_skip_a_side_cut_off_by_the_boundary(self) -> None:
        # Counts piled at 0: no lower limit, so no "better" signals there.
        limits = hand_limits(centre=0.5, sigma=0.4, ucl=1.7, lcl=None)
        assert rules_of([0.0] * 9, limits) == []
        assert rules_of([2.0], limits) == ["beyond_limits@s0"]

    def test_points_carry_moving_ranges_and_sigmas(self) -> None:
        points = chart([10.0, 12.0, 9.0]).points
        assert [p.moving_range for p in points] == [None, 2.0, 3.0]
        assert [p.sigmas for p in points] == [0.0, 2.0, -1.0]


# --- tuning ---------------------------------------------------------------------


@pytest.mark.req("TER-SPC-004")
class TestTuning:
    def test_tuned_limit_replaces_the_natural_one_for_beyond_limits(self) -> None:
        limits = hand_limits(tuning=Tuning(11.0, None, "team target"))
        assert limits.ucl == 11.0 and limits.lcl == 7.0
        [signal] = chart([11.5], limits).signals
        assert signal.rule is ControlRule.BEYOND_LIMITS
        assert signal.tuned and signal.limit == 11.0

    def test_natural_side_still_applies_when_only_one_side_is_tuned(self) -> None:
        limits = hand_limits(tuning=Tuning(11.0, None, "team target"))
        [signal] = chart([6.0], limits).signals
        assert not signal.tuned and signal.limit == 7.0

    def test_zone_rules_keep_the_natural_centre_and_sigma(self) -> None:
        loose = hand_limits(tuning=Tuning(20.0, None, "loosened for a spike"))
        assert rules_of([12.5, 10.0, 12.5], loose) == ["two_of_three@s2"]
        assert rules_of([14.0], loose) == []

    def test_document_keeps_natural_and_tuned_limits(self) -> None:
        doc = ControlLimits(
            (hand_limits(tuning=Tuning(11.0, 8.0, "why")),),
            LimitMethod.AVERAGE_MOVING_RANGE,
            "fp",
            30,
        ).as_dict()
        entry = doc["measures"]["wip_peak"]  # type: ignore[index]
        assert entry["natural"]["ucl"] == 13.0
        assert entry["tuned"] == {"ucl": 11.0, "lcl": 8.0, "reason": "why"}

    def test_place_one_session(self) -> None:
        limits = hand_limits(tuning=Tuning(12.0, None, "target"))
        inside = place(limits, "x", 11.0)
        assert inside.in_control and inside.sigmas == 1.0
        outside = place(limits, "x", 12.5)
        assert outside.signal is not None and outside.signal.tuned
        assert outside.as_dict()["ucl"] == 12.0
        assert place(hand_limits(enabled=False), "x", 99.0).in_control


# --- the limits document -----------------------------------------------------------


def _document(**entry: Any) -> dict[str, Any]:
    base: dict[str, Any] = json.loads(
        json.dumps(
            ControlLimits(
                (hand_limits(),), LimitMethod.AVERAGE_MOVING_RANGE, "fp", 30
            ).as_dict()
        )
    )
    base["measures"]["wip_peak"].update(entry)
    return base


@pytest.mark.req("TER-SPC-005")
class TestLimitsDocument:
    def test_round_trip(self) -> None:
        limits = ControlLimits(
            (
                hand_limits(
                    tuning=Tuning(12.0, None, "target"),
                    rules=frozenset({ControlRule.BEYOND_LIMITS}),
                ),
                hand_limits(
                    "flow_efficiency_tokens",
                    centre=0.8,
                    sigma=0.05,
                    ucl=0.95,
                    lcl=0.65,
                    enabled=False,
                ),
            ),
            LimitMethod.MEDIAN_MOVING_RANGE,
            "fp",
            30,
            "2026-10-10",
        )
        again = ControlLimits.from_mapping(json.loads(json.dumps(limits.as_dict())))
        assert again.as_dict() == limits.as_dict()
        assert [m.measure.key for m in again.measures] == [
            "flow_efficiency_tokens",
            "wip_peak",
        ]

    def test_tuning_needs_a_reason(self) -> None:
        with pytest.raises(ControlLimitsError, match="reason"):
            ControlLimits.from_mapping(_document(tuned={"ucl": 12, "reason": " "}))

    def test_tuned_lower_limit_must_be_below_upper(self) -> None:
        with pytest.raises(ControlLimitsError, match="below"):
            ControlLimits.from_mapping(
                _document(tuned={"ucl": 9, "lcl": 9, "reason": "x"})
            )

    def test_tuning_must_set_a_limit(self) -> None:
        with pytest.raises(ControlLimitsError, match="neither"):
            ControlLimits.from_mapping(_document(tuned={"reason": "x"}))

    def test_tuned_ratio_must_stay_within_zero_and_one(self) -> None:
        doc = _document()
        entry = doc["measures"].pop("wip_peak")
        entry["tuned"] = {"ucl": 1.5, "reason": "x"}
        doc["measures"]["avoidable_share"] = entry
        with pytest.raises(ControlLimitsError, match="0.0 to 1.0"):
            ControlLimits.from_mapping(doc)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"rules": ["beyond_limits", "nine_in_a_row"]}, "unknown value"),
            ({"enabled": "yes"}, "enabled"),
            ({"natural": None}, "natural limits"),
            ({"natural": {"centre": "x"}}, "centre"),
        ],
    )
    def test_bad_entries_are_rejected(
        self, change: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ControlLimitsError, match=message):
            ControlLimits.from_mapping(_document(**change))

    def test_unknown_measure_and_schema_are_rejected(self) -> None:
        doc = _document()
        doc["measures"]["lines_of_code"] = doc["measures"]["wip_peak"]
        with pytest.raises(ControlLimitsError, match="unknown control measure"):
            ControlLimits.from_mapping(doc)
        with pytest.raises(ControlLimitsError, match="schema"):
            ControlLimits.from_mapping({**_document(), "schema": "other/1"})
        assert _document()["schema"] == CONTROL_LIMITS_SCHEMA


# --- firing ------------------------------------------------------------------------


@pytest.mark.req("TER-SPC-006")
class TestFiring:
    def test_unfavourable_structural_signal_fires(self) -> None:
        [signal] = chart([14.0]).signals
        assert signal.unfavourable and signal.fires
        assert signal.id == "control.wip_peak.beyond_limits"

    def test_favourable_signal_never_fires(self) -> None:
        [signal] = chart([6.0]).signals
        assert not signal.unfavourable and not signal.fires

    def test_direction_follows_the_measure(self) -> None:
        flow = hand_limits(
            "flow_efficiency_tokens", centre=0.8, sigma=0.05, ucl=0.95, lcl=0.65
        )
        [low] = chart([0.5], flow).signals
        assert low.unfavourable and low.fires
        [high] = chart([0.99], flow).signals
        assert not high.unfavourable

    @pytest.mark.parametrize("key", ["generated_tokens", "agent_seconds"])
    def test_token_and_time_totals_never_fire(self, key: str) -> None:
        [signal] = chart([14.0], hand_limits(key)).signals
        assert signal.unfavourable and not signal.fires

    def test_every_measure_declares_whether_it_fires(self) -> None:
        firing = {m.key for m in CONTROL_MEASURES if m.fires}
        assert "generated_tokens" not in firing and "avoidable_share" in firing
        assert {"uncertain_findings", "unverified_waste_share"} <= firing


# --- detector sets ------------------------------------------------------------------


@pytest.mark.req("TER-SPC-007")
class TestDetectorSets:
    @pytest.mark.req("TER-SPC-013")
    def test_mixed_detector_sets_are_refused(self) -> None:
        rows = [row(i, "a" if i % 2 else "b", wip_peak=1.0) for i in range(10)]
        with pytest.raises(ControlLimitsError, match="different detector sets"):
            compute_limits(rows)
        with pytest.raises(ControlLimitsError, match="different detector sets"):
            control_report(rows, compute_limits(rows[::2]))

    def test_limits_from_another_detector_set_are_stale(self) -> None:
        limits = compute_limits(
            [row(i, "old", wip_peak=float(i % 3)) for i in range(9)]
        )
        fresh = [row(i, "new", wip_peak=1.0) for i in range(3)]
        report = control_report(fresh, limits)
        assert report.stale and report.as_dict()["stale"] is True
        same = control_report([row(0, "old", wip_peak=1.0)], limits)
        assert not same.stale

    def test_fingerprint_follows_the_detectors_and_their_rules(self) -> None:
        s = Script()
        s.prompt("fix it")
        s.bash("pytest", FAIL)
        a = explain(s.events, RegexTokenizer())
        b = explain(s.events, RegexTokenizer())
        assert detector_fingerprint(a) == detector_fingerprint(b)
        assert len(detector_fingerprint(a)) == 12


@pytest.mark.req("TER-SPC-008")
class TestRecompute:
    def test_keep_carries_tuning_rules_and_switches(self) -> None:
        rows = [
            row(i, wip_peak=float(i % 4), rework_cycles=float(i % 2)) for i in range(12)
        ]
        first = compute_limits(rows)
        wip = first.get("wip_peak")
        assert wip is not None
        tuned = ControlLimits(
            (
                MeasureLimits(
                    wip.measure,
                    wip.natural,
                    Tuning(2.5, None, "agreed in retro"),
                    frozenset({ControlRule.BEYOND_LIMITS}),
                ),
                MeasureLimits(
                    control_measure("rework_cycles"), wip.natural, enabled=False
                ),
            ),
            first.method,
            first.detectors,
            first.sessions,
        )
        more = rows + [row(20 + i, wip_peak=5.0, rework_cycles=0.0) for i in range(5)]
        again = compute_limits(more, keep=tuned, computed_on="2026-10-11")
        new_wip = again.get("wip_peak")
        assert new_wip is not None and new_wip.natural.sessions == 17
        assert new_wip.tuning == Tuning(2.5, None, "agreed in retro")
        assert new_wip.rules == frozenset({ControlRule.BEYOND_LIMITS})
        rework = again.get("rework_cycles")
        assert rework is not None and not rework.enabled
        assert again.computed_on == "2026-10-11"

    def test_without_keep_every_rule_is_on(self) -> None:
        rows = [row(i, wip_peak=float(i % 4)) for i in range(9)]
        wip = compute_limits(rows).get("wip_peak")
        assert wip is not None and wip.rules == frozenset(ControlRule) and wip.enabled


# --- measures -------------------------------------------------------------------------


def _session() -> Script:
    s = Script()
    s.prompt("fix the failing test")
    s.read("src/app.py", "def f(): return 1")
    s.edit("src/app.py", "return 1", "return 2")
    s.bash("pytest", FAIL)
    s.edit("src/app.py", "return 2", "return 3")
    s.bash("pytest", PASS)
    s.say("Fixed.")
    return s


@pytest.mark.req("TER-SPC-009")
class TestMeasures:
    def test_measures_equal_the_scorecard_dimensions(self) -> None:
        analysis = explain(_session().events, RegexTokenizer())
        measured = measure_session(analysis, T0)
        dims = scorecard_dimensions(
            analysis, software_value_efficiency(analysis.scorecard, None), None
        )
        shared = {m.key: m.value for d in dims for m in d.measures}
        for key, value in measured.values.items():
            if key in shared and value is not None:
                assert value == pytest.approx(float(shared[key])), key  # type: ignore[arg-type]
        assert measured.values["confident_findings"] == analysis.scorecard.findings
        assert set(measured.values) == {m.key for m in CONTROL_MEASURES}

    def test_measures_file_holds_no_content(self) -> None:
        analysis = explain(_session().events, RegexTokenizer())
        text = json.dumps(MeasuresDocument((measure_session(analysis, T0),)).as_dict())
        for secret in ("fix the failing test", "src/app.py", "return 2", "Fixed."):
            assert secret not in text

    def test_pseudonymise_hashes_every_session_id(self) -> None:
        rows = pseudonymise([row(1), row(2)])
        assert all(r.session_id.startswith("s-") for r in rows)
        assert len({r.session_id for r in rows}) == 2
        assert rows == pseudonymise([row(1), row(2)])

    def test_measures_document_round_trip(self) -> None:
        undated = SessionMeasures("s01", None, {"wip_peak": None}, "fp")
        rows = (row(2, wip_peak=3.0), undated)
        doc = MeasuresDocument(rows).as_dict()
        again = MeasuresDocument.from_mapping(json.loads(json.dumps(doc)))
        assert [r.session_id for r in again.rows] == ["s02", "s01"]
        assert again.rows[0].value("wip_peak") == 3.0
        assert again.rows[1].started_at is None

    @pytest.mark.parametrize(
        ("doc", "message"),
        [
            ({"schema": "x"}, "schema"),
            (
                {"schema": "ter.control-measures/1", "detectors": ["a", "b"]},
                "one detector",
            ),
            (
                {
                    "schema": "ter.control-measures/1",
                    "detectors": "a",
                    "sessions": [{"session": "s", "values": {"wip_peak": "3"}}],
                },
                "number or null",
            ),
            (
                {
                    "schema": "ter.control-measures/1",
                    "detectors": "a",
                    "sessions": [
                        {"session": "s", "started_at": "yesterday", "values": {}}
                    ],
                },
                "started_at",
            ),
        ],
    )
    def test_bad_measures_documents_are_rejected(
        self, doc: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ControlLimitsError, match=message):
            MeasuresDocument.from_mapping(doc)

    def test_process_order_is_start_time_then_id(self) -> None:
        naive = datetime(2026, 10, 1, 8, 0)
        rows = [
            row(3, started=T0),
            SessionMeasures("undated", None, {}, "fp"),
            row(1, started=naive),
            row(2, started=T0),
        ]
        assert [r.session_id for r in order_sessions(rows)] == [
            "s01",
            "s02",
            "s03",
            "undated",
        ]


def test_outcome_never_reaches_control_measures() -> None:
    # The import-linter contract that keeps behaviour blind to the outcome
    # (point 5) covers the control module.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split('id = "behaviour-blind-to-outcome"', 1)[1].split("[[", 1)[0]
    assert '"ter.domain.lean.control"' in block


@pytest.mark.req("TER-SPC-010")
def test_page_shows_tuning_switches_and_stale_limits() -> None:
    from ter.adapters.driving.reports.control import render_control_html, xmr_chart

    rows = [
        row(i, "new", wip_peak=float(i % 3), generated_tokens=100.0 + i)
        for i in range(9)
    ]
    limits = compute_limits(rows)
    wip = limits.get("wip_peak")
    tokens = limits.get("generated_tokens")
    assert wip is not None and tokens is not None
    from dataclasses import replace

    tuned = replace(
        limits,
        detectors="old",
        measures=(
            replace(
                wip,
                tuning=Tuning(1.5, None, "agreed <target>"),
                rules=frozenset({ControlRule.BEYOND_LIMITS}),
            ),
            replace(tokens, enabled=False),
        ),
    )
    report = control_report(rows, tuned)
    page = render_control_html(report, title="T & charts")
    assert "Stale limits" in page and "T &amp; charts" in page
    assert "agreed &lt;target&gt;" in page and "tuned UCL" in page
    assert "switched off" in page and "charted, never fires" in page
    assert "Rules off: Two of three" in page
    assert "fires</span>" in page
    svg = xmr_chart(report.charts[0], width=600)
    assert svg.startswith("<svg") and 'role="img"' in svg


# --- review findings (PR #71) ---------------------------------------------------


@pytest.mark.req("TER-SPC-005")
@pytest.mark.parametrize(
    "tuned",
    [{"lcl": 13.5, "reason": "x"}, {"ucl": 6.5, "reason": "x"}],
    ids=["tuned-lcl-above-natural-ucl", "tuned-ucl-below-natural-lcl"],
)
def test_one_sided_tuning_cannot_cross_the_other_natural_limit(
    tuned: dict[str, Any],
) -> None:
    with pytest.raises(ControlLimitsError, match="effective lower limit"):
        ControlLimits.from_mapping(_document(tuned=tuned))


@pytest.mark.req("TER-SPC-005")
@pytest.mark.parametrize(
    ("natural", "message"),
    [
        ({"ucl": 9.0}, "ucl must not be below the centre"),
        ({"lcl": 11.0}, "lcl must not be above the centre"),
        ({"centre": -1.0, "lcl": None}, "centre must be 0 or more"),
        ({"sigma": -0.5}, "spread"),
    ],
)
def test_edited_natural_limits_must_still_describe_a_process(
    natural: dict[str, Any], message: str
) -> None:
    doc = _document()
    doc["measures"]["wip_peak"]["natural"].update(natural)
    with pytest.raises(ControlLimitsError, match=message):
        ControlLimits.from_mapping(doc)


@pytest.mark.req("TER-SPC-005")
def test_natural_ratio_limits_stay_within_zero_and_one() -> None:
    doc = _document()
    entry = doc["measures"].pop("wip_peak")
    entry["natural"].update({"centre": 0.5, "ucl": 1.4, "lcl": 0.1})
    doc["measures"]["avoidable_share"] = entry
    with pytest.raises(ControlLimitsError, match="natural.ucl must be 0.0 to 1.0"):
        ControlLimits.from_mapping(doc)


@pytest.mark.req("TER-SPC-009")
def test_zero_spread_process_reports_json_safe_sigmas() -> None:
    rows = [row(i, rework_cycles=0.0) for i in range(8)]
    limits = compute_limits(rows)
    report = control_report([*rows, row(9, rework_cycles=2.0)], limits)
    text = json.dumps(report.as_dict(), allow_nan=False)
    [chart_] = report.charts
    assert chart_.points[-1].as_dict()["sigmas"] is None
    assert '"sigmas": null' in text
    rework = limits.get("rework_cycles")
    assert rework is not None
    assert place(rework, "x", 1.0).as_dict()["sigmas"] is None


@pytest.mark.req("TER-SPC-008")
def test_keep_holds_an_entry_whose_measure_lacks_sessions_now() -> None:
    rows = [
        row(i, wip_peak=float(i % 4), rework_cycles=float(i % 2)) for i in range(10)
    ]
    first = compute_limits(rows)
    rework = first.get("rework_cycles")
    assert rework is not None
    kept_entry = replace_measure(rework, Tuning(0.5, None, "agreed"))
    kept = ControlLimits((kept_entry,), first.method, first.detectors, first.sessions)
    fewer = [row(i, wip_peak=float(i % 4)) for i in range(10)]
    again = compute_limits(fewer, keep=kept)
    assert again.get("rework_cycles") == kept_entry
    assert compute_limits(fewer).get("rework_cycles") is None


def replace_measure(entry: MeasureLimits, tuning: Tuning) -> MeasureLimits:
    from dataclasses import replace

    return replace(entry, tuning=tuning, enabled=False)


@pytest.mark.req("TER-SPC-009")
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_measures_are_rejected(bad: float) -> None:
    doc = {
        "schema": "ter.control-measures/1",
        "detectors": "a",
        "sessions": [{"session": "s", "values": {"wip_peak": bad}}],
    }
    with pytest.raises(ControlLimitsError, match="finite"):
        MeasuresDocument.from_mapping(doc)


@pytest.mark.req("TER-SPC-010")
def test_a_point_with_several_signals_lists_every_rule_and_shows_the_worst() -> None:
    from ter.adapters.driving.reports.control import xmr_chart

    result = chart([12.5, 10.0, 13.5])
    assert {s.rule for s in result.signals} == {
        ControlRule.BEYOND_LIMITS,
        ControlRule.TWO_OF_THREE,
    }
    svg = xmr_chart(result)
    assert "Beyond limits, Two of three" in svg


# --- the A3 places a session against its limits (TER-SPC-011) ----------------


def _rework_session() -> Script:
    s = Script()
    s.prompt("fix the failing test")
    s.read("src/app.py", "def f(): return 1")
    s.bash("pytest", FAIL)
    s.edit("src/app.py", "return 1", "return 2")
    s.bash("pytest", FAIL)
    s.edit("src/app.py", "return 2", "return 3")
    s.bash("pytest", PASS)
    s.say("Fixed.")
    return s


def _session_limits(detectors: str, *entries: MeasureLimits) -> ControlLimits:
    return ControlLimits(entries, LimitMethod.AVERAGE_MOVING_RANGE, detectors, 30)


@pytest.mark.req("TER-SPC-011")
class TestSessionControl:
    def _analysis(self) -> Any:
        analysis = explain(_rework_session().events, RegexTokenizer())
        assert analysis.scorecard.rework_cycles == 1
        return analysis

    def test_a_measure_beyond_its_limit_links_the_findings_behind_it(self) -> None:
        analysis = self._analysis()
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits("rework_cycles", centre=0.0, sigma=0.2, ucl=0.5, lcl=None),
        )
        control = session_control(analysis, limits)
        (placed,) = control.placements
        assert placed.signal is not None and placed.signal.fires
        rework = [f.id for f in analysis.findings if f.waste is LeanWaste.REWORK]
        assert rework and control.behind["rework_cycles"] == tuple(rework)
        entry = control.as_dict()["measures"][0]  # type: ignore[index]
        assert entry["findings"] == rework
        assert entry["signal"]["id"] == "control.rework_cycles.beyond_limits"

    def test_a_measure_inside_its_limits_links_nothing(self) -> None:
        analysis = self._analysis()
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits("rework_cycles", centre=1.0, sigma=0.5, ucl=3.0, lcl=None),
        )
        control = session_control(analysis, limits)
        assert control.signals == () and control.behind == {}
        entry = control.as_dict()["measures"][0]  # type: ignore[index]
        assert entry["signal"] is None and "findings" not in entry

    def test_measures_the_limits_leave_out_are_not_placed(self) -> None:
        analysis = self._analysis()
        limits = _session_limits(
            detector_fingerprint(analysis), hand_limits("wip_peak", centre=1.0)
        )
        control = session_control(analysis, limits)
        assert [p.limits.measure.key for p in control.placements] == ["wip_peak"]

    def test_limits_from_another_detector_set_are_stale(self) -> None:
        analysis = self._analysis()
        control = session_control(analysis, _session_limits("other"))
        assert control.stale and control.as_dict()["stale"] is True

    def test_every_measure_names_the_findings_that_move_it(self) -> None:
        assert set(control_module._BEHIND) == {m.key for m in CONTROL_MEASURES}

    def test_the_a3_carries_process_control_only_with_limits(self) -> None:
        analysis = self._analysis()
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits("rework_cycles", centre=0.0, sigma=0.2, ucl=0.5, lcl=None),
        )
        plain = build_a3(analysis)
        assert plain.process_control is None
        assert "process_control" not in plain.as_dict()
        placed = build_a3(analysis, control=limits)
        assert (
            placed.as_dict()["process_control"]
            == plain.placed(limits).as_dict()["process_control"]
        )
        assert placed.as_dict()["process_control"]["firing"] == 1  # type: ignore[index]

    def test_the_a3_page_links_the_signal_to_its_findings(self) -> None:
        from ter.adapters.driving.reports.a3 import render_a3_html

        analysis = self._analysis()
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits("rework_cycles", centre=0.0, sigma=0.2, ucl=0.5, lcl=None),
        )
        assert 'id="s-control"' not in render_a3_html(build_a3(analysis))
        page = render_a3_html(build_a3(analysis, control=limits))
        assert 'id="s-control"' in page and "above limit, fires" in page
        rework = next(f for f in analysis.findings if f.waste is LeanWaste.REWORK)
        slug = "".join(c if c.isalnum() or c in "-_:." else "-" for c in rework.id)
        assert f'href="#f-{slug}"' in page and f'id="f-{slug}"' in page

    def test_a_favourable_signal_links_no_findings(self) -> None:
        analysis = self._analysis()
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits(
                "value_adding_share", centre=0.1, sigma=0.01, ucl=0.12, lcl=0.08
            ),
        )
        control = session_control(analysis, limits)
        (placed,) = control.placements
        assert placed.signal is not None and not placed.signal.unfavourable
        assert control.behind == {}
        assert "findings" not in control.as_dict()["measures"][0]  # type: ignore[index]

    def test_an_unchecked_measure_is_neither_in_nor_out(self) -> None:
        from ter.adapters.driving.reports.a3 import render_a3_html

        analysis = self._analysis()
        off = hand_limits(
            "rework_cycles", centre=0.0, sigma=0.2, ucl=0.5, lcl=None, enabled=False
        )
        control = session_control(
            analysis, _session_limits(detector_fingerprint(analysis), off)
        )
        (placed,) = control.placements
        assert not placed.checked and placed.signal is None
        assert control.as_dict()["measures"][0]["checked"] is False  # type: ignore[index]
        page = render_a3_html(
            build_a3(
                analysis, control=_session_limits(detector_fingerprint(analysis), off)
            )
        )
        assert "not checked, switched off" in page
        assert "No measure was checked against its limits." in page

    def test_edits_open_at_the_end_link_the_findings_that_cite_them(self) -> None:
        s = Script()
        s.prompt("add a flag")
        s.read("src/app.py", "def f(): return 1")
        s.edit("src/app.py", "return 1", "return 2")
        s.say("Done.")
        analysis = explain(s.events, RegexTokenizer())
        still_open = set(analysis.wip.still_open(WipKind.EDITS))
        assert still_open
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits(
                "unvalidated_edits_at_end", centre=0.0, sigma=0.1, ucl=0.3, lcl=None
            ),
        )
        control = session_control(analysis, limits)
        linked = control.behind["unvalidated_edits_at_end"]
        assert linked
        for fid in linked:
            assert still_open.intersection(analysis.finding(fid).evidence)

    def test_failures_open_at_the_end_link_only_the_findings_that_cite_them(
        self,
    ) -> None:
        s = Script()
        s.prompt("fix the failing test")
        s.read("src/app.py", "def f(): return 1")
        s.edit("src/app.py", "return 1", "return 2")
        s.bash("pytest", FAIL)
        s.say("I could not fix it.")
        analysis = explain(s.events, RegexTokenizer())
        still_open = set(analysis.wip.still_open(WipKind.FAILURES))
        assert still_open
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits(
                "unresolved_failures_at_end", centre=0.0, sigma=0.1, ucl=0.3, lcl=None
            ),
        )
        linked = session_control(analysis, limits).behind["unresolved_failures_at_end"]
        assert linked
        unrelated = [
            f.id for f in analysis.findings if not still_open.intersection(f.evidence)
        ]
        assert not set(linked).intersection(unrelated)

    def test_a_check_that_failed_twice_links_the_finding_on_its_last_run(
        self,
    ) -> None:
        s = Script()
        s.prompt("fix the failing test")
        s.read("src/app.py", "def f(): return 1")
        s.edit("src/app.py", "return 1", "return 2")
        s.bash("pytest", FAIL)
        s.bash("pytest", FAIL)
        s.say("I could not fix it.")
        analysis = explain(s.events, RegexTokenizer())
        responded = [
            f.id
            for f in analysis.findings
            if f.title == "Responded after a failing check"
        ]
        assert responded
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits(
                "unresolved_failures_at_end", centre=0.0, sigma=0.1, ucl=0.3, lcl=None
            ),
        )
        linked = session_control(analysis, limits).behind["unresolved_failures_at_end"]
        assert set(responded) <= set(linked)

    def test_the_wip_peak_links_findings_up_to_the_peak(self) -> None:
        analysis = self._analysis()
        peak = analysis.wip.peak
        assert peak is not None
        order = {st.event_id: st.index for st in analysis.steps}
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits("wip_peak", centre=0.0, sigma=0.1, ucl=0.3, lcl=None),
        )
        linked = session_control(analysis, limits).behind["wip_peak"]
        expected = [
            f.id
            for f in analysis.findings
            if any(order[e] <= order[peak.event_id] for e in f.evidence)
        ]
        assert list(linked) == expected

    def test_folded_rows_print(self) -> None:
        from ter.adapters.driving.reports.a3 import render_a3_html

        analysis = self._analysis()
        limits = _session_limits(
            detector_fingerprint(analysis),
            hand_limits("wip_peak", centre=3.0, ucl=100.0, lcl=None),
        )
        page = render_a3_html(build_a3(analysis, control=limits))
        assert '<details class="control">' in page
        assert "details.control::details-content{content-visibility:visible" in page
