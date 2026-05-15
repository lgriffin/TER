"""Tests for consistency.py — cross-session version skew detection."""

import time
import pytest

from ter_calculator.models import (
    ConsistencyAction,
    ConsistencyMode,
    FragmentVersion,
    InvalidationEvent,
    VersionSkew,
)
from ter_calculator.consistency import (
    ConsistencyCoordinator,
    _classify_severity,
)


@pytest.fixture
def coordinator():
    return ConsistencyCoordinator()


class TestSeverityClassification:
    def test_low_severity(self):
        assert _classify_severity(2, 1) == "LOW"

    def test_medium_severity(self):
        assert _classify_severity(3, 1) == "MEDIUM"

    def test_high_severity_many_sessions(self):
        assert _classify_severity(5, 1) == "HIGH"

    def test_high_severity_large_gap(self):
        assert _classify_severity(2, 3) == "HIGH"


class TestRegisterSessionFragments:
    def test_register(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1", "f2"])
        assert "s1" in coordinator._session_fragments
        assert len(coordinator._session_fragments["s1"]) == 2

    def test_register_multiple_sessions(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1", "f2"])
        coordinator.register_session_fragments("s2", ["f2", "f3"])
        assert len(coordinator._session_fragments) == 2


class TestCheckConsistency:
    def test_no_skew_single_session(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        skews = coordinator.check_consistency("s1")
        assert len(skews) == 0

    def test_no_skew_same_version(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        coordinator.register_session_fragments("s2", ["f1"])
        skews = coordinator.check_consistency("s1")
        assert len(skews) == 0

    def test_detects_skew(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        coordinator.on_fragment_updated("f1", "hash_v2", time.time())
        coordinator.register_session_fragments("s2", ["f1"])
        skews = coordinator.check_consistency("s1")
        assert len(skews) == 1
        assert skews[0].fragment_id == "f1"
        assert len(skews[0].sessions_involved) == 2

    def test_multiple_skews(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1", "f2"])
        coordinator.on_fragment_updated("f1", "new1", time.time())
        coordinator.on_fragment_updated("f2", "new2", time.time())
        coordinator.register_session_fragments("s2", ["f1", "f2"])
        skews = coordinator.check_consistency("s1")
        assert len(skews) == 2


class TestResolveSkew:
    def test_strict_mode_blocks(self, coordinator):
        skew = VersionSkew(
            fragment_id="f1",
            sessions_involved=["s1", "s2"],
            versions_seen={"s1": 1, "s2": 2},
            severity="LOW",
        )
        action = coordinator.resolve_skew(skew, ConsistencyMode.STRICT)
        assert action.block is True
        assert "f1" in action.refresh_fragment_ids

    def test_relaxed_mode_warns(self, coordinator):
        skew = VersionSkew(
            fragment_id="f1",
            sessions_involved=["s1", "s2"],
            versions_seen={"s1": 1, "s2": 2},
            severity="LOW",
        )
        action = coordinator.resolve_skew(skew, ConsistencyMode.RELAXED)
        assert action.block is False
        assert "warning" in action.message.lower()


class TestOnFragmentUpdated:
    def test_creates_invalidation_events(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        events = coordinator.on_fragment_updated("f1", "new_hash", time.time())
        assert len(events) == 1
        assert events[0].fragment_id == "f1"

    def test_increments_version(self, coordinator):
        coordinator.on_fragment_updated("f1", "hash1", time.time())
        assert coordinator._fragment_versions["f1"].version == 1
        coordinator.on_fragment_updated("f1", "hash2", time.time())
        assert coordinator._fragment_versions["f1"].version == 2

    def test_no_events_for_unregistered(self, coordinator):
        events = coordinator.on_fragment_updated("f1", "hash1", time.time())
        assert len(events) == 0

    def test_events_for_multiple_sessions(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        coordinator.register_session_fragments("s2", ["f1"])
        events = coordinator.on_fragment_updated("f1", "new", time.time())
        assert len(events) == 2


class TestGetStaleFragments:
    def test_detects_stale(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        coordinator._fragment_versions["f1"] = FragmentVersion(
            fragment_id="f1",
            version=1,
            content_hash="old",
            timestamp=1.0,
        )
        stale = coordinator.get_stale_fragments("s1", max_age_seconds=1)
        assert "f1" in stale

    def test_no_stale_when_fresh(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        coordinator._fragment_versions["f1"] = FragmentVersion(
            fragment_id="f1",
            version=1,
            content_hash="current",
            timestamp=time.time(),
        )
        stale = coordinator.get_stale_fragments("s1", max_age_seconds=9999)
        assert len(stale) == 0


class TestUpdateSessionVersion:
    def test_updates_version(self, coordinator):
        coordinator.register_session_fragments("s1", ["f1"])
        coordinator.on_fragment_updated("f1", "v2", time.time())
        coordinator.update_session_version("s1", "f1")
        assert coordinator._session_versions["s1"]["f1"] == 1
