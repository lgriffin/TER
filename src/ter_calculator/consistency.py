"""Cross-session consistency coordination.

Implements the Consistency Coordinator from the Token Aware Microprompt
Orchestrator patent: monitors for version skew when multiple sessions
reference different versions of the same logical fragment, and enforces
consistency policies (strict blocking or relaxed annotation).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Sequence

from .context_graph import ContextGraph
from .fragment_store import FragmentStore
from .models import (
    ConsistencyAction,
    ConsistencyMode,
    FragmentVersion,
    InvalidationEvent,
    VersionSkew,
)


def _classify_severity(
    num_sessions: int, max_version_gap: int
) -> str:
    if max_version_gap > 2 or num_sessions > 4:
        return "HIGH"
    if num_sessions >= 3:
        return "MEDIUM"
    return "LOW"


class ConsistencyCoordinator:
    """Tracks fragment versions across sessions and detects skew."""

    def __init__(self) -> None:
        self._session_fragments: dict[str, set[str]] = defaultdict(set)
        self._fragment_versions: dict[str, FragmentVersion] = {}
        self._session_versions: dict[str, dict[str, int]] = defaultdict(dict)

    def register_session_fragments(
        self,
        session_id: str,
        fragment_ids: list[str],
    ) -> None:
        self._session_fragments[session_id] = set(fragment_ids)
        for fid in fragment_ids:
            version = self._fragment_versions.get(fid)
            ver_num = version.version if version else 0
            self._session_versions[session_id][fid] = ver_num

    def check_consistency(self, session_id: str) -> list[VersionSkew]:
        skews: list[VersionSkew] = []
        my_frags = self._session_fragments.get(session_id, set())

        for frag_id in my_frags:
            my_version = self._session_versions.get(session_id, {}).get(frag_id, 1)
            sessions_with_frag: dict[str, int] = {}

            for sid, frags in self._session_fragments.items():
                if frag_id in frags:
                    ver = self._session_versions.get(sid, {}).get(frag_id, 1)
                    sessions_with_frag[sid] = ver

            if len(sessions_with_frag) < 2:
                continue

            versions = set(sessions_with_frag.values())
            if len(versions) <= 1:
                continue

            max_gap = max(versions) - min(versions)
            severity = _classify_severity(len(sessions_with_frag), max_gap)

            skews.append(VersionSkew(
                fragment_id=frag_id,
                sessions_involved=list(sessions_with_frag.keys()),
                versions_seen=dict(sessions_with_frag),
                severity=severity,
            ))

        return skews

    def get_stale_fragments(
        self,
        session_id: str,
        max_age_seconds: float,
    ) -> list[str]:
        now = time.time()
        stale: list[str] = []
        my_frags = self._session_fragments.get(session_id, set())

        for frag_id in my_frags:
            version = self._fragment_versions.get(frag_id)
            if version is not None:
                if (now - version.timestamp) > max_age_seconds:
                    stale.append(frag_id)

        return stale

    def resolve_skew(
        self,
        skew: VersionSkew,
        mode: ConsistencyMode,
    ) -> ConsistencyAction:
        latest_version = max(skew.versions_seen.values())
        stale_sessions = [
            sid for sid, ver in skew.versions_seen.items()
            if ver < latest_version
        ]

        if mode == ConsistencyMode.STRICT:
            return ConsistencyAction(
                block=True,
                message=(
                    f"Version skew detected for fragment {skew.fragment_id[:12]}...: "
                    f"{len(stale_sessions)} session(s) have stale versions. "
                    f"Blocking until refreshed."
                ),
                refresh_fragment_ids=[skew.fragment_id],
            )

        return ConsistencyAction(
            block=False,
            message=(
                f"Version skew warning for fragment {skew.fragment_id[:12]}...: "
                f"{len(stale_sessions)} session(s) have stale versions "
                f"(severity: {skew.severity})."
            ),
            refresh_fragment_ids=[skew.fragment_id],
        )

    def on_fragment_updated(
        self,
        fragment_id: str,
        new_hash: str,
        timestamp: float,
    ) -> list[InvalidationEvent]:
        existing = self._fragment_versions.get(fragment_id)
        new_version = (existing.version + 1) if existing else 1

        self._fragment_versions[fragment_id] = FragmentVersion(
            fragment_id=fragment_id,
            version=new_version,
            content_hash=new_hash,
            timestamp=timestamp,
        )

        events: list[InvalidationEvent] = []
        for sid, frags in self._session_fragments.items():
            if fragment_id in frags:
                sess_ver = self._session_versions.get(sid, {}).get(fragment_id, 0)
                if sess_ver < new_version:
                    events.append(InvalidationEvent(
                        fragment_id=fragment_id,
                        timestamp=timestamp,
                        reason=f"Fragment updated to version {new_version} "
                               f"(session {sid} has version {sess_ver})",
                    ))

        return events

    def update_session_version(
        self,
        session_id: str,
        fragment_id: str,
    ) -> None:
        version = self._fragment_versions.get(fragment_id)
        if version is not None:
            self._session_versions[session_id][fragment_id] = version.version


def monitor_group_consistency(
    session_paths: list[str],
    store: FragmentStore,
    graph: ContextGraph,
) -> list[VersionSkew]:
    coordinator = ConsistencyCoordinator()

    for path in session_paths:
        fragments = store.find_by_session(path)
        frag_ids = [f.id for f in fragments]
        coordinator.register_session_fragments(path, frag_ids)

    all_skews: list[VersionSkew] = []
    seen_frags: set[str] = set()

    for path in session_paths:
        skews = coordinator.check_consistency(path)
        for skew in skews:
            if skew.fragment_id not in seen_frags:
                all_skews.append(skew)
                seen_frags.add(skew.fragment_id)

    return all_skews
