"""Directed acyclic graph of context fragment relationships.

Implements the Shared Context Graph from the Token Aware Microprompt
Orchestrator patent: a DAG encoding dependency, derivation, and co-occurrence
relationships between content-addressable fragments.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .models import EdgeType, Fragment, FragmentEdge

_DEFAULT_DB_DIR = Path.home() / ".cache" / "ter"


def _default_db_path() -> Path:
    return _DEFAULT_DB_DIR / "fragments.db"


class ContextGraph:
    """In-memory adjacency list with SQLite persistence."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=5)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._adj: dict[str, list[FragmentEdge]] = defaultdict(list)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._init_schema()
        self._load_from_db()

    def _init_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS edges ("
            "  source_id TEXT NOT NULL,"
            "  target_id TEXT NOT NULL,"
            "  edge_type TEXT NOT NULL,"
            "  weight REAL NOT NULL DEFAULT 1.0,"
            "  PRIMARY KEY (source_id, target_id, edge_type)"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
        )
        self._conn.commit()

    def _load_from_db(self) -> None:
        rows = self._conn.execute(
            "SELECT source_id, target_id, edge_type, weight FROM edges"
        ).fetchall()
        for src, tgt, etype, weight in rows:
            edge = FragmentEdge(
                source_id=src,
                target_id=tgt,
                edge_type=EdgeType(etype),
                weight=weight,
            )
            self._adj[src].append(edge)
            self._nodes.setdefault(src, {})
            self._nodes.setdefault(tgt, {})

    def add_node(
        self, fragment_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._nodes[fragment_id] = metadata or {}

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> None:
        edge = FragmentEdge(
            source_id=source,
            target_id=target,
            edge_type=edge_type,
            weight=weight,
        )
        existing = self._adj.get(source, [])
        for i, e in enumerate(existing):
            if e.target_id == target and e.edge_type == edge_type:
                existing[i] = edge
                break
        else:
            self._adj[source].append(edge)

        self._nodes.setdefault(source, {})
        self._nodes.setdefault(target, {})

        self._conn.execute(
            "INSERT OR REPLACE INTO edges "
            "(source_id, target_id, edge_type, weight) VALUES (?, ?, ?, ?)",
            (source, target, edge_type.value, weight),
        )
        self._conn.commit()

    def remove_node(self, fragment_id: str) -> None:
        self._adj.pop(fragment_id, None)
        for src in list(self._adj):
            self._adj[src] = [
                e for e in self._adj[src] if e.target_id != fragment_id
            ]
        self._nodes.pop(fragment_id, None)
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (fragment_id, fragment_id),
        )
        self._conn.commit()

    def get_neighbors(
        self,
        fragment_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[FragmentEdge]:
        edges = self._adj.get(fragment_id, [])
        if edge_type is not None:
            return [e for e in edges if e.edge_type == edge_type]
        return list(edges)

    def get_reverse_neighbors(
        self,
        fragment_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[FragmentEdge]:
        result: list[FragmentEdge] = []
        for edges in self._adj.values():
            for e in edges:
                if e.target_id == fragment_id:
                    if edge_type is None or e.edge_type == edge_type:
                        result.append(e)
        return result

    def get_subgraph(
        self, root_ids: list[str], max_depth: int = 3
    ) -> ContextGraph:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        for rid in root_ids:
            if rid in self._nodes:
                queue.append((rid, 0))

        subgraph = ContextGraph.__new__(ContextGraph)
        subgraph._adj = defaultdict(list)
        subgraph._nodes = {}
        subgraph._conn = self._conn
        subgraph._db_path = self._db_path

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            subgraph._nodes[node_id] = self._nodes.get(node_id, {})

            if depth < max_depth:
                for edge in self._adj.get(node_id, []):
                    subgraph._adj[node_id].append(edge)
                    if edge.target_id not in visited:
                        queue.append((edge.target_id, depth + 1))

        return subgraph

    def topological_sort(self) -> list[str]:
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for edges in self._adj.values():
            for e in edges:
                if e.target_id in in_degree:
                    in_degree[e.target_id] += 1

        queue: deque[str] = deque(
            n for n, d in in_degree.items() if d == 0
        )
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for edge in self._adj.get(node, []):
                if edge.target_id in in_degree:
                    in_degree[edge.target_id] -= 1
                    if in_degree[edge.target_id] == 0:
                        queue.append(edge.target_id)

        return result

    def detect_cycles(self) -> list[list[str]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in self._nodes}
        parent: dict[str, str | None] = {n: None for n in self._nodes}
        cycles: list[list[str]] = []

        def _dfs(u: str) -> None:
            color[u] = GRAY
            for edge in self._adj.get(u, []):
                v = edge.target_id
                if v not in color:
                    continue
                if color[v] == GRAY:
                    cycle = [v, u]
                    node = u
                    while node != v:
                        node = parent.get(node)  # type: ignore[assignment]
                        if node is None or node == v:
                            break
                        cycle.append(node)
                    cycle.reverse()
                    cycles.append(cycle)
                elif color[v] == WHITE:
                    parent[v] = u
                    _dfs(v)
            color[u] = BLACK

        for n in self._nodes:
            if color.get(n, WHITE) == WHITE:
                _dfs(n)

        return cycles

    def build_from_session(self, fragments: list[Fragment]) -> None:
        if not fragments:
            return

        for frag in fragments:
            self.add_node(frag.id, {
                "phase": frag.phase.value,
                "origin_session": frag.origin_session,
                "token_weight": frag.token_count,
                "created_at": frag.created_at,
                "ttl_seconds": frag.ttl_seconds,
            })

        by_message: dict[str, list[Fragment]] = defaultdict(list)
        for frag in fragments:
            key = frag.origin_session
            by_message[key].append(frag)

        prev_reasoning: Fragment | None = None
        prev_tool_use: Fragment | None = None

        for frag in fragments:
            if frag.phase.value == "reasoning":
                prev_reasoning = frag
            elif frag.phase.value == "tool_use":
                if prev_tool_use is not None and frag is not prev_tool_use:
                    self.add_edge(
                        frag.id,
                        prev_tool_use.id,
                        EdgeType.DEPENDENCY,
                        weight=1.0,
                    )
                prev_tool_use = frag
            elif frag.phase.value == "generation":
                if prev_reasoning is not None:
                    self.add_edge(
                        frag.id,
                        prev_reasoning.id,
                        EdgeType.DERIVATION,
                        weight=1.0,
                    )

        if len(fragments) > 1:
            weight = 1.0 / len(fragments)
            for i, fa in enumerate(fragments):
                for fb in fragments[i + 1:]:
                    self.add_edge(
                        fa.id, fb.id, EdgeType.CO_OCCURRENCE, weight=weight
                    )

    def prune_stale(self, max_age_hours: float = 24.0) -> int:
        cutoff = time.time() - (max_age_hours * 3600)
        stale: list[str] = []
        for nid, meta in self._nodes.items():
            created = meta.get("created_at", 0)
            ttl = meta.get("ttl_seconds", 3600)
            if (created + ttl) < cutoff:
                stale.append(nid)

        for nid in stale:
            self.remove_node(nid)

        return len(stale)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adj.values())

    def all_edges(self) -> list[FragmentEdge]:
        result: list[FragmentEdge] = []
        for edges in self._adj.values():
            result.extend(edges)
        return result

    def close(self) -> None:
        self._conn.close()
