"""Tests for context_graph.py — DAG of fragment relationships."""

import time
import pytest

from ter_calculator.models import EdgeType, Fragment, FragmentEdge, SpanPhase
from ter_calculator.context_graph import ContextGraph


@pytest.fixture
def graph(tmp_path):
    db = tmp_path / "test_graph.db"
    g = ContextGraph(db_path=db)
    yield g
    g.close()


def _frag(fid, phase="reasoning", **kwargs):
    defaults = {
        "id": fid,
        "text": f"text for {fid}",
        "token_count": 10,
        "phase": SpanPhase(phase),
        "origin_session": "test-session",
        "created_at": time.time(),
    }
    defaults.update(kwargs)
    return Fragment(**defaults)


class TestNodeAndEdge:
    def test_add_node(self, graph):
        graph.add_node("a", {"role": "test"})
        assert graph.node_count == 1

    def test_add_edge(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        assert graph.edge_count == 1
        assert graph.node_count == 2

    def test_get_neighbors(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("a", "c", EdgeType.DERIVATION)
        neighbors = graph.get_neighbors("a")
        assert len(neighbors) == 2

    def test_get_neighbors_filtered(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("a", "c", EdgeType.DERIVATION)
        deps = graph.get_neighbors("a", EdgeType.DEPENDENCY)
        assert len(deps) == 1
        assert deps[0].target_id == "b"

    def test_get_reverse_neighbors(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("c", "b", EdgeType.DERIVATION)
        reverse = graph.get_reverse_neighbors("b")
        assert len(reverse) == 2

    def test_remove_node(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("b", "c", EdgeType.DERIVATION)
        graph.remove_node("b")
        assert graph.node_count == 2
        assert graph.edge_count == 0

    def test_edge_update(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY, weight=1.0)
        graph.add_edge("a", "b", EdgeType.DEPENDENCY, weight=2.0)
        neighbors = graph.get_neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0].weight == 2.0


class TestSubgraph:
    def test_subgraph_basic(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("b", "c", EdgeType.DEPENDENCY)
        graph.add_edge("c", "d", EdgeType.DEPENDENCY)
        sub = graph.get_subgraph(["a"], max_depth=2)
        assert sub.node_count == 3  # a, b, c
        assert "d" not in sub._nodes

    def test_subgraph_max_depth_0(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        sub = graph.get_subgraph(["a"], max_depth=0)
        assert sub.node_count == 1

    def test_subgraph_multiple_roots(self, graph):
        graph.add_edge("a", "c", EdgeType.DEPENDENCY)
        graph.add_edge("b", "c", EdgeType.DEPENDENCY)
        sub = graph.get_subgraph(["a", "b"], max_depth=1)
        assert sub.node_count == 3


class TestTopologicalSort:
    def test_linear_dag(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("b", "c", EdgeType.DEPENDENCY)
        order = graph.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond_dag(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("a", "c", EdgeType.DEPENDENCY)
        graph.add_edge("b", "d", EdgeType.DEPENDENCY)
        graph.add_edge("c", "d", EdgeType.DEPENDENCY)
        order = graph.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_single_node(self, graph):
        graph.add_node("solo")
        order = graph.topological_sort()
        assert order == ["solo"]


class TestCycleDetection:
    def test_no_cycles_in_dag(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("b", "c", EdgeType.DEPENDENCY)
        cycles = graph.detect_cycles()
        assert len(cycles) == 0

    def test_detect_simple_cycle(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("b", "a", EdgeType.DEPENDENCY)
        cycles = graph.detect_cycles()
        assert len(cycles) > 0

    def test_detect_longer_cycle(self, graph):
        graph.add_edge("a", "b", EdgeType.DEPENDENCY)
        graph.add_edge("b", "c", EdgeType.DEPENDENCY)
        graph.add_edge("c", "a", EdgeType.DEPENDENCY)
        cycles = graph.detect_cycles()
        assert len(cycles) > 0


class TestBuildFromSession:
    def test_builds_edges(self, graph):
        frags = [
            _frag("r1", phase="reasoning"),
            _frag("t1", phase="tool_use"),
            _frag("g1", phase="generation"),
        ]
        graph.build_from_session(frags)
        assert graph.node_count == 3
        assert graph.edge_count > 0

    def test_derivation_edges(self, graph):
        frags = [
            _frag("r1", phase="reasoning"),
            _frag("g1", phase="generation"),
        ]
        graph.build_from_session(frags)
        derivations = graph.get_neighbors("g1", EdgeType.DERIVATION)
        assert len(derivations) == 1
        assert derivations[0].target_id == "r1"

    def test_co_occurrence_edges(self, graph):
        frags = [_frag("a"), _frag("b"), _frag("c")]
        graph.build_from_session(frags)
        co_occur = [
            e for e in graph.all_edges() if e.edge_type == EdgeType.CO_OCCURRENCE
        ]
        assert len(co_occur) == 3  # a-b, a-c, b-c

    def test_empty_session(self, graph):
        graph.build_from_session([])
        assert graph.node_count == 0
        assert graph.edge_count == 0


class TestPruneStale:
    def test_prune_removes_expired(self, graph):
        graph.add_node(
            "old",
            {
                "created_at": 1.0,
                "ttl_seconds": 1,
            },
        )
        graph.add_edge("old", "other", EdgeType.DEPENDENCY)
        removed = graph.prune_stale(max_age_hours=0)
        assert removed >= 1

    def test_prune_keeps_fresh(self, graph):
        graph.add_node(
            "fresh",
            {
                "created_at": time.time(),
                "ttl_seconds": 99999,
            },
        )
        removed = graph.prune_stale(max_age_hours=0)
        assert removed == 0


class TestPersistence:
    def test_edges_persist_across_instances(self, tmp_path):
        db = tmp_path / "persist.db"
        g1 = ContextGraph(db_path=db)
        g1.add_edge("a", "b", EdgeType.DEPENDENCY, weight=0.5)
        g1.close()

        g2 = ContextGraph(db_path=db)
        neighbors = g2.get_neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0].target_id == "b"
        assert neighbors[0].weight == 0.5
        g2.close()
