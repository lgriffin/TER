import numpy as np
import pytest

from ter_calculator import compare
from ter_calculator.budget_optimizer import (
    _prune_redundant,
    optimize_knapsack,
    recommend_context,
)
from ter_calculator.config_parse import parse_cost_model, parse_phase_weights
from ter_calculator.models import (
    Fragment,
    FragmentEdge,
    IntentVector,
    ScoredFragment,
    Session,
    SpanPhase,
    TERResult,
    EdgeType,
)


def result(sid, ter, tokens, waste):
    return TERResult(sid, ter, ter, {}, tokens, tokens - waste, waste)


def test_compare_sort_all_modes_and_unknown():
    values = [
        result("a", 0.5, 30, 20),
        result("b", 0.9, 10, 3),
        result("c", 0.7, 20, 7),
    ]
    assert [r.session_id for r in compare.sort_results(values)] == ["b", "c", "a"]
    assert [r.session_id for r in compare.sort_results(values, "tokens")] == [
        "b",
        "c",
        "a",
    ]
    assert [r.session_id for r in compare.sort_results(values, "waste")] == [
        "b",
        "c",
        "a",
    ]
    assert [r.session_id for r in compare.sort_results(values, "bogus")] == [
        "a",
        "c",
        "b",
    ]
    assert compare.compute_average_ter([]) == 0.0
    assert compare.compute_average_ter(values) == 0.7


def test_parse_cost_model_defaults_custom_and_errors():
    assert parse_cost_model("SONNET").output_rate == 15
    cm = parse_cost_model("1,2,3,4")
    assert (cm.input_rate, cm.output_rate, cm.cache_read_rate, cm.cache_write_rate) == (
        1,
        2,
        3,
        4,
    )
    with pytest.raises(ValueError, match="4 comma-separated"):
        parse_cost_model("1,2")
    with pytest.raises(ValueError, match="Invalid cost model"):
        parse_cost_model("1,x,3,4")


def test_parse_phase_weights_valid_tolerance_and_errors():
    got = parse_phase_weights("0.3,0.4,0.3")
    assert got[SpanPhase.TOOL_USE] == 0.4
    assert parse_phase_weights("0.33,0.33,0.34")[SpanPhase.GENERATION] == 0.34
    with pytest.raises(ValueError, match="3 comma-separated"):
        parse_phase_weights("1,0")
    with pytest.raises(ValueError, match="Invalid phase"):
        parse_phase_weights("a,b,c")
    with pytest.raises(ValueError, match="sum to 1.0"):
        parse_phase_weights(".2,.2,.2")


class Store:
    def __init__(self, frags):
        self.frags = {f.id: f for f in frags}

    def get(self, fid):
        return self.frags.get(fid)

    def find_by_session(self, sid):
        return [f for f in self.frags.values() if f.origin_session == sid]


class Graph:
    def __init__(self, edges=None):
        self.edges = edges or {}

    def get_neighbors(self, fid):
        return self.edges.get(fid, [])


def frag(fid, vec, tokens=100, session="s"):
    return Fragment(
        fid,
        fid,
        tokens,
        SpanPhase.REASONING,
        session,
        embedding=np.array(vec, dtype=np.float32),
    )


def scored(fid, rel, cost):
    return ScoredFragment(fid, rel, cost, SpanPhase.REASONING)


def test_prune_redundant_and_missing_neighbors():
    fs = [frag("a", [1, 0]), frag("b", [1, 0]), frag("c", [0, 1])]
    graph = Graph(
        {
            "a": [
                FragmentEdge("a", "b", EdgeType.DERIVATION),
                FragmentEdge("a", "missing", EdgeType.DEPENDENCY),
            ]
        }
    )
    pruned, count = _prune_redundant(
        [scored("a", 0.9, 100), scored("b", 0.8, 100), scored("c", 0.7, 100)],
        Store(fs),
        graph,
    )
    assert count == 1 and {x.fragment_id for x in pruned} == {"a", "c"}
    same, count = _prune_redundant([scored("a", 0.9, 100)], Store(fs), graph)
    assert count == 0 and len(same) == 1


def test_optimize_greedy_path_and_zero_budget(monkeypatch):
    import ter_calculator.budget_optimizer as bo

    monkeypatch.setattr(bo, "_DP_FRAGMENT_LIMIT", 2)
    items = [scored(str(i), 1 / (i + 1), 100) for i in range(3)]
    out = optimize_knapsack(items, 150, relevance_threshold=0)
    assert "greedy approximation" in out.reasoning and out.budget_used <= 150
    out = optimize_knapsack(items, 0, relevance_threshold=0)
    assert out.selected_fragment_ids == []


def test_recommend_context_empty_and_nonempty():
    intent = IntentVector("x", np.array([1, 0], dtype=np.float32))
    session = Session("s", "/tmp/s")
    empty = recommend_context(session, intent, 100, Store([]), Graph())
    assert empty.reasoning == "No fragments found for session"
    full = recommend_context(
        session,
        intent,
        100,
        Store([frag("a", [1, 0], 50)]),
        Graph(),
        relevance_threshold=0,
    )
    assert full.selected_fragment_ids == ["a"]
