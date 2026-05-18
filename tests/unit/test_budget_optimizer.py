"""Tests for budget_optimizer.py — knapsack token budget optimization."""

import numpy as np
import pytest

from ter_calculator.models import (
    Fragment,
    IntentVector,
    OptimizationResult,
    ScoredFragment,
    SpanPhase,
)
from ter_calculator.budget_optimizer import (
    score_fragments,
    optimize_knapsack,
    _dp_knapsack,
    _greedy_knapsack,
)


def _make_intent(dim=384):
    emb = np.random.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return IntentVector(
        text="test intent",
        embedding=emb,
        confidence=0.9,
        source_prompts=["test"],
    )


def _make_fragment(fid, phase=SpanPhase.REASONING, token_count=100, intent=None):
    if intent is not None:
        emb = intent.embedding + np.random.randn(384).astype(np.float32) * 0.1
        emb /= np.linalg.norm(emb)
    else:
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
    return Fragment(
        id=fid,
        text=f"text for {fid}",
        token_count=token_count,
        phase=phase,
        origin_session="test",
        created_at=0.0,
        embedding=emb.astype(np.float32),
    )


def _make_scored(fid, relevance, cost, phase=SpanPhase.REASONING):
    return ScoredFragment(
        fragment_id=fid,
        relevance_score=relevance,
        token_cost=cost,
        phase=phase,
    )


class TestScoreFragments:
    def test_scores_with_embeddings(self):
        intent = _make_intent()
        frags = [_make_fragment(f"f{i}", intent=intent) for i in range(3)]
        scored = score_fragments(frags, intent)
        assert len(scored) == 3
        for s in scored:
            assert s.relevance_score > 0

    def test_null_embedding_gets_zero(self):
        intent = _make_intent()
        frag = Fragment(
            id="no_emb",
            text="no embedding",
            token_count=10,
            phase=SpanPhase.REASONING,
            origin_session="test",
            created_at=0.0,
            embedding=None,
        )
        scored = score_fragments([frag], intent)
        assert scored[0].relevance_score == 0.0

    def test_phase_weights_applied(self):
        intent = _make_intent()
        emb = intent.embedding.copy()
        r_frag = Fragment(
            id="r", text="r", token_count=10,
            phase=SpanPhase.REASONING, origin_session="t",
            created_at=0.0, embedding=emb.copy(),
        )
        t_frag = Fragment(
            id="t", text="t", token_count=10,
            phase=SpanPhase.TOOL_USE, origin_session="t",
            created_at=0.0, embedding=emb.copy(),
        )
        scored = score_fragments([r_frag, t_frag], intent)
        r_score = next(s for s in scored if s.fragment_id == "r")
        t_score = next(s for s in scored if s.fragment_id == "t")
        assert t_score.relevance_score > r_score.relevance_score


class TestDPKnapsack:
    def test_selects_best_fit(self):
        items = [
            _make_scored("a", 0.9, 500),
            _make_scored("b", 0.5, 300),
            _make_scored("c", 0.3, 200),
        ]
        selected = _dp_knapsack(items, 700)
        ids = {s.fragment_id for s in selected}
        assert "a" in ids

    def test_respects_budget(self):
        items = [
            _make_scored("a", 0.9, 500),
            _make_scored("b", 0.8, 400),
        ]
        selected = _dp_knapsack(items, 600)
        total = sum(s.token_cost for s in selected)
        assert total <= 600

    def test_empty_items(self):
        assert _dp_knapsack([], 1000) == []

    def test_zero_budget(self):
        items = [_make_scored("a", 0.5, 100)]
        assert _dp_knapsack(items, 0) == []


class TestGreedyKnapsack:
    def test_selects_best_ratio(self):
        items = [
            _make_scored("a", 0.9, 900),   # ratio 0.001
            _make_scored("b", 0.8, 100),   # ratio 0.008 (best)
        ]
        selected = _greedy_knapsack(items, 200)
        ids = {s.fragment_id for s in selected}
        assert "b" in ids

    def test_respects_budget(self):
        items = [_make_scored(f"f{i}", 0.5, 200) for i in range(10)]
        selected = _greedy_knapsack(items, 500)
        total = sum(s.token_cost for s in selected)
        assert total <= 500

    def test_empty_items(self):
        assert _greedy_knapsack([], 1000) == []


class TestOptimizeKnapsack:
    def test_filters_low_relevance(self):
        scored = [
            _make_scored("a", 0.5, 100),
            _make_scored("b", 0.05, 100),  # below threshold
        ]
        result = optimize_knapsack(scored, 300, relevance_threshold=0.1)
        assert "b" not in result.selected_fragment_ids

    def test_all_fit_within_budget(self):
        scored = [
            _make_scored("a", 0.5, 100),
            _make_scored("b", 0.3, 100),
        ]
        result = optimize_knapsack(scored, 1000)
        assert len(result.selected_fragment_ids) == 2
        assert "All" in result.reasoning

    def test_budget_constraint(self):
        scored = [_make_scored(f"f{i}", 0.5, 200) for i in range(10)]
        result = optimize_knapsack(scored, 500)
        assert result.budget_used <= 500

    def test_result_structure(self):
        scored = [_make_scored("a", 0.5, 100)]
        result = optimize_knapsack(scored, 1000)
        assert isinstance(result, OptimizationResult)
        assert result.budget_ceiling == 1000
        assert result.total_relevance > 0

    def test_empty_input(self):
        result = optimize_knapsack([], 1000)
        assert result.selected_fragment_ids == []
        assert result.budget_used == 0
