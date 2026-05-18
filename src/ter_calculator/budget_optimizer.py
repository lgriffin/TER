"""Knapsack-based token budget optimization for context fragments.

Implements the Token Budget Planner from the Token Aware Microprompt
Orchestrator patent: scores fragments by relevance to intent, applies
knapsack optimization to select the maximum-relevance subset within a
token budget, and prunes redundant fragments using the context graph.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .classifier import cosine_similarity
from .context_graph import ContextGraph
from .fragment_store import FragmentStore
from .models import (
    Fragment,
    IntentVector,
    OptimizationResult,
    PHASE_WEIGHTS_DEFAULT,
    ScoredFragment,
    Session,
    SpanPhase,
)

_DP_FRAGMENT_LIMIT = 500
_TOKEN_QUANT = 100


def score_fragments(
    fragments: list[Fragment],
    intent: IntentVector,
) -> list[ScoredFragment]:
    scored: list[ScoredFragment] = []
    for frag in fragments:
        if frag.embedding is None:
            sim = 0.0
        else:
            sim = cosine_similarity(frag.embedding, intent.embedding)

        phase_weight = PHASE_WEIGHTS_DEFAULT.get(frag.phase, 0.3)
        relevance = sim * phase_weight

        scored.append(ScoredFragment(
            fragment_id=frag.id,
            relevance_score=relevance,
            token_cost=frag.token_count,
            phase=frag.phase,
        ))
    return scored


def _prune_redundant(
    scored: list[ScoredFragment],
    store: FragmentStore,
    graph: ContextGraph,
    threshold: float = 0.85,
) -> tuple[list[ScoredFragment], int]:
    if len(scored) <= 1:
        return scored, 0

    by_id = {s.fragment_id: s for s in scored}
    removed: set[str] = set()

    sorted_scored = sorted(scored, key=lambda s: s.relevance_score, reverse=True)

    for sf in sorted_scored:
        if sf.fragment_id in removed:
            continue

        neighbors = graph.get_neighbors(sf.fragment_id)
        for edge in neighbors:
            child_id = edge.target_id
            if child_id in removed or child_id not in by_id:
                continue

            parent_frag = store.get(sf.fragment_id)
            child_frag = store.get(child_id)
            if (
                parent_frag is not None
                and child_frag is not None
                and parent_frag.embedding is not None
                and child_frag.embedding is not None
            ):
                sim = cosine_similarity(parent_frag.embedding, child_frag.embedding)
                if sim > threshold:
                    removed.add(child_id)

    pruned = [s for s in scored if s.fragment_id not in removed]
    return pruned, len(removed)


def _dp_knapsack(
    items: list[ScoredFragment], capacity: int
) -> list[ScoredFragment]:
    if not items or capacity <= 0:
        return []

    quant_cap = max(1, capacity // _TOKEN_QUANT)
    n = len(items)

    quant_costs = [max(1, item.token_cost // _TOKEN_QUANT) for item in items]

    dp = np.zeros((n + 1, quant_cap + 1), dtype=np.float64)
    for i in range(1, n + 1):
        w = quant_costs[i - 1]
        v = items[i - 1].relevance_score
        for c in range(quant_cap + 1):
            if w <= c:
                dp[i, c] = max(dp[i - 1, c], dp[i - 1, c - w] + v)
            else:
                dp[i, c] = dp[i - 1, c]

    selected: list[ScoredFragment] = []
    c = quant_cap
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            selected.append(items[i - 1])
            c -= quant_costs[i - 1]

    selected.reverse()
    return selected


def _greedy_knapsack(
    items: list[ScoredFragment], capacity: int
) -> list[ScoredFragment]:
    if not items or capacity <= 0:
        return []

    ranked = sorted(
        items,
        key=lambda s: s.relevance_score / max(1, s.token_cost),
        reverse=True,
    )

    selected: list[ScoredFragment] = []
    used = 0
    for item in ranked:
        if used + item.token_cost <= capacity:
            selected.append(item)
            used += item.token_cost
    return selected


def optimize_knapsack(
    scored: list[ScoredFragment],
    budget_tokens: int,
    *,
    relevance_threshold: float = 0.1,
    store: FragmentStore | None = None,
    graph: ContextGraph | None = None,
    redundancy_threshold: float = 0.85,
) -> OptimizationResult:
    filtered = [s for s in scored if s.relevance_score >= relevance_threshold]

    pruned_count = 0
    if store is not None and graph is not None:
        filtered, pruned_count = _prune_redundant(
            filtered, store, graph, threshold=redundancy_threshold
        )

    total_cost = sum(s.token_cost for s in filtered)
    if total_cost <= budget_tokens:
        return OptimizationResult(
            selected_fragment_ids=[s.fragment_id for s in filtered],
            total_tokens=total_cost,
            total_relevance=sum(s.relevance_score for s in filtered),
            budget_used=total_cost,
            budget_ceiling=budget_tokens,
            pruned_count=pruned_count,
            reasoning=f"All {len(filtered)} fragments fit within budget",
        )

    if len(filtered) <= _DP_FRAGMENT_LIMIT:
        selected = _dp_knapsack(filtered, budget_tokens)
        method = "dynamic programming"
    else:
        selected = _greedy_knapsack(filtered, budget_tokens)
        method = "greedy approximation"

    used = sum(s.token_cost for s in selected)
    return OptimizationResult(
        selected_fragment_ids=[s.fragment_id for s in selected],
        total_tokens=used,
        total_relevance=sum(s.relevance_score for s in selected),
        budget_used=used,
        budget_ceiling=budget_tokens,
        pruned_count=pruned_count,
        reasoning=(
            f"Selected {len(selected)}/{len(filtered)} fragments via "
            f"{method} ({used}/{budget_tokens} tokens)"
        ),
    )


def recommend_context(
    session: Session,
    intent: IntentVector,
    budget_tokens: int,
    store: FragmentStore,
    graph: ContextGraph,
    *,
    relevance_threshold: float = 0.1,
) -> OptimizationResult:
    fragments = store.find_by_session(session.session_id)
    if not fragments:
        return OptimizationResult(
            selected_fragment_ids=[],
            total_tokens=0,
            total_relevance=0.0,
            budget_used=0,
            budget_ceiling=budget_tokens,
            pruned_count=0,
            reasoning="No fragments found for session",
        )

    scored = score_fragments(fragments, intent)
    return optimize_knapsack(
        scored,
        budget_tokens,
        relevance_threshold=relevance_threshold,
        store=store,
        graph=graph,
    )
