"""JSON formatting for TER results."""

from __future__ import annotations

import json

from .models import TERResult
from .waste import summarize_waste


def format_json(result: TERResult) -> str:
    data = ter_result_to_dict(result)
    return json.dumps(data, indent=2)


def format_comparison_json(results: list[TERResult]) -> str:
    data = {
        "sessions": [ter_result_to_dict(r) for r in results],
        "average_ter": round(
            sum(r.aggregate_ter for r in results) / len(results), 4
        ) if results else 0.0,
    }
    return json.dumps(data, indent=2)


def format_grouped_json(
    parent_result: TERResult,
    subagent_results: list[TERResult],
) -> str:
    from .formatter import _compute_group_aggregates

    all_results = [parent_result] + subagent_results
    agg = _compute_group_aggregates(all_results)

    data = {
        "group": {
            "parent_session_id": parent_result.session_id,
            "subagent_count": len(subagent_results),
            **agg,
        },
        "parent": ter_result_to_dict(parent_result),
        "subagents": [ter_result_to_dict(r) for r in subagent_results],
    }
    return json.dumps(data, indent=2)


def ter_result_to_dict(result: TERResult) -> dict:
    from .formatter import _build_waste_breakdown, _compute_waste_cost
    from .models import CostModel

    data: dict = {
        "session_id": result.session_id,
        "aggregate_ter": result.aggregate_ter,
        "raw_ratio": result.raw_ratio,
        "phase_scores": result.phase_scores,
        "total_tokens": result.total_tokens,
        "aligned_tokens": result.aligned_tokens,
        "waste_tokens": result.waste_tokens,
    }
    if result.intent:
        data["intent_confidence"] = result.intent.confidence
    if result.waste_patterns:
        data["waste_patterns"] = [
            {
                "type": wp.pattern_type,
                "start_position": wp.start_position,
                "end_position": wp.end_position,
                "spans_involved": wp.spans_involved,
                "tokens_wasted": wp.tokens_wasted,
                "description": wp.description,
            }
            for wp in result.waste_patterns
        ]
    if result.classified_spans:
        summary = summarize_waste(result.classified_spans, result.waste_patterns or [])
        data["waste_summary"] = {
            "total_waste_tokens": summary["total_waste_tokens"],
            "waste_by_category": summary["waste_by_category"],
            "waste_by_phase": summary["waste_by_phase"],
            "top_patterns": summary["top_patterns"],
            "explanation": summary["explanation"],
        }
    rows = _build_waste_breakdown(result)
    if rows:
        total_waste = sum(t for _, t, _, _ in rows)
        cm = result.economics.cost_model if result.economics else CostModel()
        sources = []
        for label, tokens, count, kind in rows:
            rate = cm.output_rate if kind == "output" else cm.input_rate
            row_cost = float(tokens) * rate / 1_000_000
            sources.append({
                "source": label,
                "tokens": tokens,
                "percentage": round(tokens / total_waste * 100, 1) if total_waste > 0 else 0,
                "cost_usd": round(row_cost, 6),
                "count": count,
                "pricing": kind,
            })
        data["waste_breakdown"] = {
            "sources": sources,
            "total_tokens": total_waste,
            "total_cost_usd": round(_compute_waste_cost(result), 6),
        }
    if result.economics is not None:
        econ = result.economics
        data["economics"] = {
            "total_input_tokens": econ.total_input_tokens,
            "total_output_tokens": econ.total_output_tokens,
            "total_cache_creation_tokens": econ.total_cache_creation_tokens,
            "total_cache_read_tokens": econ.total_cache_read_tokens,
            "input_output_ratio": econ.input_output_ratio,
            "cache_hit_rate": econ.cache_hit_rate,
            "estimated_cost_usd": econ.estimated_cost_usd,
            "estimated_waste_cost_usd": econ.estimated_waste_cost_usd,
            "cost_model": {
                "input_rate": econ.cost_model.input_rate,
                "output_rate": econ.cost_model.output_rate,
                "cache_read_rate": econ.cost_model.cache_read_rate,
                "cache_write_rate": econ.cost_model.cache_write_rate,
            },
            "positional": {
                "early_ter": econ.positional.early_ter,
                "mid_ter": econ.positional.mid_ter,
                "late_ter": econ.positional.late_ter,
                "early_span_count": econ.positional.early_span_count,
                "mid_span_count": econ.positional.mid_span_count,
                "late_span_count": econ.positional.late_span_count,
            },
            "input_growth": {
                "turn_input_tokens": econ.input_growth.turn_input_tokens,
                "growth_rate": econ.input_growth.growth_rate,
                "is_superlinear": econ.input_growth.is_superlinear,
                "context_bloat_detected": econ.input_growth.context_bloat_detected,
            },
        }
    if result.input_analysis is not None:
        ia = result.input_analysis
        bd = ia.token_breakdown
        ps = ia.prompt_similarity
        data["input_analysis"] = {
            "token_breakdown": {
                "user_input_tokens": bd.user_input_tokens,
                "user_result_tokens": bd.user_result_tokens,
                "model_reasoning_tokens": bd.model_reasoning_tokens,
                "model_tool_tokens": bd.model_tool_tokens,
                "model_generation_tokens": bd.model_generation_tokens,
                "total_user_tokens": bd.total_user_tokens,
                "total_model_tokens": bd.total_model_tokens,
                "user_ratio": bd.user_ratio,
            },
            "prompt_similarity": {
                "prompt_count": ps.prompt_count,
                "prompt_redundancy_score": ps.prompt_redundancy_score,
                "similar_pairs": [
                    {
                        "prompt_a_index": p.prompt_a_index,
                        "prompt_b_index": p.prompt_b_index,
                        "similarity": p.similarity,
                        "prompt_a_text": p.prompt_a_text,
                        "prompt_b_text": p.prompt_b_text,
                    }
                    for p in ps.similar_pairs
                ],
            },
            "intent_drift": {
                "overall_trajectory": ia.intent_drift.overall_trajectory,
                "average_drift": ia.intent_drift.average_drift,
                "steps": [
                    {
                        "from_index": s.from_index,
                        "to_index": s.to_index,
                        "similarity": s.similarity,
                        "drift_type": s.drift_type,
                    }
                    for s in ia.intent_drift.steps
                ],
            },
            "prompt_response_alignment": {
                "average_alignment": ia.prompt_response_alignment.average_alignment,
                "low_alignment_count": ia.prompt_response_alignment.low_alignment_count,
                "pairs": [
                    {
                        "prompt_index": p.prompt_index,
                        "prompt_text": p.prompt_text,
                        "response_text": p.response_text,
                        "alignment": p.alignment,
                    }
                    for p in ia.prompt_response_alignment.pairs
                ],
            },
        }
    if result.cost_report is not None:
        cr = result.cost_report
        data["cost_report"] = {
            "cost_weighted_ter": cr.cost_ter.cost_weighted_ter,
            "raw_ter": cr.cost_ter.raw_ter,
            "total_cost_usd": cr.cost_ter.total_cost_usd,
            "waste_cost_usd": cr.cost_ter.waste_cost_usd,
            "savings_if_perfect": cr.cost_ter.savings_if_perfect,
            "semantic_density": {
                "density_score": cr.session_density.density_score,
                "vocabulary_richness": cr.session_density.vocabulary_richness,
                "information_entropy": cr.session_density.information_entropy,
                "redundancy_ratio": cr.session_density.redundancy_ratio,
            },
            "recommendations": cr.recommendations,
            "model_tier": cr.model_tier,
        }
    if result.overthinking_result is not None:
        ot = result.overthinking_result
        data["overthinking_analysis"] = {
            "is_overthinking": ot.is_overthinking,
            "total_reasoning_tokens": ot.total_reasoning_tokens,
            "useful_reasoning_tokens": ot.useful_reasoning_tokens,
            "wasted_reasoning_tokens": ot.wasted_reasoning_tokens,
            "reasoning_efficiency": ot.reasoning_efficiency,
            "optimal_cutoff_index": ot.optimal_cutoff_index,
            "recommended_budget": ot.recommended_budget,
            "explanation": ot.explanation,
        }
    return data
