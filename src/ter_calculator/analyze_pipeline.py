"""Shared single-session analysis pipeline (analyze / report)."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TERResult


def default_analyze_args(session_path: str) -> argparse.Namespace:
    """Defaults for `ter compare --baseline` when analyze flags are not on the CLI."""
    return argparse.Namespace(
        session_path=session_path,
        similarity_threshold=0.40,
        confidence_threshold=0.75,
        restatement_threshold=0.85,
        phase_weights="0.3,0.4,0.3",
        no_waste_patterns=False,
        cost_model="sonnet",
        no_input_analysis=False,
        prompt_similarity_threshold=0.75,
        fine_segmentation=False,
        segment_min_tokens=12,
        segment_max_tokens=180,
    )


def analyze_session(args) -> "TERResult":
    """Run full TER pipeline for one JSONL session. `args` matches analyze subcommand."""
    from .config_parse import parse_cost_model, parse_phase_weights
    from .loader import load_session, segment_spans
    from .span_segmentation import SegmentationConfig
    from .intent_extraction import SlidingIntentExtractor
    from .classifier import classify_spans
    from .compute import compute_ter
    from .economics import compute_economics

    phase_weights = parse_phase_weights(args.phase_weights)

    session = load_session(args.session_path)
    fine_segmentation = getattr(args, "fine_segmentation", False)
    if fine_segmentation:
        spans = segment_spans(
            session,
            SegmentationConfig(
                enabled=True,
                min_tokens=getattr(args, "segment_min_tokens", 12),
                max_tokens=getattr(args, "segment_max_tokens", 180),
            ),
        )
    else:
        # Preserve the historical one-argument call contract. Besides keeping the
        # default path lightweight, this remains compatible with integrations and
        # tests that replace ``segment_spans`` with a one-argument callable.
        spans = segment_spans(session)
    intents = SlidingIntentExtractor().extract(session.user_prompts)

    classified = classify_spans(
        spans,
        intents,
        similarity_threshold=args.similarity_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    result = compute_ter(
        classified,
        session_id=session.session_id,
        intent=intents[0],
        phase_weights=phase_weights,
    )

    if not args.no_waste_patterns:
        from .waste import detect_waste_patterns

        result.waste_patterns = detect_waste_patterns(
            classified,
            restatement_threshold=args.restatement_threshold,
            session=session,
        )

    cost_model = parse_cost_model(args.cost_model)
    result.economics = compute_economics(session, classified, cost_model)

    if not args.no_input_analysis:
        from .input_analysis import analyze_input

        result.input_analysis = analyze_input(
            session,
            similarity_threshold=args.prompt_similarity_threshold,
        )

    if hasattr(args, "cost_weighted") and args.cost_weighted:
        from .cost_model import generate_cost_report
        from .models import ALIGNED_LABELS

        # Extract usage dict from session economics
        usage = {
            "input_tokens": result.economics.total_input_tokens
            if result.economics
            else 0,
            "output_tokens": result.economics.total_output_tokens
            if result.economics
            else 0,
            "cache_creation_tokens": result.economics.total_cache_creation_tokens
            if result.economics
            else 0,
            "cache_read_tokens": result.economics.total_cache_read_tokens
            if result.economics
            else 0,
        }

        # Generate cost report
        result.cost_report = generate_cost_report(
            spans=[
                {
                    "phase": cs.span.phase.value,
                    "token_count": cs.span.token_count,
                    "is_aligned": cs.label in ALIGNED_LABELS,
                    "text": cs.span.text,
                }
                for cs in classified
            ],
            full_text=" ".join(cs.span.text for cs in classified),
            model=args.cost_model,
            raw_ter=result.aggregate_ter,
            usage=usage,
        )

    if hasattr(args, "check_overthinking") and args.check_overthinking:
        from .overthinking import analyze_overthinking
        from .models import SpanPhase

        # Extract reasoning spans
        reasoning_spans = [
            cs.span.text for cs in classified if cs.span.phase == SpanPhase.REASONING
        ]

        if len(reasoning_spans) >= 3:  # MIN_REASONING_SPANS
            result.overthinking_result = analyze_overthinking(reasoning_spans)

    return result
