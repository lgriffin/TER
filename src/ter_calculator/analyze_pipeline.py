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
    )


def analyze_session(args) -> "TERResult":
    """Run full TER pipeline for one JSONL session. `args` matches analyze subcommand."""
    from .config_parse import parse_cost_model, parse_phase_weights
    from .loader import load_session, segment_spans
    from .intent import extract_intent
    from .classifier import classify_spans
    from .compute import compute_ter
    from .economics import compute_economics

    phase_weights = parse_phase_weights(args.phase_weights)

    session = load_session(args.session_path)
    spans = segment_spans(session)
    intent = extract_intent(session)

    classified = classify_spans(
        spans,
        intent,
        similarity_threshold=args.similarity_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    result = compute_ter(
        classified,
        session_id=session.session_id,
        intent=intent,
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

    return result
