"""Command implementation module extracted from :mod:`ter_calculator.cli`."""

from __future__ import annotations

import sys


def _cmd_budget(args) -> int:
    """Execute the budget subcommand for token budget recommendations."""
    import json as json_mod
    from ..adaptive_budget import recommend_budget, HistoricalBudgetAnalyzer

    if not args.intent_text.strip():
        print("Error: Intent text cannot be empty", file=sys.stderr)
        return 1

    # Load historical data if requested
    history = None
    if args.use_history:
        try:
            history = HistoricalBudgetAnalyzer(
                history_path=args.history_path if args.history_path else None
            )
        except Exception as e:
            print(f"Warning: Could not load history: {e}", file=sys.stderr)

    # Get recommendation
    try:
        rec = recommend_budget(args.intent_text, history=history)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return 1

    # Format output
    if args.output_format == "json":
        print(
            json_mod.dumps(
                {
                    "complexity": rec.complexity.value,
                    "model_tier": rec.model_tier.value,
                    "max_thinking_tokens": rec.max_thinking_tokens,
                    "estimated_total_tokens": rec.estimated_total_tokens,
                    "estimated_cost_usd": rec.estimated_cost_usd,
                    "confidence": rec.confidence,
                    "reasoning": rec.reasoning,
                },
                indent=2,
            )
        )
    else:
        print("Budget Recommendation")
        print("═" * 50)
        print(f"Complexity: {rec.complexity.value} ({rec.confidence:.0%} confidence)")
        print(f"Model: {rec.model_tier.value}")
        print(f"Max Thinking Tokens: {rec.max_thinking_tokens:,}")
        print(f"Est. Total Tokens: {rec.estimated_total_tokens:,}")
        print(f"Est. Cost: ${rec.estimated_cost_usd:.4f}")
        print(f"\nReasoning:\n{rec.reasoning}")

    return 0
