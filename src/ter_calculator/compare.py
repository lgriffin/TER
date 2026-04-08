"""Multi-session comparison logic."""

from __future__ import annotations

from .models import TERResult


def sort_results(
    results: list[TERResult],
    sort_by: str = "ter",
) -> list[TERResult]:
    """Sort TER results by the specified field."""
    sort_key = {
        "ter": lambda r: r.aggregate_ter,
        "tokens": lambda r: r.total_tokens,
        "waste": lambda r: r.waste_tokens,
    }
    key_fn = sort_key.get(sort_by, sort_key["ter"])
    reverse = sort_by == "ter"  # Higher TER first, lower tokens/waste first.
    return sorted(results, key=key_fn, reverse=reverse)


def compute_average_ter(results: list[TERResult]) -> float:
    """Compute the average aggregate TER across results."""
    if not results:
        return 0.0
    return round(
        sum(r.aggregate_ter for r in results) / len(results), 4
    )
