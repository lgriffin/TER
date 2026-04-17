"""Parse CLI cost model and phase-weight strings (shared by analyze/report)."""

from __future__ import annotations

from .models import CostModel, SpanPhase


def parse_cost_model(value: str) -> CostModel:
    """Parse cost model argument."""
    if value.lower() == "sonnet":
        return CostModel()
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"Cost model must be 'sonnet' or 4 comma-separated rates, got: {value}"
        )
    try:
        return CostModel(
            input_rate=float(parts[0]),
            output_rate=float(parts[1]),
            cache_read_rate=float(parts[2]),
            cache_write_rate=float(parts[3]),
        )
    except ValueError:
        raise ValueError(f"Invalid cost model rates: {value}")


def parse_phase_weights(weights_str: str) -> dict[SpanPhase, float]:
    """Parse comma-separated phase weights."""
    parts = weights_str.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"Phase weights must be 3 comma-separated values, got: {weights_str}"
        )
    try:
        r, t, g = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        raise ValueError(f"Invalid phase weight values: {weights_str}")

    total = r + t + g
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Phase weights must sum to 1.0, got {total}: {weights_str}"
        )

    return {
        SpanPhase.REASONING: r,
        SpanPhase.TOOL_USE: t,
        SpanPhase.GENERATION: g,
    }
