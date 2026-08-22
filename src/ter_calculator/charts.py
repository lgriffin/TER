"""Inline SVG chart generation for TER analysis results.

Produces standalone SVG strings with no external dependencies.
Uses a validated categorical palette following dataviz accessibility guidelines.
"""

from __future__ import annotations

import html
from collections import Counter

from .models import TERResult

# Validated categorical palette (light mode) — fixed order, CVD-safe adjacent pairs.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_SURFACE = "#fcfcfb"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def _esc(text: str) -> str:
    return html.escape(str(text))


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _fmt_cost(val: float) -> str:
    return f"${val:.4f}"


# ---------------------------------------------------------------------------
# Horizontal stacked bar
# ---------------------------------------------------------------------------


def _stacked_bar_svg(
    title: str,
    segments: list[tuple[str, int, str]],
    width: int = 640,
    bar_height: int = 42,
) -> str:
    """Horizontal stacked bar chart. segments: [(label, value, color), ...]."""
    total = sum(v for _, v, _ in segments)
    if total == 0:
        return ""

    top_margin = 36
    bottom_margin = 56
    left_margin = 16
    right_margin = 16
    bar_width = width - left_margin - right_margin
    height = top_margin + bar_height + bottom_margin

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
        f' width="{width}" height="{height}" role="img"'
        f' aria-label="{_esc(title)}">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{_SURFACE}" rx="8"/>'
    )
    parts.append(
        f'<text x="{left_margin}" y="24" fill="{_INK_PRIMARY}"'
        f' font-family="system-ui,sans-serif" font-size="15" font-weight="600">'
        f"{_esc(title)}</text>"
    )

    x = float(left_margin)
    gap = 2
    for i, (label, value, color) in enumerate(segments):
        w = (value / total) * bar_width - (gap if i < len(segments) - 1 else 0)
        if w < 1:
            x += w + gap
            continue
        parts.append(
            f'<rect x="{x:.1f}" y="{top_margin}" width="{max(w, 0):.1f}"'
            f' height="{bar_height}" rx="4" fill="{color}">'
            f"<title>{_esc(label)}: {_fmt_tokens(value)}</title></rect>"
        )
        if w > 50:
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{top_margin + bar_height / 2 + 5}"'
                f' text-anchor="middle" fill="#fff"'
                f' font-family="system-ui,sans-serif" font-size="12" font-weight="500">'
                f"{_fmt_pct(value / total)}</text>"
            )
        x += w + gap

    legend_y = top_margin + bar_height + 18
    lx = float(left_margin)
    for label, value, color in segments:
        parts.append(
            f'<rect x="{lx:.1f}" y="{legend_y}" width="10" height="10" rx="2"'
            f' fill="{color}"/>'
        )
        text = f"{_esc(label)} ({_fmt_tokens(value)})"
        parts.append(
            f'<text x="{lx + 14:.1f}" y="{legend_y + 9}" fill="{_INK_SECONDARY}"'
            f' font-family="system-ui,sans-serif" font-size="11">{text}</text>'
        )
        lx += len(text) * 6.2 + 30

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Horizontal bar chart (magnitude comparison)
# ---------------------------------------------------------------------------


def _horizontal_bar_svg(
    title: str,
    items: list[tuple[str, float, str]],
    width: int = 640,
    bar_height: int = 28,
    format_value=None,
) -> str:
    """Horizontal bar chart. items: [(label, value, color), ...]."""
    if not items:
        return ""

    format_value = format_value or (lambda v: f"{v:.2f}")
    max_val = max(v for _, v, _ in items) or 1.0

    top_margin = 36
    label_width = 140
    right_margin = 70
    row_gap = 8
    chart_width = width - label_width - right_margin
    bar_area_height = len(items) * (bar_height + row_gap)
    height = top_margin + bar_area_height + 8

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
        f' width="{width}" height="{height}" role="img"'
        f' aria-label="{_esc(title)}">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{_SURFACE}" rx="8"/>'
    )
    parts.append(
        f'<text x="16" y="24" fill="{_INK_PRIMARY}"'
        f' font-family="system-ui,sans-serif" font-size="15" font-weight="600">'
        f"{_esc(title)}</text>"
    )

    for i, (label, value, color) in enumerate(items):
        y = top_margin + i * (bar_height + row_gap)
        bar_w = (value / max_val) * chart_width if max_val > 0 else 0

        parts.append(
            f'<text x="{label_width - 8}" y="{y + bar_height / 2 + 4}"'
            f' text-anchor="end" fill="{_INK_SECONDARY}"'
            f' font-family="system-ui,sans-serif" font-size="12">'
            f"{_esc(label)}</text>"
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{max(bar_w, 2):.1f}"'
            f' height="{bar_height}" rx="4" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{label_width + bar_w + 6:.1f}" y="{y + bar_height / 2 + 4}"'
            f' fill="{_INK_PRIMARY}"'
            f' font-family="system-ui,sans-serif" font-size="12" font-weight="500">'
            f"{_esc(format_value(value))}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stat tiles row
# ---------------------------------------------------------------------------


def _stat_tile_svg(
    metrics: list[tuple[str, str]],
    width: int = 640,
) -> str:
    """Row of stat tiles. metrics: [(label, value_str), ...]."""
    if not metrics:
        return ""

    tile_count = len(metrics)
    tile_w = width // tile_count
    height = 80

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
        f' width="{width}" height="{height}" role="img" aria-label="Key metrics">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{_SURFACE}" rx="8"/>'
    )

    for i, (label, value) in enumerate(metrics):
        x = i * tile_w + tile_w // 2
        parts.append(
            f'<text x="{x}" y="28" text-anchor="middle" fill="{_INK_MUTED}"'
            f' font-family="system-ui,sans-serif" font-size="12">'
            f"{_esc(label)}</text>"
        )
        parts.append(
            f'<text x="{x}" y="58" text-anchor="middle" fill="{_INK_PRIMARY}"'
            f' font-family="system-ui,sans-serif" font-size="25" font-weight="700">'
            f"{_esc(value)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public chart generators from TERResult
# ---------------------------------------------------------------------------


def chart_key_metrics(result: TERResult) -> str:
    """Stat tile row: TER, total tokens, waste %, cost."""
    waste_pct = (
        f"{result.waste_tokens / result.total_tokens * 100:.1f}%"
        if result.total_tokens
        else "0%"
    )
    metrics = [
        ("TER Score", f"{result.aggregate_ter:.2f}"),
        ("Total Tokens", _fmt_tokens(result.total_tokens)),
        ("Waste", waste_pct),
    ]
    if result.economics:
        metrics.append(("Est. Cost", _fmt_cost(result.economics.estimated_cost_usd)))
    if result.uncertainty:
        metrics.append(("Reliability", result.uncertainty.reliability))
    return _stat_tile_svg(metrics)


def chart_composition(result: TERResult) -> str:
    """Stacked bar: token composition by classification label."""
    label_tokens: Counter[str] = Counter()
    for item in result.classified_spans:
        label_tokens[item.label.value] += item.span.token_count

    label_colors = {
        "aligned_reasoning": PALETTE[0],
        "aligned_tool_call": PALETTE[1],
        "aligned_response": PALETTE[2],
        "redundant_reasoning": PALETTE[3],
        "unnecessary_tool_call": PALETTE[4],
        "over_explanation": PALETTE[5],
    }

    segments = []
    for label_val, tokens in label_tokens.most_common():
        color = label_colors.get(label_val, PALETTE[6])
        name = label_val.replace("_", " ").title()
        segments.append((name, tokens, color))

    return _stacked_bar_svg("Token Composition", segments)


def chart_phase_scores(result: TERResult) -> str:
    """Horizontal bar: per-phase TER scores."""
    items = []
    phase_colors = {
        "reasoning": PALETTE[0],
        "tool_use": PALETTE[1],
        "generation": PALETTE[2],
    }
    for phase, score in result.phase_scores.items():
        color = phase_colors.get(phase, PALETTE[3])
        items.append((phase.replace("_", " ").title(), score, color))

    return _horizontal_bar_svg(
        "Phase Scores",
        items,
        format_value=lambda v: f"{v:.3f}",
    )


def chart_waste_patterns(result: TERResult) -> str:
    """Horizontal bar: waste patterns ranked by tokens wasted."""
    if not result.waste_patterns:
        return ""

    by_type: dict[str, int] = {}
    for wp in result.waste_patterns:
        label = wp.pattern_type.replace("_", " ").title()
        by_type[label] = by_type.get(label, 0) + wp.tokens_wasted

    sorted_items = sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:6]
    items = [
        (label, float(tokens), PALETTE[i % len(PALETTE)])
        for i, (label, tokens) in enumerate(sorted_items)
    ]

    return _horizontal_bar_svg(
        "Waste Patterns",
        items,
        format_value=lambda v: _fmt_tokens(int(v)),
    )


def chart_positional_ter(result: TERResult) -> str:
    """Horizontal bar: early/mid/late TER from economics positional breakdown."""
    if not result.economics:
        return ""

    pos = result.economics.positional
    items = [
        ("Early", pos.early_ter, PALETTE[0]),
        ("Mid", pos.mid_ter, PALETTE[1]),
        ("Late", pos.late_ter, PALETTE[2]),
    ]
    return _horizontal_bar_svg(
        "Positional TER (Session Thirds)",
        items,
        format_value=lambda v: f"{v:.3f}",
    )


def chart_economics(result: TERResult) -> str:
    """Stacked bar: token economics (input/output/cache)."""
    if not result.economics:
        return ""

    e = result.economics
    segments = [
        ("Output Tokens", e.total_output_tokens, PALETTE[0]),
        ("Input Tokens", e.total_input_tokens, PALETTE[1]),
        ("Cache Read", e.total_cache_read_tokens, PALETTE[2]),
        ("Cache Write", e.total_cache_creation_tokens, PALETTE[3]),
    ]
    segments = [(l, v, c) for l, v, c in segments if v > 0]
    return _stacked_bar_svg("Token Economics", segments)


def chart_waste_breakdown(result: TERResult) -> str:
    """Stacked bar: aligned vs waste tokens."""
    segments = [
        ("Aligned", result.aligned_tokens, PALETTE[0]),
        ("Waste", result.waste_tokens, PALETTE[1]),
    ]
    return _stacked_bar_svg("Aligned vs Waste Tokens", segments)


def generate_all_charts(result: TERResult) -> dict[str, str]:
    """Generate all available charts, returning {name: svg_string}."""
    charts = {}
    charts["key_metrics"] = chart_key_metrics(result)
    charts["waste_breakdown"] = chart_waste_breakdown(result)

    if result.classified_spans:
        charts["composition"] = chart_composition(result)

    charts["phase_scores"] = chart_phase_scores(result)

    if result.waste_patterns:
        charts["waste_patterns"] = chart_waste_patterns(result)

    if result.economics:
        charts["positional_ter"] = chart_positional_ter(result)
        charts["economics"] = chart_economics(result)

    return {k: v for k, v in charts.items() if v}
