"""Control chart report: XmR charts of session measures (point 201).

Reads the :class:`~ter.domain.lean.control.ControlReport` view-model only.
One self-contained page, no scripts and no external requests, light and dark
themes from :mod:`.palette`, like the A3 and the session report.

Each measure gets an individuals chart (sessions in process order, centre
line, natural limits dashed, tuned action limits solid) above its moving
range chart, then its signals. Unfavourable signals are red and carry a
ring, so colour is never the only cue; favourable ones are aqua.
"""

from __future__ import annotations

from collections.abc import Sequence

from ter.domain.lean.control import (
    Basis,
    ControlChart,
    ControlReport,
    ControlRule,
    ControlSignal,
)

from .html import _CSP, _CSS, _PAGE_VARS
from .palette import stylesheet
from .svg import _open, _paint, _text, esc, fmt_tokens, humanise

__all__ = ["render_control_html", "xmr_chart"]

_UNFAVOURABLE = "waste"
_FAVOURABLE = "series-3"
_TUNED = "series-2"
_POINT = "series-1"

_CONTROL_CSS = """
.signals li{margin:0 0 6px}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;
font-weight:600;border:1px solid var(--ter-grid);color:var(--ter-ink-2);
margin-right:6px}
.badge.fire{border-color:var(--ter-waste);color:var(--ter-waste)}
.badge.warn{border-color:var(--ter-series-4);color:var(--ter-ink)}
.measure h2 small{color:var(--ter-muted);font-weight:400;font-size:13px;
margin-left:6px}
.stale{border-left:4px solid var(--ter-series-4)}
"""


def _fmt(value: float | None, basis: Basis) -> str:
    if value is None:
        return "none"
    if basis is Basis.RATIO:
        return f"{value:.1%}"
    if basis is Basis.TOKENS:
        return fmt_tokens(value)
    if basis is Basis.SECONDS:
        return f"{value:,.0f}s"
    return f"{value:.2f}" if value % 1 else f"{value:.0f}"


def _scale(
    low: float, high: float, top: float, height: float
) -> tuple[float, float, float]:
    if high - low < 1e-12:
        low, high = low - 1.0, high + 1.0
    pad = (high - low) * 0.08
    return low - pad, high + pad, height / ((high - low) + 2 * pad)


def _hline(
    x1: float, x2: float, y: float, role: str, *, dash: str | None = None, w: int = 1
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}"'
        f' stroke-width="{w}"{dashed} {_paint(role, "s")}/>'
    )


def xmr_chart(chart: ControlChart, *, width: int = 980) -> str:
    """The individuals chart above the moving range chart, as one SVG."""
    limits = chart.limits
    natural = limits.natural
    basis = chart.measure.basis
    points = chart.points
    left, right = 64, 92
    x_top, x_h = 40, 170
    mr_top, mr_h = x_top + x_h + 34, 70
    height = mr_top + mr_h + 30
    plot_w = width - left - right
    lines = [natural.centre, natural.ucl, natural.lcl, limits.ucl, limits.lcl]
    known = [v for v in lines if v is not None] + [p.value for p in points]
    lo, hi, k = _scale(min(known), max(known), x_top, x_h)

    def y(v: float) -> float:
        return x_top + x_h - (v - lo) * k

    n = max(len(points), 1)

    def x(i: int) -> float:
        return left + (plot_w * (i + 0.5) / n)

    title = chart.measure.label
    flagged = {s.session_id: s for s in chart.signals}
    desc = (
        f"XmR chart of {title} over {len(points)} sessions. Centre "
        f"{_fmt(natural.centre, basis)}, upper limit {_fmt(limits.ucl, basis)}, "
        f"lower limit {_fmt(limits.lcl, basis)}. {len(chart.signals)} signal(s)."
    )
    parts = _open(f"xmr-{chart.measure.key}", title, desc, width, height)
    parts.append(_text(left, 24, "Individuals (X)", role="ink", size=13, weight=600))
    parts.append(_hline(left, width - right, y(natural.centre), "ink-2", w=1))
    parts.append(
        _text(
            width - right + 6,
            y(natural.centre) + 4,
            "CL " + esc(_fmt(natural.centre, basis)),
            size=10,
        )
    )
    for value, label in ((natural.ucl, "UCL"), (natural.lcl, "LCL")):
        if value is not None:
            parts.append(_hline(left, width - right, y(value), "muted", dash="5 4"))
            parts.append(
                _text(
                    width - right + 6,
                    y(value) + 4,
                    f"{label} {esc(_fmt(value, basis))}",
                    role="muted",
                    size=10,
                )
            )
    tuning = limits.tuning
    if tuning is not None:
        for value, label in ((tuning.ucl, "tuned UCL"), (tuning.lcl, "tuned LCL")):
            if value is not None:
                parts.append(_hline(left, width - right, y(value), _TUNED, w=2))
                parts.append(
                    _text(
                        width - right + 6,
                        y(value) - 6,
                        f"{label} {esc(_fmt(value, basis))}",
                        role=_TUNED,
                        size=10,
                    )
                )
    if points:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(p.value):.1f}"
            for i, p in enumerate(points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke-width="1.2" {_paint("baseline", "s")}/>'
        )
    radius = 3.5 if n <= 60 else 2.5
    for i, p in enumerate(points):
        signal = flagged.get(p.session_id)
        role = (
            _POINT
            if signal is None
            else _UNFAVOURABLE
            if signal.unfavourable
            else _FAVOURABLE
        )
        tip = f"{p.session_id}: {_fmt(p.value, basis)} ({p.sigmas:+.1f} sigma)"
        if signal is not None:
            tip += " · " + ", ".join(humanise(s.rule.value) for s in chart.signalled(i))
        parts.append(
            f'<circle cx="{x(i):.1f}" cy="{y(p.value):.1f}" r="{radius}" {_paint(role)}>'
            f"<title>{esc(tip)}</title></circle>"
        )
        if signal is not None and signal.unfavourable:
            parts.append(
                f'<circle cx="{x(i):.1f}" cy="{y(p.value):.1f}" r="{radius + 3}"'
                f' fill="none" stroke-width="1.5" {_paint(_UNFAVOURABLE, "s")}/>'
            )
    # Moving range chart.
    ranges = [p.moving_range for p in points if p.moving_range is not None]
    mr_hi = max([natural.mr_ucl, *ranges]) if ranges else natural.mr_ucl
    mr_k = mr_h / (mr_hi * 1.1) if mr_hi > 0 else 0.0

    def ym(v: float) -> float:
        return mr_top + mr_h - v * mr_k

    parts.append(
        _text(left, mr_top - 10, "Moving range (mR)", role="ink", size=13, weight=600)
    )
    parts.append(_hline(left, width - right, ym(0), "baseline"))
    parts.append(_hline(left, width - right, ym(natural.moving_range), "ink-2"))
    parts.append(
        _text(
            width - right + 6,
            ym(natural.moving_range) + 4,
            "mR " + esc(_fmt(natural.moving_range, basis)),
            size=10,
        )
    )
    parts.append(_hline(left, width - right, ym(natural.mr_ucl), "muted", dash="5 4"))
    parts.append(
        _text(
            width - right + 6,
            ym(natural.mr_ucl) + 4,
            "URL " + esc(_fmt(natural.mr_ucl, basis)),
            role="muted",
            size=10,
        )
    )
    mr_points = [
        (i, p.moving_range) for i, p in enumerate(points) if p.moving_range is not None
    ]
    if mr_points:
        path = " ".join(
            f"{'M' if j == 0 else 'L'}{x(i):.1f},{ym(v):.1f}"
            for j, (i, v) in enumerate(mr_points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke-width="1.2" {_paint(_POINT, "s")}/>'
        )
    parts.append(
        _text(
            left,
            height - 10,
            f"{len(points)} sessions in process order",
            role="muted",
            size=11,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _signal_line(signal: ControlSignal, basis: Basis) -> str:
    badge = (
        '<span class="badge fire">fires</span>'
        if signal.fires
        else '<span class="badge">unfavourable</span>'
        if signal.unfavourable
        else '<span class="badge">favourable</span>'
    )
    limit = (
        "tuned limit"
        if signal.tuned
        else ("centre line" if signal.rule is ControlRule.RUN_OF_EIGHT else "limit")
    )
    evidence = ", ".join(signal.evidence[-8:])
    return (
        f"<li>{badge}<b>{esc(humanise(signal.rule.value))}</b> {esc(signal.side.value)} "
        f"the {limit} {esc(_fmt(signal.limit, basis))} at <code>{esc(signal.session_id)}</code> "
        f"({esc(_fmt(signal.value, basis))}). Evidence: <code>{esc(evidence)}</code></li>"
    )


def _measure_section(chart: ControlChart) -> str:
    limits = chart.limits
    m = chart.measure
    notes: list[str] = []
    if not m.fires:
        notes.append(
            '<span class="badge">charted, never fires</span>'
            f"{esc(humanise(m.basis.value))} are not a structural basis for an intervention."
        )
    if limits.natural.provisional:
        notes.append(
            '<span class="badge warn">provisional</span>'
            f"Limits from {limits.natural.sessions} sessions; they settle at 20 or more."
        )
    if not limits.enabled:
        notes.append('<span class="badge">switched off</span>No signals are raised.')
    if limits.tuning is not None:
        notes.append(
            '<span class="badge warn">tuned</span>' + esc(limits.tuning.reason)
        )
    off = [r for r in ControlRule if r not in limits.rules]
    if off:
        notes.append(
            "Rules off: " + esc(", ".join(humanise(r.value) for r in off)) + "."
        )
    signals = (
        '<ul class="signals">'
        + "".join(_signal_line(s, m.basis) for s in chart.signals)
        + "</ul>"
        if chart.signals
        else '<p class="empty">No signals: every session is inside the limits.</p>'
    )
    return (
        f'<section class="card measure" id="measure-{esc(m.key)}">'
        f"<h2>{esc(m.label)}<small>{esc(m.direction.value.replace('_', ' '))}</small></h2>"
        + "".join(f"<p>{n}</p>" for n in notes)
        + f'<div class="chart">{xmr_chart(chart)}</div>'
        + signals
        + "</section>"
    )


def _kpis(report: ControlReport) -> str:
    tiles = (
        ("Sessions", str(report.sessions), "charted in process order"),
        ("Measures", str(len(report.charts)), "with natural limits"),
        ("Signals", str(len(report.signals)), "rule matches, either direction"),
        ("Would fire", str(len(report.firing)), "unfavourable, structural"),
    )
    return (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi"><span>{esc(a)}</span><b>{esc(b)}</b><small>{esc(c)}</small></div>'
            for a, b, c in tiles
        )
        + "</div>"
    )


def _how_to_read() -> str:
    rules = "".join(
        f"<dt>{esc(humanise(r.value))}</dt><dd>{esc(r.description)}</dd>"
        for r in ControlRule
    )
    return (
        '<footer class="card" id="how-to-read"><h2>How to read these charts</h2><dl>'
        "<dt>Natural limits</dt><dd>What the process does on its own: the mean of "
        "the baseline sessions plus or minus 2.66 average moving ranges (3.145 "
        "median moving ranges). Sessions inside them are routine variation.</dd>"
        "<dt>Tuned limits</dt><dd>Action limits a developer set, with a reason. "
        "They replace the natural limits for the beyond-limits rule only.</dd>"
        f"{rules}"
        "<dt>Fires</dt><dd>An unfavourable signal on a ratio or count measure: the "
        "kind an advisory policy may act on at L4. Token and time totals never "
        "fire.</dd></dl></footer>"
    )


def render_control_html(report: ControlReport, *, title: str = "Control charts") -> str:
    """Render ``report`` as one self-contained HTML document."""
    limits = report.limits
    stale = (
        '<section class="card note stale"><h2>Stale limits</h2><p>These limits '
        f"were computed with detector set <code>{esc(limits.detectors)}</code>; "
        f"the sessions were measured with <code>{esc(report.detectors)}</code>. "
        "Finding counts may mean something different now: recompute the "
        "limits.</p></section>"
        if report.stale
        else ""
    )
    computed = f" · computed {esc(limits.computed_on)}" if limits.computed_on else ""
    sections: Sequence[str] = [_measure_section(c) for c in report.charts]
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">',
        '<meta name="color-scheme" content="light dark">',
        f"<title>{esc(title)}</title>",
        f"<style>\n{stylesheet('.ter-chart')}\n{_PAGE_VARS}\n{_CSS}{_CONTROL_CSS}</style>",
        "</head>",
        '<body><main class="ter-chart">',
        '<header><p class="eyebrow">TER process control</p>'
        f"<h1>{esc(title)}</h1>"
        f'<p class="meta">{report.sessions} sessions · '
        f"{esc(humanise(limits.method.value))} · limits from "
        f"{limits.sessions} baseline sessions{computed} · detectors "
        f"<code>{esc(report.detectors)}</code></p></header>",
        _kpis(report),
        stale,
        *sections,
        _how_to_read(),
        "</main></body>",
        "</html>",
    ]
    return "\n".join(p for p in parts if p) + "\n"
