"""The A3 report: one self-contained HTML page in Toyota A3 order.

Reads only the :class:`~ter.domain.lean.a3.A3Report` view-model. Like the
session report, it carries no scripts and makes no requests (a CSP forbids
them), every chart is an accessible SVG (``role="img"``, ``<title>``,
``<desc>``), colours come from :mod:`.palette` and switch with the reader's
light or dark theme, and every session-derived string is escaped. It prints
on one landscape A3 sheet.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ter.domain.lean import (
    LEAN_MEASURES,
    SVE_DEFINITION,
    A3Report,
    ActivityClass,
    Countermeasure,
    Finding,
    FindingKind,
    FlowState,
    Measure,
    StageSummary,
    ValueStatus,
    WipKind,
)
from ter.domain.events import describe_limit
from ter.domain.lean.a3 import A3_SCHEMA
from ter.domain.lean.countermeasures import Action, ActionKind
from ter.domain.lean.model import UNCERTAIN_BELOW, Stage
from ter.domain.lean.usage import ReadUsage, UsageStatus
from ter.domain.lean.value import JudgedKind, ValueClass
from ter.domain.report import WasteByType
from ter.domain.outcome import CheckResult

from .palette import stylesheet
from .svg import (
    _fill_stroke,
    _heading,
    _legend,
    _open,
    _paint,
    _text,
    esc,
    fmt_pct,
    fmt_tokens,
    stacked_bar,
    waste_pareto,
)

__all__ = [
    "activity_bar",
    "flow_bar",
    "fmt_seconds",
    "render_a3_html",
    "value_stream_map",
    "wip_chart",
]

_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"

#: Colour role of each activity class; uncertain is a separate, hatched slot.
ACTIVITY_ROLES: dict[str, str] = {
    ActivityClass.VALUE_ADDING.value: "series-1",
    ActivityClass.NECESSARY_NON_VALUE_ADDING.value: "series-3",
    ActivityClass.AVOIDABLE.value: "waste",
    "uncertain": "series-4",
}

#: Colour role of each kind of work in progress.
WIP_ROLES: dict[WipKind, str] = {
    WipKind.HYPOTHESES: "series-2",
    WipKind.TASKS: "series-7",
    WipKind.EDITS: "series-1",
    WipKind.FAILURES: "waste",
}

FLOW_ROLES: dict[FlowState, str] = {
    FlowState.PROGRESSING: "series-1",
    FlowState.RECOVERING: "series-3",
    FlowState.REPEATING: "series-2",
    FlowState.REWORKING: "waste",
    FlowState.WAITING: "series-7",
    FlowState.INVENTORY: "series-4",
}


def fmt_seconds(seconds: float) -> str:
    """45s, 3m 20s, 1h 05m."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _hatch(cid: str) -> str:
    return (
        f'<defs><pattern id="{cid}-hatch" width="6" height="6"'
        ' patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff" stroke-width="2"'
        ' stroke-opacity="0.6"/></pattern></defs>'
    )


def value_stream_map(
    stages: Sequence[StageSummary],
    *,
    title: str = "Agentic value stream",
    width: int = 1040,
    chart_id: str = "value-stream",
) -> str:
    """Stages as process boxes, left to right, with tokens, time and waste.

    A red outline and badge mark stages where findings claimed waste; the bar
    in each box shows the stage's avoidable share (red) and uncertain share
    (hatched yellow) of its generated tokens.
    """
    if not stages:
        return ""
    side, gap, top = 16, 26, 58
    box_w = (width - 2 * side - gap * (len(stages) - 1)) / len(stages)
    box_h = 148
    height = top + box_h + 44
    total_tokens = sum(s.tokens for s in stages)
    denominator = total_tokens or 1  # for shares only; the total shown stays real
    desc = "Value stream, in order. " + "; ".join(
        f"{s.stage.label}: {s.steps} steps, {s.tokens} generated tokens, "
        f"{s.context_tokens} context tokens, {fmt_seconds(s.seconds)}, "
        f"{s.avoidable_tokens} avoidable and {s.uncertain_tokens} uncertain tokens, "
        f"{len(s.findings)} findings"
        for s in stages
    )
    cid = esc(chart_id)
    parts = _open(chart_id, title, desc, width, height)
    parts.append(_hatch(cid))
    parts.append(
        '<defs><marker id="' + cid + '-arrow" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" {_paint("muted")}/></marker></defs>'
    )
    parts.append(_text(side, 24, esc(title), role="ink", size=15, weight=600))
    legend_y = 36
    parts.append(
        f'<rect x="{side}" y="{legend_y}" width="10" height="10" rx="2" {_paint("waste")}/>'
    )
    parts.append(_text(side + 14, legend_y + 9, "Avoidable (confident)", size=11))
    ux = side + 14 + 30 * 6.2 + 16
    parts.append(
        f'<rect x="{ux:.1f}" y="{legend_y}" width="10" height="10" rx="2" {_paint("series-4")}/>'
        f'<rect x="{ux:.1f}" y="{legend_y}" width="10" height="10" rx="2" fill="url(#{cid}-hatch)"/>'
    )
    parts.append(_text(ux + 14, legend_y + 9, "Uncertain (verify)", size=11))
    parts.append(
        _text(
            width - side,
            legend_y + 9,
            "developer intent → verified response",
            role="muted",
            size=11,
            anchor="end",
        )
    )
    for i, s in enumerate(stages):
        x = side + i * (box_w + gap)
        flagged = s.avoidable_tokens > 0
        stroke = "waste" if flagged else "baseline"
        stroke_w = 2 if flagged else 1
        tip = (
            f"{s.stage.label}: {s.steps} steps, {s.tokens} generated and "
            f"{s.context_tokens} context tokens, {fmt_seconds(s.seconds)}; "
            f"findings: {', '.join(s.findings) or 'none'}"
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{top}" width="{box_w:.1f}" height="{box_h}" rx="8"'
            f' stroke-width="{stroke_w}" {_fill_stroke("surface", stroke)}>'
            f"<title>{esc(tip)}</title></rect>"
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{top}" width="{box_w:.1f}" height="26" rx="8" {_paint("grid")}/>'
            f'<rect x="{x:.1f}" y="{top + 18}" width="{box_w:.1f}" height="8" {_paint("grid")}/>'
        )
        parts.append(
            _text(x + 10, top + 18, esc(s.stage.label), role="ink", size=13, weight=700)
        )
        if s.findings:
            bx = x + box_w - 16
            role = "waste" if flagged else "muted"
            parts.append(
                f'<circle cx="{bx:.1f}" cy="{top + 13}" r="9" {_paint(role)}>'
                f"<title>{len(s.findings)} finding(s) touch this stage</title></circle>"
            )
            parts.append(
                f'<text x="{bx:.1f}" y="{top + 17}" text-anchor="middle" font-size="11"'
                f' font-weight="700" fill="#ffffff">{len(s.findings)}</text>'
            )
        if s.stage is Stage.INTENT:
            rows = [
                (f"{s.steps} prompt{'s' if s.steps != 1 else ''}", "ink"),
                ("developer input,", "muted"),
                ("not scored", "muted"),
                (f"{fmt_seconds(s.seconds)} developer", "ink-2"),
            ]
        else:
            rows = [
                (f"{s.steps} step{'s' if s.steps != 1 else ''}", "ink"),
                (f"{fmt_tokens(s.tokens)} generated", "ink-2"),
                (f"{fmt_tokens(s.context_tokens)} context", "ink-2"),
                (
                    f"{fmt_seconds(s.seconds)} · {fmt_pct(s.tokens / denominator, 0)} of tokens",
                    "ink-2",
                ),
            ]
        for j, (text, role) in enumerate(rows):
            parts.append(
                _text(
                    x + 10,
                    top + 46 + j * 17,
                    esc(text),
                    role=role,
                    size=12,
                    weight=600 if j == 0 else None,
                )
            )
        # Waste bar.
        by = top + box_h - 20
        bw = box_w - 20
        parts.append(
            f'<rect x="{x + 10:.1f}" y="{by}" width="{bw:.1f}" height="8" rx="4" {_paint("grid")}/>'
        )
        if s.tokens > 0 and s.stage is not Stage.INTENT:
            a = s.avoidable_tokens / s.tokens * bw
            u = s.uncertain_tokens / s.tokens * bw
            if a > 0:
                parts.append(
                    f'<rect x="{x + 10:.1f}" y="{by}" width="{max(a, 3):.1f}" height="8" rx="4" {_paint("waste")}>'
                    f"<title>{s.avoidable_tokens} avoidable tokens</title></rect>"
                )
            if u > 0:
                ux0 = x + 10 + a
                parts.append(
                    f'<rect x="{ux0:.1f}" y="{by}" width="{max(u, 3):.1f}" height="8" rx="4" {_paint("series-4")}>'
                    f"<title>{s.uncertain_tokens} uncertain tokens</title></rect>"
                    f'<rect x="{ux0:.1f}" y="{by}" width="{max(u, 3):.1f}" height="8" rx="4"'
                    f' fill="url(#{cid}-hatch)" pointer-events="none"/>'
                )
            label = (
                f"waste {fmt_pct(s.avoidable_tokens / s.tokens, 0)}"
                if s.avoidable_tokens
                else ("uncertain only" if s.uncertain_tokens else "no waste found")
            )
            parts.append(_text(x + 10, by - 5, esc(label), role="muted", size=10))
        if i < len(stages) - 1:
            ay = top + box_h / 2
            parts.append(
                f'<line x1="{x + box_w + 3:.1f}" y1="{ay}" x2="{x + box_w + gap - 3:.1f}" y2="{ay}"'
                f' stroke-width="2" marker-end="url(#{cid}-arrow)" {_paint("muted", "s")}/>'
            )
    total_s = sum(s.seconds for s in stages if s.stage is not Stage.INTENT)
    parts.append(
        _text(
            side,
            top + box_h + 26,
            esc(
                f"Agent lead time {fmt_seconds(total_s)} · {fmt_tokens(total_tokens)} generated tokens · "
                "time is the wall-clock gap each event closed (tool time counts toward its request)"
            ),
            role="muted",
            size=11,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def activity_bar(report: A3Report, *, width: int = 520) -> str:
    """Generated tokens by Lean activity class, as a 100% bar."""
    sc = report.analysis.scorecard
    labels = {
        ActivityClass.VALUE_ADDING.value: ActivityClass.VALUE_ADDING.label,
        ActivityClass.NECESSARY_NON_VALUE_ADDING.value: "Necessary NVA",
        ActivityClass.AVOIDABLE.value: ActivityClass.AVOIDABLE.label,
        "uncertain": "Uncertain",
    }
    segments = [
        (labels[k], float(n), ACTIVITY_ROLES[k]) for k, n in sc.activity_tokens if n > 0
    ]
    return stacked_bar(
        "Activity classes (generated tokens)",
        segments,
        width=width,
        chart_id="a3-activity",
    )


def flow_bar(report: A3Report, *, time: bool = False, width: int = 520) -> str:
    """Generated tokens (or agent time) by flow state, as a 100% bar."""
    sc = report.analysis.scorecard
    pairs: Sequence[tuple[FlowState, float]] = (
        sc.flow_seconds if time else [(f, float(n)) for f, n in sc.flow_tokens]
    )
    segments = [(f.label, v, FLOW_ROLES[f]) for f, v in pairs if v > 0]
    what = "agent time, seconds" if time else "generated tokens"
    return stacked_bar(
        f"Flow ({what})",
        segments,
        width=width,
        chart_id="a3-flow-time" if time else "a3-flow",
    )


def wip_chart(report: A3Report, *, width: int = 520) -> str:
    """Unresolved work after every event, stacked by kind, with the peak marked."""
    wip = report.analysis.wip
    peak = wip.peak
    if peak is None or peak.total == 0:
        return ""
    side, top, plot_h = 16, 44, 110
    plot_w = width - 2 * side
    n = len(wip.samples)
    step = plot_w / n
    bar = max(step - (1 if step > 3 else 0), 0.5)
    scale = plot_h / peak.total
    base = top + plot_h
    legend, bottom = _legend(
        [(f"{k.label} (peak {wip.peak_of(k)})", WIP_ROLES[k]) for k in WipKind],
        x0=side,
        y0=base + 14,
        max_x=width - side,
    )
    height = int(bottom + 14)
    peaks = ", ".join(f"{k.value} {wip.peak_of(k)}" for k in WipKind)
    desc = (
        f"Work in progress after each of {n} events. Peak {peak.total} open items "
        f"after event {peak.event_id}; peak by kind: {peaks}."
    )
    parts = _open("a3-wip", "Work in progress", desc, width, height)
    parts.append(_heading(f"Work in progress (peak {peak.total})", side))
    for i, sample in enumerate(wip.samples):
        y = float(base)
        for kind in WipKind:
            count = sample.count(kind)
            if not count:
                continue
            h = count * scale
            y -= h
            parts.append(
                f'<rect x="{side + i * step:.1f}" y="{y:.1f}" width="{bar:.1f}"'
                f' height="{h:.1f}" {_paint(WIP_ROLES[kind])}>'
                f"<title>Event {esc(sample.event_id)}: {count} {kind.value}</title></rect>"
            )
    parts.append(
        f'<line x1="{side}" y1="{top}" x2="{width - side}" y2="{top}"'
        f' stroke-dasharray="4 3" {_paint("muted", "s")}/>'
    )
    parts.append(
        _text(width - side, top - 4, f"peak {peak.total}", role="muted", anchor="end")
    )
    parts.append(
        f'<line x1="{side}" y1="{base}" x2="{width - side}" y2="{base}"'
        f" {_paint('baseline', 's')}/>"
    )
    parts.extend(legend)
    parts.append("</svg>")
    return "\n".join(parts)


def pareto(report: A3Report, *, width: int = 520) -> str:
    entries = [WasteByType(p.waste.value, p.tokens, p.findings) for p in report.pareto]
    return waste_pareto(
        entries,
        title="Waste Pareto (generated tokens, uncertain included)",
        width=width,
        chart_id="a3-pareto",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_CSS = """
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;background:var(--ter-page);color:var(--ter-ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:24px 24px 40px}
a{color:var(--ter-series-1);text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--ter-series-1);outline-offset:2px;border-radius:4px}
.skip{position:absolute;left:-9999px;top:8px;z-index:3;background:var(--ter-surface);padding:6px 10px;
border-radius:6px}
.skip:focus{left:8px}
header.a3{border-bottom:3px solid var(--ter-ink);padding-bottom:12px;margin-bottom:0}
.eyebrow{margin:0;color:var(--ter-muted);font-size:12px;letter-spacing:.08em;
text-transform:uppercase;font-weight:700}
h1{margin:2px 0 8px;font-size:24px;line-height:1.25;max-width:1000px;overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0;padding:0;list-style:none}
.chips li{border:1px solid var(--ter-grid);background:var(--ter-surface);border-radius:999px;
padding:1px 10px;font-size:12.5px;color:var(--ter-ink-2);max-width:100%;overflow-wrap:anywhere}
.chips li b{color:var(--ter-ink)}
.chips .level{border-color:var(--ter-ink);color:var(--ter-ink);font-weight:600}
.verdict{font-weight:700}
.verdict.ok{border-color:var(--ter-series-3)}
.verdict.bad{border-color:var(--ter-waste)}
.verdict.unsure{border-color:var(--ter-series-4)}
nav.toc{position:sticky;top:0;z-index:2;background:var(--ter-page);margin:0 0 14px;
padding:8px 0;border-bottom:1px solid var(--ter-grid)}
nav.toc ol{display:flex;gap:4px 14px;flex-wrap:wrap;margin:0;padding:0;list-style:none;font-size:13px}
nav.toc a{color:var(--ter-ink-2);text-decoration:none;white-space:nowrap}
nav.toc a:hover{color:var(--ter-ink);text-decoration:underline}
nav.toc b{display:inline-block;min-width:1.4em;color:var(--ter-muted)}
code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}
.sheet{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start}
.box{background:var(--ter-surface);border:1px solid var(--ter-grid);border-radius:10px;
padding:14px 16px;min-width:0}
.full{grid-column:1/-1}
.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px 24px;align-items:start}
.split>*{min-width:0}
h2{margin:0 0 10px;font-size:15px;display:flex;align-items:center;gap:10px}
h2 .n{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;
border-radius:50%;background:var(--ter-ink);color:var(--ter-surface);font-size:13px;flex:none}
h3{margin:14px 0 6px;font-size:13.5px}
.split>div>h3:first-child{margin-top:0}
p{margin:0 0 8px}
.problem{border-left:4px solid var(--ter-waste);padding:6px 10px;margin:8px 0;
background:var(--ter-page);border-radius:0 6px 6px 0}
blockquote{margin:0 0 8px;padding:8px 12px;border-left:3px solid var(--ter-series-1);
background:var(--ter-page);border-radius:0 6px 6px 0;color:var(--ter-ink-2);white-space:pre-wrap;
max-height:9.5em;overflow:auto;overflow-wrap:anywhere}
.summary{border:2px solid var(--ter-ink)}
.summary .problem{font-size:15px;margin-top:0}
.hero{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:4px 0 6px}
.hero div{border:1px solid var(--ter-grid);border-radius:8px;padding:10px 12px;min-width:0}
.hero span{display:block;color:var(--ter-muted);font-size:12px}
.hero b{display:block;font-size:28px;line-height:1.15;margin:2px 0 6px;font-variant-numeric:tabular-nums}
.hero small{display:block;color:var(--ter-ink-2);font-size:12px;line-height:1.35;margin-top:6px}
.meter{display:block;height:8px;border-radius:4px;background:var(--ter-grid);overflow:hidden}
.meter i{display:block;height:100%;border-radius:4px;background:var(--ter-series-1)}
.meter.waste i{background:var(--ter-waste)}
.meter.flow i{background:var(--ter-series-3)}
.meter.ter i{background:var(--ter-series-7)}
.meter.warn i{background:var(--ter-series-4)}
.meter.thin{height:5px;width:72px;display:inline-block;vertical-align:middle;margin-left:6px}
.top{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:0;padding:0;list-style:none}
.top li{border:1px solid var(--ter-grid);border-left:4px solid var(--ter-series-1);border-radius:8px;
padding:10px 12px;min-width:0;display:flex;flex-direction:column;gap:4px}
.top li.risk{border-left-color:var(--ter-series-7)}
.top li.verify{border-left-color:var(--ter-series-4)}
.top .rank{font-size:12px;color:var(--ter-muted);font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.top a.t{font-weight:700;color:var(--ter-ink);text-decoration:none}
.top a.t:hover{text-decoration:underline}
.top .impact{font-size:12.5px;color:var(--ter-ink-2)}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.kpi{border:1px solid var(--ter-grid);border-radius:8px;padding:8px 12px;min-width:0}
.kpi span{display:block;color:var(--ter-muted);font-size:12px}
.kpi b{display:block;font-size:21px;line-height:1.2;margin:2px 0;font-variant-numeric:tabular-nums}
.kpi small{display:block;color:var(--ter-ink-2);font-size:12px;line-height:1.35}
.chart{overflow-x:auto}
.chart svg{display:block;width:100%;height:auto}
.chart.wide svg{min-width:760px}
.charts{display:grid;gap:12px}
figure{margin:0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--ter-grid);vertical-align:top}
th{color:var(--ter-ink-2);font-weight:600;font-size:12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td small,th small{display:block;color:var(--ter-ink-2);font-weight:400}
.tag{display:inline-block;border-radius:999px;padding:0 8px;font-size:11.5px;font-weight:600;
border:1px solid var(--ter-grid);white-space:nowrap;vertical-align:1px}
.tag.warn{border-color:var(--ter-series-4);background:color-mix(in srgb,var(--ter-series-4) 18%,transparent)}
.tag.risk{border-color:var(--ter-series-7)}
.tag.waste{border-color:var(--ter-waste)}
.tag.ok{border-color:var(--ter-series-3)}
.tag.kind{border-color:var(--ter-baseline);color:var(--ter-ink-2);font-weight:600}
.ev{margin-top:6px;font-size:11.5px;color:var(--ter-muted)}
.ev code{display:inline-block;margin:0 4px 2px 0;padding:0 4px;border-radius:4px;
background:var(--ter-page);font-size:11.5px}
.ev summary{display:inline;font-size:11.5px}
.findings{display:grid;gap:10px;margin:0;padding:0;list-style:none}
.finding{border:1px solid var(--ter-grid);border-left:4px solid var(--ter-waste);border-radius:8px;
padding:10px 12px;min-width:0}
.finding.risk{border-left-color:var(--ter-series-7)}
.finding.unsure{border-left-color:var(--ter-series-4);border-left-style:dashed}
.finding:target,.cm:target{box-shadow:0 0 0 3px color-mix(in srgb,var(--ter-series-1) 45%,transparent)}
.fh{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.fh .rk{color:var(--ter-muted);font-variant-numeric:tabular-nums;font-weight:700}
.fh b{flex:1 1 220px;overflow-wrap:anywhere}
.fh .cost{font-size:12.5px;color:var(--ter-ink-2);font-variant-numeric:tabular-nums;white-space:nowrap}
.finding p{margin:4px 0 0;color:var(--ter-ink-2);font-size:12.5px;overflow-wrap:anywhere}
.fm{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;margin-top:6px;font-size:12px;color:var(--ter-ink-2)}
.intent{margin:0 0 8px;padding-left:20px}
.intent li{margin:0 0 6px}
.intent small{display:block;color:var(--ter-ink-2)}
.cms{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;align-items:start}
.cm{border:1px solid var(--ter-grid);border-radius:8px;padding:12px 14px;min-width:0}
.cm.verify{border-style:dashed;border-color:var(--ter-series-4)}
.cm h3{margin:0 0 4px;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.cm h3 .n{display:inline-flex;align-items:center;justify-content:center;min-width:22px;height:22px;
border-radius:6px;background:var(--ter-series-1);color:var(--ter-on-series-1);font-size:12px;flex:none}
.cm.verify h3 .n{background:var(--ter-series-4);color:var(--ter-on-series-4)}
.cm .impact{font-size:12.5px;color:var(--ter-ink-2);margin:0 0 6px}
.cm .why{color:var(--ter-ink-2);font-size:12.5px}
.cm .answers{margin:0 0 6px;padding-left:18px;font-size:12.5px}
.cm ol.acts{margin:8px 0 0;padding:0;list-style:none;display:grid;gap:10px}
.cm ol.acts>li{border-top:1px solid var(--ter-grid);padding-top:8px}
.kind{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--ter-muted)}
.snip{margin-top:6px;border:1px solid var(--ter-grid);border-radius:6px;overflow:hidden}
.snip .file{display:block;padding:3px 10px;background:var(--ter-grid);color:var(--ter-ink-2);
font:600 11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{margin:0;padding:8px 10px;background:var(--ter-page);
overflow:auto;max-height:16em;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:pre-wrap;overflow-wrap:anywhere}
details{margin-top:4px}
summary{cursor:pointer;color:var(--ter-ink-2);font-size:12.5px}
.empty{color:var(--ter-muted);margin:0}
.fine{color:var(--ter-muted);font-size:12px}
dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:4px 14px;margin:0;font-size:12.5px}
dt{font-weight:600}
dd{margin:0;color:var(--ter-ink-2);overflow-wrap:anywhere}
.files{margin:0;padding:0;list-style:none;font-size:12.5px;display:grid;gap:2px}
.files li{display:flex;gap:8px;align-items:baseline;min-width:0}
.files code{overflow-wrap:anywhere}
.mark{flex:none;width:1.1em;text-align:center;font-weight:700}
.mark.yes{color:var(--ter-series-3)}
.mark.no{color:var(--ter-waste)}
.target{display:grid;grid-template-columns:auto auto auto;gap:0 6px;align-items:baseline;justify-content:end}
@media (max-width:1000px){
.hero{grid-template-columns:repeat(2,minmax(0,1fr))}
.top{grid-template-columns:minmax(0,1fr)}
}
@media (max-width:900px){
main{padding:16px 12px 28px}
.sheet,.cms,.split{grid-template-columns:minmax(0,1fr)}
h1{font-size:21px}
.box{padding:12px}
nav.toc ol{flex-wrap:nowrap;overflow-x:auto}
}
@media (max-width:480px){
.hero b{font-size:23px}
.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@page{size:A3 landscape;margin:10mm}
@media print{
body{background:#fff;font-size:11px}
main{max-width:none;padding:0}
nav.toc,.skip{display:none}
.box,.cm,.finding,.top li{break-inside:avoid}
.chart{overflow:visible}
.chart.wide svg{min-width:0}
pre,blockquote{max-height:none}
details>summary{display:none}
}
"""

_PAGE_VARS = (
    ":root{--ter-page:#f3f2ee}"
    "@media (prefers-color-scheme: dark){:root:not([data-theme=light])"
    "{--ter-page:#0f0f0e}}"
    ":root[data-theme=dark]{--ter-page:#0f0f0e}"
)

#: Where each kind of action lands in the developer's setup.
ACTION_TARGETS: dict[ActionKind, str] = {
    ActionKind.CLAUDE_MD: "CLAUDE.md",
    ActionKind.HOOK: ".claude/settings.json",
    ActionKind.SETTING: ".claude/settings.json",
    ActionKind.PRACTICE: "way of working",
}

#: Top countermeasures shown in the summary.
TOP_ACTIONS = 3


def _slug(text: str) -> str:
    return esc("".join(c if c.isalnum() or c in "-_:." else "-" for c in text))


def _box(
    number: int | None, title: str, body: str, css: str = "", anchor: str = ""
) -> str:
    badge = f'<span class="n">{number}</span>' if number is not None else ""
    sid = anchor or (
        f"s-{number}" if number is not None else f"s-{_slug(title.lower())}"
    )
    return (
        f'<section class="box {css}" id="{sid}" aria-labelledby="{sid}-h">'
        f'<h2 id="{sid}-h">{badge}{esc(title)}</h2>{body}</section>'
    )


def _figure(svg: str, caption: str) -> str:
    if not svg:
        return ""
    return f'<figure><div class="chart">{svg}</div><figcaption class="fine">{esc(caption)}</figcaption></figure>'


def _meter(share: float | None, css: str = "", *, thin: bool = False) -> str:
    """A decorative bar; the number it draws is always in the text beside it."""
    if share is None:
        return ""
    pct = max(0.0, min(1.0, share)) * 100
    cls = " ".join(x for x in ("meter", css, "thin" if thin else "") if x)
    return f'<span class="{cls}" aria-hidden="true"><i style="width:{pct:.1f}%"></i></span>'


# ---------------------------------------------------------------------------
# Countermeasures in priority order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ranked:
    """A countermeasure with the findings it answers and what they claim."""

    number: int
    measure: Countermeasure
    findings: tuple[Finding, ...]
    #: Generated and context tokens its findings claim (risks claim none).
    tokens: int
    seconds: float
    #: Its share of the scorecard's waste (uncertain waste counts until
    #: verified, ADR 0006), or None when it answers only risks.
    waste_share: float | None

    @property
    def protects_outcome(self) -> bool:
        return any(f.kind is FindingKind.RISK for f in self.findings)


def _ranked(report: A3Report) -> list[_Ranked]:
    """Countermeasures in the order to act on them: confident before verify
    first, outcome risks first, then by the tokens their findings claim."""
    a = report.analysis
    by_id = {f.id: f for f in a.findings}
    allocated = a.allocated_waste_tokens()
    waste = a.scorecard.waste_tokens
    rows: list[
        tuple[Countermeasure, tuple[Finding, ...], int, float, float | None]
    ] = []
    for c in report.countermeasures:
        found = tuple(by_id[i] for i in c.addresses if i in by_id)
        wastes = [f for f in found if f.kind is not FindingKind.RISK]
        tokens = sum(f.tokens + f.context_tokens for f in wastes)
        seconds = sum(f.seconds for f in wastes)
        share = None
        if waste > 0:
            share = sum(allocated.get(f.id, 0.0) for f in wastes) / waste
        rows.append((c, found, tokens, seconds, share))
    rows.sort(
        key=lambda r: (
            r[0].uncertain,
            not any(f.kind is FindingKind.RISK for f in r[1]),
            -r[2],
        )
    )
    return [
        _Ranked(i, c, found, tokens, seconds, share)
        for i, (c, found, tokens, seconds, share) in enumerate(rows, 1)
    ]


def _impact(r: _Ranked) -> str:
    n = len(r.findings)
    parts = [f"answers {n} finding{'s' if n != 1 else ''}"]
    if r.tokens:
        claim = f"{r.tokens:,} tokens"
        if r.seconds:
            claim += f", {fmt_seconds(r.seconds)}"
        parts.append(
            claim + (" claimed (uncertain)" if r.measure.uncertain else " claimed")
        )
    if r.waste_share:
        parts.append(f"{fmt_pct(r.waste_share, 0)} of waste")
    if r.protects_outcome:
        parts.append("protects the outcome")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _chips(report: A3Report) -> str:
    a = report.analysis
    sc = a.scorecard
    chips = [
        f'<li class="level">{"L3 Grounded" if a.repository is not None else "L2 Explained"}</li>',
        f"<li>Session <code>{esc(report.session_id or '-')}</code></li>",
        f"<li><b>{a.events}</b> events</li>",
        f"<li><b>{fmt_tokens(sc.generated_tokens)}</b> generated tokens</li>",
    ]
    if sc.agent_seconds > 0:
        chips.append(f"<li><b>{fmt_seconds(sc.agent_seconds)}</b> agent time</li>")
    if report.cost is not None and report.cost.turns > report.cost.unpriced_turns:
        est = " (est.)" if report.cost.estimated else ""
        chips.append(f"<li><b>{esc(_usd(report.cost.usd))}</b>{est}</li>")
    if report.outcome is not None:
        v = report.outcome.verdict.value
        tone = {"accepted": "ok", "rejected": "bad"}.get(v, "unsure")
        chips.append(f'<li class="verdict {tone}">Outcome {esc(v)}</li>')
    return f'<ul class="chips" aria-label="Session facts">{"".join(chips)}</ul>'


def _nav(report: A3Report) -> str:
    items = [("summary", "", "Summary"), ("s-1", "1", "Background")]
    items.append(("s-scorecard", "", "Scorecard"))
    if report.outcome is not None:
        items.append(("s-outcome", "", "Outcome"))
    items += [
        ("s-2", "2", "Current state"),
        ("s-3", "3", "Analysis"),
        ("s-4", "4", "Root causes"),
    ]
    if report.analysis.usage is not None:
        items.append(("s-repository", "", "Repository"))
    items += [
        ("s-5", "5", "Countermeasures"),
        ("s-6", "6", "Follow-up"),
        ("s-method", "", "Method"),
    ]
    links = "".join(
        f'<li><a href="#{href}">{f"<b>{n}</b>" if n else ""}{esc(label)}</a></li>'
        for href, n, label in items
    )
    return f'<nav class="toc" aria-label="A3 sections"><ol>{links}</ol></nav>'


def _summary(report: A3Report, ranked: Sequence[_Ranked]) -> str:
    sc = report.analysis.scorecard
    va = sc.activity_share(ActivityClass.VALUE_ADDING)
    unsure = sc.activity_share("uncertain")
    # Uncertain waste counts until verified (ADR 0006).
    avoid = sc.activity_share(ActivityClass.AVOIDABLE) + unsure
    fe = sc.flow_efficiency_tokens
    hero = [
        (
            "Value-adding work",
            fmt_pct(va, 0),
            _meter(va),
            "of generated tokens classed value-adding",
        ),
        (
            "Avoidable waste",
            fmt_pct(avoid, 0),
            _meter(avoid, "waste"),
            f"{sc.waste_tokens:,} generated tokens, {fmt_seconds(sc.waste_seconds)}"
            + (
                f"; {fmt_pct(unsure, 0)} of tokens uncertain, counted until verified"
                if unsure
                else ""
            ),
        ),
        (
            "Flow efficiency",
            "n/a" if fe is None else fmt_pct(fe, 0),
            _meter(fe, "flow"),
            "of generated tokens progressing or iterating",
        ),
        (
            "TER",
            "not computed" if sc.ter is None else f"{sc.ter.value:.2f}",
            _meter(None if sc.ter is None else sc.ter.value, "ter"),
            "token efficiency ratio, 0 to 1",
        ),
    ]
    tiles = "".join(
        f'<div><span class="hl">{esc(label)}</span><b>{esc(value)}</b>{meter}<small>{esc(sub)}</small></div>'
        for label, value, meter, sub in hero
    )
    if ranked:
        items = "".join(_top_item(r) for r in ranked[:TOP_ACTIONS])
        more = len(ranked) - TOP_ACTIONS
        tail = (
            f'<p class="fine" style="margin-top:6px">{more} more countermeasure(s) in '
            '<a href="#s-5">section 5</a>.</p>'
            if more > 0
            else ""
        )
        actions = f'<h3>Do these first</h3><ol class="top">{items}</ol>{tail}'
    else:
        actions = (
            '<h3>Do these first</h3><p class="empty">Nothing to change: no detector '
            "fired on this session.</p>"
        )
    return (
        f'<p class="problem"><b>Problem.</b> {esc(report.problem)}</p>'
        f'<div class="hero" role="list" aria-label="Headline measures">{tiles}</div>'
        + actions
    )


def _top_item(r: _Ranked) -> str:
    c = r.measure
    css = "verify" if c.uncertain else ("risk" if r.protects_outcome else "")
    kinds = " ".join(
        f'<span class="tag kind">{esc(ACTION_TARGETS[k])}</span>'
        for k in dict.fromkeys(a.kind for a in c.actions)
    )
    unsure = ' <span class="tag warn">verify first</span>' if c.uncertain else ""
    return (
        f'<li class="{css}"><span class="rank">Action {r.number}</span>'
        f'<a class="t" href="#cm-{r.number}">{esc(c.title)}</a>'
        f'<span class="impact">{esc(_impact(r))}{unsure}</span><span>{kinds}</span></li>'
    )


def _background(report: A3Report) -> str:
    intents = report.intents
    if intents:
        quotes = "".join(
            f"<blockquote>{esc(t.strip())}</blockquote>" for t in intents[:3]
        )
        if len(intents) > 3:
            quotes += f'<p class="fine">… and {len(intents) - 3} more prompt(s).</p>'
        lead = (
            f"<p>The developer asked for the outcome below"
            f"{' and changed it during the session' if len(intents) > 1 else ''}. "
            "Value is judged against it.</p>"
        )
    else:
        quotes = '<p class="empty">No prompt was recorded.</p>'
        lead = ""
    a = report.analysis
    left = (
        lead
        + quotes
        + f'<p class="fine">{a.events} events analysed · session <code>{esc(report.session_id or "-")}</code></p>'
        + "".join(
            f'<p class="fine"><b>Limit.</b> {esc(describe_limit(limit))}</p>'
            for limit in report.usage_limits
        )
    )
    timeline = _intent(report)
    if not timeline:
        return left
    return f'<div class="split"><div>{left}</div><div>{timeline}</div></div>'


def _intent(report: A3Report) -> str:
    """The intent timeline: each revision, what changed it, and the drift and
    low-alignment periods judged against it, all citing event ids."""
    a = report.analysis
    timeline = a.intent
    revisions = timeline.record.revisions
    if not revisions:
        return ""
    rows: list[str] = []
    for r in revisions:
        scored = [
            x
            for x in timeline.alignments
            if x.revision == r.revision and x.score is not None
        ]
        bands = Counter(x.band.value for x in scored)
        change = next(
            (c for c in timeline.changes if c.to_revision == r.revision), None
        )
        detail = [
            f'<span class="tag{" warn" if change else ""}">{esc(r.relation.value)}</span> '
            f"<b>Revision {r.revision}</b> <code>{esc(r.event_id)}</code>"
        ]
        if change is not None:
            dropped = (
                f"; dropped {esc(', '.join(sorted(change.abandoned)))}"
                if change.abandoned
                else ""
            )
            detail.append(
                f"<small>Changed by the developer: {esc(change.reason)}{dropped}.</small>"
            )
        detail.append(
            f"<small>{len(scored)} agent event(s) scored: {bands['aligned']} aligned, "
            f"{bands['partial']} partial, {bands['low']} low.</small>"
        )
        for f in a.drift_findings:
            x = timeline.alignment_of(f.waste_events[0]) if f.waste_events else None
            if x is None or x.revision != r.revision:
                continue
            unsure = ' <span class="tag warn">uncertain</span>' if f.uncertain else ""
            detail.append(
                f'<small><span class="tag waste">drift</span>{unsure} {esc(f.title)} '
                f"(confidence {f.confidence:.2f})</small>{_evidence(f.evidence)}"
            )
        for period in timeline.periods:
            if period.revision != r.revision:
                continue
            detail.append(
                f'<small><span class="tag risk">low alignment</span> {len(period.events)} '
                f"consecutive agent events, mean score {period.mean_score:.2f}</small>"
                f"{_evidence(period.events)}"
            )
        rows.append(f"<li>{''.join(detail)}</li>")
    cfg = timeline.config
    return (
        "<h3>Intent timeline</h3>"
        f'<ol class="intent">{"".join(rows)}</ol>'
        f'<p class="fine">Alignment: {esc(timeline.scorer)} scorer. Low alignment is a score '
        f"below {cfg.low_below:.2f} for {cfg.min_events}+ agent events in a row; drift is an "
        f"edit below {cfg.drift_below:.2f} with no intent change recorded.</p>"
    )


def _scorecard(report: A3Report) -> str:
    sc = report.analysis.scorecard
    tiles: list[tuple[str, str, str]] = []
    fe = (
        "n/a"
        if sc.flow_efficiency_tokens is None
        else fmt_pct(sc.flow_efficiency_tokens, 0)
    )
    ft = (
        "no timestamps"
        if sc.flow_efficiency_time is None
        else f"{fmt_pct(sc.flow_efficiency_time, 0)} of agent time"
    )
    tiles.append(
        ("Flow efficiency", fe, f"of generated tokens progressing or iterating; {ft}")
    )
    tiles.append(
        (
            "Waste cost",
            fmt_tokens(sc.waste_tokens + sc.waste_context_tokens),
            f"tokens: {sc.waste_tokens:,} generated, {sc.waste_context_tokens:,} context; "
            f"{fmt_seconds(sc.waste_seconds)} of agent time",
        )
    )
    tiles.append(
        (
            "Value-adding",
            fmt_pct(sc.activity_share(ActivityClass.VALUE_ADDING), 0),
            f"necessary NVA {fmt_pct(sc.activity_share(ActivityClass.NECESSARY_NON_VALUE_ADDING), 0)}, "
            f"avoidable {fmt_pct(sc.activity_share(ActivityClass.AVOIDABLE), 0)}, "
            f"uncertain {fmt_pct(sc.activity_share('uncertain'), 0)}",
        )
    )
    tiles.extend(_inventory_tiles(report))
    if sc.ter is not None:
        tiles.append(("TER", f"{sc.ter.value:.2f}", sc.ter.method))
    else:
        tiles.append(("TER", "not computed", "run with --ter offline or --ter model"))
    sve = report.value_efficiency
    if sve.status is ValueStatus.UNKNOWN or sve.tokens is None:
        tiles.append(("Software Value Efficiency", "unknown", sve.reason))
    else:
        time = "" if sve.time is None else f", {fmt_pct(sve.time, 0)} of agent time"
        verdict = "" if sve.verdict is None else sve.verdict.value
        tiles.append(
            (
                "Software Value Efficiency",
                fmt_pct(sve.tokens, 0),
                f"value-adding work toward the {verdict} outcome, of generated tokens"
                f"{time}",
            )
        )
    wip = report.analysis.wip
    if wip.peak is not None:
        final = wip.final
        tiles.append(
            (
                "Peak WIP",
                str(wip.peak.total),
                ", ".join(f"{wip.peak_of(k)} {k.value}" for k in WipKind)
                + f" at most; {0 if final is None else final.total} open at the end",
            )
        )
    tiles.append(
        (
            "Findings",
            str(sc.findings),
            f"waste; {sc.uncertain_findings} of them uncertain, {sc.risks} risk(s); "
            f"{sc.iterations} iteration / {sc.rework_cycles} rework cycle(s)",
        )
    )
    if sc.composite is not None:
        parts = " + ".join(f"{n} {v:.2f}" for n, v, _ in sc.composite.components)
        tiles.append(
            (
                "Composite",
                f"{sc.composite.value:.2f}",
                f"= mean of {parts}. Read the parts, not the sum.",
            )
        )
    cards = "".join(
        f'<div class="kpi"><span>{esc(label)}</span><b>{esc(value)}</b><small>{esc(sub)}</small></div>'
        for label, value, sub in tiles
    )
    return (
        '<div class="split"><div>'
        f'<div class="kpis" role="list" aria-label="Scorecard">{cards}</div>'
        '<p class="fine" style="margin-top:8px">Each dimension stands alone: no single score '
        f"hides the others. Findings below confidence {UNCERTAIN_BELOW:.2f} are labelled "
        "uncertain and counted as waste until verified. Token minimisation is not a goal: efficiency is value "
        f"delivered per unit of resource. {esc(SVE_DEFINITION)}</p>"
        f"</div><div>{_dimensions(report)}</div></div>"
    )


def _measure_value(m: Measure) -> str:
    v = m.value
    if v is None:
        return "unknown"
    if isinstance(v, str):
        return v
    if m.unit == "ratio":
        return fmt_pct(float(v), 0)
    if m.unit == "tokens":
        return fmt_tokens(v)
    if m.unit == "seconds":
        return fmt_seconds(float(v))
    return str(v)


def _dimensions(report: A3Report) -> str:
    """The six scorecard dimensions, each with its named measures."""
    rows = "".join(
        f'<tr><th scope="row">{esc(d.dimension.label)}'
        f"<small>{esc(d.dimension.question)}</small></th><td>"
        + " · ".join(
            f"{esc(m.label)} <b>{esc(_measure_value(m))}</b>" for m in d.measures
        )
        + "</td></tr>"
        for d in report.dimensions
    )
    return (
        '<h3>Scorecard dimensions</h3><div class="chart"><table>'
        '<thead><tr><th scope="col">Dimension</th><th scope="col">Measures</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _usd(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def _inventory_tiles(report: A3Report) -> list[tuple[str, str, str]]:
    """Context inventory (TER-DET-004) and, when priced, the session cost."""
    inv = report.inventory
    cost = report.cost
    tiles: list[tuple[str, str, str]] = []
    if inv is not None:
        sub = (
            f"tokens: {inv.unused_tokens:,} read and never used, "
            f"{inv.reread_tokens:,} read again ({inv.unchanged_reread_tokens:,} unchanged)"
        )
        if cost is not None:
            sub += (
                f"; carrying cost {_usd(cost.unused_context_usd)} unused, "
                f"{_usd(cost.reread_context_usd)} re-read"
            )
        tiles.append(
            (
                "Context inventory",
                fmt_tokens(inv.unused_tokens + inv.reread_tokens),
                sub,
            )
        )
    if cost is not None:
        on = cost.priced_on.isoformat() if cost.priced_on else "latest prices"
        sub = f"{cost.turns} model turn(s) at prices in force on {on}"
        if cost.unpriced_turns:
            sub += (
                f"; {cost.unpriced_turns} unpriced ({', '.join(cost.unpriced_models)})"
            )
        if cost.estimated:
            sub += "; estimated: " + ", ".join(cost.estimate_reasons)
        tiles.append(
            (
                "Session cost" + (" (estimated)" if cost.estimated else ""),
                _usd(cost.usd),
                sub,
            )
        )
    return tiles


def _outcome(report: A3Report) -> str:
    """The verdict, judged apart from the scorecard (no measure reads it)."""
    verdict = report.outcome
    if verdict is None:
        return ""
    required = [r for r in verdict.results if r.check.required]
    passed = sum(
        1 for r in required if r.status is not None and r.status.value == "passed"
    )
    per = report.tokens_per_verified_outcome
    tiles = [
        ("Verdict", verdict.verdict.value, "; ".join(verdict.reasons)),
        (
            "Required checks",
            f"{passed} / {len(required)}",
            f"passed, against “{verdict.contract.name}”",
        ),
        (
            "Tokens per verified outcome",
            "n/a" if per is None else fmt_tokens(round(per)),
            "generated tokens / accepted outcomes"
            if per is not None
            else "no accepted outcome to divide by",
        ),
    ]
    cards = "".join(
        f'<div class="kpi"><span>{esc(label)}</span><b>{esc(value)}</b><small>{esc(sub)}</small></div>'
        for label, value, sub in tiles
    )
    # Every check with its status and evidence source, open checks first, so
    # an accepted verdict shows what it rests on too.
    ordered = sorted(
        verdict.results,
        key=lambda r: r.status is not None and r.status.value == "passed",
    )
    head = (
        '<thead><tr><th scope="col">Status</th><th scope="col">Check</th>'
        '<th scope="col">Evidence</th></tr></thead>'
    )

    def row(r: CheckResult) -> str:
        status = "no evidence" if r.status is None else r.status.value
        tone = (
            "ok"
            if status == "passed"
            else ("waste" if r.status is not None and r.status.is_failure else "warn")
        )
        optional = "" if r.check.required else " (optional)"
        sources = ", ".join(e.source for e in r.evidence) or "-"
        detail = next((e.detail for e in r.evidence if e.detail), "")
        return (
            f'<tr><td><span class="tag {tone}">{esc(status)}</span></td>'
            f"<td><code>{esc(r.check.id)}</code>{esc(optional)}"
            + (f"<br><small>{esc(detail)}</small>" if detail else "")
            + f"</td><td><code>{esc(sources)}</code></td></tr>"
        )

    shown, rest = ordered[:8], ordered[8:]
    checks = (
        f'<div class="chart"><table>{head}<tbody>{"".join(row(r) for r in shown)}</tbody></table></div>'
        if shown
        else ""
    )
    if rest:
        checks += (
            f"<details><summary>Show the other {len(rest)} checks</summary>"
            f'<div class="chart"><table>{head}<tbody>{"".join(row(r) for r in rest)}</tbody></table></div></details>'
        )
    return (
        '<div class="split"><div>'
        f'<div class="kpis" role="list" aria-label="Outcome">{cards}</div>'
        f'<p class="fine" style="margin-top:8px">Judged from <code>{esc(verdict.run_ref)}</code> '
        f"({esc(verdict.source)}), separately from the scorecard: no measure on this page "
        "reads the verdict.</p>"
        f"</div><div>{checks}</div></div>"
    )


def _current_state(report: A3Report) -> str:
    svg = value_stream_map(report.analysis.value_stream)
    if not svg:
        return ""
    return (
        f'<figure><div class="chart wide">{svg}</div><figcaption class="fine">'
        "Each box is a stage of the agentic value stream. Hover a box for its findings."
        "</figcaption></figure>"
    )


def _analysis(report: A3Report) -> str:
    charts = [
        _figure(
            pareto(report),
            "Wastes by tokens claimed (generated and context), with the running share.",
        )
        or '<p class="empty">No waste was found, so there is no Pareto to draw.</p>',
        _figure(
            activity_bar(report),
            "Value-adding, necessary but non-value-adding, avoidable, uncertain.",
        ),
        _figure(
            flow_bar(report),
            "Progressing versus repeating, reworking, recovering, waiting, inventory.",
        ),
    ]
    charts.append(
        _figure(
            wip_chart(report),
            "Unresolved hypotheses, tasks, edits and failures after every event.",
        )
    )
    if report.analysis.scorecard.agent_seconds > 0:
        charts.append(
            _figure(
                flow_bar(report, time=True), "The same split for wall-clock agent time."
            )
        )
    cycles = report.analysis.cycles
    note = ""
    if cycles:
        rows = "".join(
            f"<li><code>{esc(c.command)}</code>: {esc(c.verdict.value)}, {esc(c.reason)}.</li>"
            for c in cycles
        )
        note = (
            '<h3>Fail → fix → re-run cycles</h3><ul class="fine">' + rows + "</ul>"
            '<p class="fine">Iteration (the next run passed or failed differently) is productive '
            "and counted as recovering; only an unchanged failure is rework.</p>"
        )
    return f'<div class="charts">{"".join(charts)}</div>{note}'


#: Files listed per column of the repository section before the rest fold.
FILES_SHOWN = 10


@dataclass(frozen=True)
class _Changes:
    """When each repository file was first read and successfully edited."""

    #: Files with at least one edit that did not fail, in first-edit order.
    changed: tuple[str, ...]
    first_read: dict[str, int]
    first_edit: dict[str, int]
    last_edit: dict[str, int]
    #: Changed files that were not in the repository at the start commit.
    created: frozenset[str]

    def read_before_edit(self, path: str) -> bool:
        read = self.first_read.get(path)
        return read is not None and read < self.first_edit[path]

    def changed_after_read(self, path: str) -> bool:
        read = self.first_read.get(path)
        return (
            read is not None and path in self.last_edit and read < self.last_edit[path]
        )


def _changes(report: A3Report) -> _Changes:
    """Order reads against edits, leaving out edits whose tool call failed."""
    a = report.analysis
    usage, repo = a.usage, a.repository
    index = {s.event_id: s.index for s in a.steps}
    first_read: dict[str, int] = {}
    for r in usage.reads if usage is not None else ():
        at = index.get(r.event_id)
        if at is not None and (r.path not in first_read or at < first_read[r.path]):
            first_read[r.path] = at
    first_edit: dict[str, int] = {}
    last_edit: dict[str, int] = {}
    if repo is not None:
        for s in a.steps:
            if not (s.is_edit and s.paths) or s.event_id in repo.failed_edits:
                continue
            path = repo.repository_path(s.paths[0])
            if path is None:
                continue
            first_edit.setdefault(path, s.index)
            last_edit[path] = s.index
    files = repo.files if repo is not None else frozenset()
    return _Changes(
        changed=tuple(first_edit),
        first_read=first_read,
        first_edit=first_edit,
        last_edit=last_edit,
        created=frozenset(p for p in first_edit if p not in files),
    )


#: Unused reads listed before the rest fold into a disclosure.
UNUSED_SHOWN = 8


def _repository(report: A3Report) -> str:
    """L3: what the session read from its repository and whether later work
    used it (TER-EVD-008), files explored against files changed, and how the
    outcome valued exploration, reasoning and validation (TER-LEN-009)."""
    a = report.analysis
    usage = a.usage
    if usage is None:
        return ""
    used = usage.count(UsageStatus.USED)
    unused = usage.count(UsageStatus.UNUSED)
    pending = usage.count(UsageStatus.PENDING)
    material = sum(r.material for r in usage.reads)
    share = usage.share_used
    ch = _changes(report)
    both = sum(ch.changed_after_read(p) for p in usage.explored)
    tiles = [
        (
            "Reads later used",
            "n/a" if share is None else fmt_pct(share, 0),
            _meter(share),
            f"{used} used ({material} by a change, command or check), {unused} unused, "
            f"{pending} not yet judged, of {len(usage.reads)} repository read(s)",
        ),
        (
            "Unused context",
            fmt_tokens(usage.unused_tokens),
            "",
            "tokens carried in context from reads nothing later used",
        ),
        (
            "Explored → changed",
            f"{both} / {len(usage.explored)}",
            _meter(both / len(usage.explored) if usage.explored else None, "flow"),
            f"files read and then edited; {len(ch.changed)} file(s) changed in all",
        ),
    ]
    cards = "".join(
        f'<div class="kpi"><span>{esc(label)}</span><b>{esc(value)}</b>{meter}<small>{esc(sub)}</small></div>'
        for label, value, meter, sub in tiles
    )

    def explored_mark(p: str) -> str:
        if ch.changed_after_read(p):
            return (
                '<span class="mark yes" aria-label="changed after it was read">✓</span>'
            )
        return '<span class="mark" aria-label="not changed after it was read">·</span>'

    def changed_mark(p: str) -> str:
        if p in ch.created:
            return '<span class="mark" aria-label="created by the session">+</span>'
        if ch.read_before_edit(p):
            return '<span class="mark yes" aria-label="read before its first edit">✓</span>'
        return '<span class="mark no" aria-label="edited before it was read">!</span>'

    def files(paths: Sequence[str], mark: Callable[[str], str]) -> str:
        def li(p: str) -> str:
            return f"<li>{mark(p)}<code>{esc(p)}</code></li>"

        head = "".join(li(p) for p in paths[:FILES_SHOWN])
        body = f'<ul class="files">{head}</ul>'
        if len(paths) > FILES_SHOWN:
            rest = "".join(li(p) for p in paths[FILES_SHOWN:])
            body += (
                f"<details><summary>{len(paths) - FILES_SHOWN} more</summary>"
                f'<ul class="files">{rest}</ul></details>'
            )
        return body if paths else '<p class="empty">None.</p>'

    unused_reads = sorted(
        (r for r in usage.reads if r.status is UsageStatus.UNUSED),
        key=lambda r: -r.context_tokens,
    )
    head_row = (
        '<thead><tr><th scope="col">File and read event</th>'
        '<th scope="col" class="num">Context tokens</th></tr></thead>'
    )

    def unused_rows(reads: Sequence[ReadUsage]) -> str:
        return "".join(
            f"<tr><td><code>{esc(r.path)}</code>{_evidence((r.event_id,))}</td>"
            f'<td class="num">{r.context_tokens:,}</td></tr>'
            for r in reads
        )

    unused_table = ""
    if unused_reads:
        shown, rest = unused_reads[:UNUSED_SHOWN], unused_reads[UNUSED_SHOWN:]
        unused_table = (
            f'<h3>Reads nothing used</h3><div class="chart"><table>{head_row}'
            f"<tbody>{unused_rows(shown)}</tbody></table></div>"
        )
        if rest:
            unused_table += (
                f"<details><summary>Show the other {len(rest)} unused read(s)</summary>"
                f'<div class="chart"><table>{head_row}<tbody>{unused_rows(rest)}'
                "</tbody></table></div></details>"
            )
    value = a.value
    value_table = ""
    if value is not None:
        counts = value.counts()
        classes = [v for v in ValueClass]
        head = "".join(
            f'<th scope="col" class="num">{esc(v.value.replace("_", " "))}</th>'
            for v in classes
        )
        rows = "".join(
            f'<tr><th scope="row">{esc(k.value)}</th>'
            + "".join(
                f'<td class="num">{counts[k.value][v.value]}</td>' for v in classes
            )
            + "</tr>"
            for k in JudgedKind
        )
        unsure = sum(
            j.uncertain and j.value is not ValueClass.UNJUDGED for j in value.judgements
        )
        value_table = (
            "<h3>Outcome value</h3>"
            '<div class="chart"><table><thead><tr><th scope="col">Work</th>'
            f"{head}</tr></thead><tbody>{rows}</tbody></table></div>"
            f'<p class="fine">Each step judged against what the outcome required; '
            f"{unsure} judgement(s) are uncertain.</p>"
        )
    repo = a.repository
    where = ""
    if repo is not None:
        roots = ", ".join(f"<code>{esc(r)}</code>" for r in repo.roots) or "-"
        where = (
            f'<p class="fine">Grounded on the repository at its start commit with the '
            f"<code>{esc(repo.engine)}</code> engine; session roots {roots}.</p>"
        )
    return (
        where
        + '<div class="split"><div>'
        + f'<div class="kpis" role="list" aria-label="Repository evidence">{cards}</div>'
        + '<h3>Files explored <small class="fine">(✓ changed after it was read)</small></h3>'
        + files(usage.explored, explored_mark)
        + '<h3>Files changed <small class="fine">(✓ read before its first edit, '
        "! edited before it was read, + created)</small></h3>"
        + files(ch.changed, changed_mark)
        + f"</div><div>{unused_table}{value_table}</div></div>"
    )


def _evidence(ids: Sequence[str], limit: int = 6) -> str:
    shown = "".join(f"<code>{esc(i)}</code>" for i in ids[:limit])
    if len(ids) <= limit:
        return f'<div class="ev" aria-label="Evidence event ids">{shown}</div>'
    rest = "".join(f"<code>{esc(i)}</code>" for i in ids[limit:])
    return (
        f'<div class="ev" aria-label="Evidence event ids">{shown}'
        f"<details><summary>+{len(ids) - limit} more</summary>{rest}</details></div>"
    )


def _root_causes(report: A3Report, ranked: Sequence[_Ranked]) -> str:
    if not report.root_causes:
        return '<p class="empty">No findings: every step was classified from its stage alone.</p>'
    fixes: dict[str, _Ranked] = {}
    for r in ranked:
        for f in r.findings:
            fixes.setdefault(f.id, r)
    rows = "".join(
        _finding_card(i, f, fixes.get(f.id))
        for i, f in enumerate(report.root_causes, 1)
    )
    shown = {f.id for f in report.root_causes}
    hidden = [f for f in report.analysis.findings if f.id not in shown]
    tail = ""
    if hidden:
        start = len(report.root_causes) + 1
        more = "".join(
            _finding_card(i, f, fixes.get(f.id)) for i, f in enumerate(hidden, start)
        )
        tail = (
            f"<details><summary>Show {len(hidden)} more finding(s), also in the JSON "
            f'output</summary><ol class="findings" start="{start}">{more}</ol></details>'
        )
    return (
        '<p class="fine">Largest cost first; uncertain findings and risks after. Each cites the '
        "events it rests on and links to the countermeasure that answers it.</p>"
        f'<ol class="findings">{rows}</ol>{tail}'
    )


def _finding_card(i: int, f: Finding, fix: _Ranked | None) -> str:
    risk = f.kind is FindingKind.RISK
    css = "risk" if risk else ("unsure" if f.uncertain else "")
    tag_cls = "risk" if risk else "waste"
    kind = "Risk" if risk else f.waste.label
    unsure = '<span class="tag warn">uncertain</span>' if f.uncertain else ""
    cost = "outcome at risk" if risk else f"{f.tokens + f.context_tokens:,} tokens"
    if not risk and f.seconds:
        cost += f", {fmt_seconds(f.seconds)}"
    answer = (
        f'<span>Fix: <a href="#cm-{fix.number}">Action {fix.number}, {esc(fix.measure.title)}</a></span>'
        if fix is not None
        else ""
    )
    return (
        f'<li class="finding {css}" id="f-{_slug(f.id)}">'
        f'<div class="fh"><span class="rk">{i}</span><b>{esc(f.title)}</b>'
        f'<span class="cost">{cost}</span></div>'
        f"<p>{esc(f.explanation)}</p>"
        f'<div class="fm"><span class="tag {tag_cls}">{esc(kind)}</span>{unsure}'
        f"<span>confidence {f.confidence:.2f}"
        f"{_meter(f.confidence, 'warn' if f.uncertain else '', thin=True)}</span>{answer}</div>"
        f"{_evidence(f.evidence)}</li>"
    )


def _snippet(file: str, body: str) -> str:
    return (
        f'<div class="snip"><span class="file">{esc(file)}</span>'
        f"<pre><code>{esc(body)}</code></pre></div>"
    )


def _action(a: Action) -> str:
    snippet = ""
    if a.snippet:
        target = ACTION_TARGETS[a.kind]
        if a.language == "json+bash":
            config, _, script = a.snippet.partition("\n\n")
            snippet = _snippet(target, config) + (
                "<details><summary>Hook script</summary>"
                f"{_snippet('hook script', script)}</details>"
            )
        else:
            snippet = _snippet(
                target if a.kind is not ActionKind.PRACTICE else "", a.snippet
            )
    return (
        f'<li><span class="kind">{esc(a.kind.label)}</span> {esc(a.text)}{snippet}</li>'
    )


def _countermeasure(r: _Ranked) -> str:
    c = r.measure
    unsure = ' <span class="tag warn">verify first</span>' if c.uncertain else ""
    # Every finding has a card (the ones past the first six in a disclosure),
    # so every answer links.
    answers = "".join(
        f'<li><a href="#f-{_slug(f.id)}">{esc(f.title)}</a></li>' for f in r.findings
    )
    missing = [i for i in c.addresses if i not in {f.id for f in r.findings}]
    answers += "".join(f"<li><code>{esc(i)}</code></li>" for i in missing)
    return (
        f'<article class="cm{" verify" if c.uncertain else ""}" id="cm-{r.number}">'
        f'<h3><span class="n">{r.number}</span>{esc(c.title)}{unsure}</h3>'
        f'<p class="impact">{esc(_impact(r))}</p>'
        f'<p class="why">{esc(c.rationale)}</p>'
        f'<p class="fine" style="margin:0">Answers</p><ul class="answers">{answers}</ul>'
        f'<ol class="acts">{"".join(_action(a) for a in c.actions)}</ol></article>'
    )


def _countermeasures(report: A3Report, ranked: Sequence[_Ranked]) -> str:
    if not ranked:
        return (
            '<p class="empty">Nothing to change: no detector fired on this session.</p>'
        )
    return (
        '<p class="fine">In the order to act on them: confident before verify-first, outcome '
        "risks first, then by the tokens their findings claim. Derived from the findings, "
        "never from token thresholds. Hook scripts use documented Claude Code behaviour (exit 2 "
        "blocks a PreToolUse call, or returns stderr to the agent after PostToolUse); adapt "
        "paths and commands before use.</p>"
        f'<div class="cms">{"".join(_countermeasure(r) for r in ranked)}</div>'
    )


def _follow_up(report: A3Report) -> str:
    rows = "".join(
        f'<tr><td>{esc(f.metric)}</td><td class="num">{esc(f.current)}</td>'
        f'<td class="num">→ {esc(f.target)}</td><td><code>{esc(f.how)}</code></td></tr>'
        for f in report.follow_up
    )
    return (
        '<p class="fine">Run <code>ter a3 &lt;next-session.jsonl&gt; --json</code> and read the '
        "field under How. Targets are directions, not token budgets.</p>"
        '<div class="chart"><table><thead><tr><th scope="col">Measure next run</th>'
        '<th scope="col" class="num">Now</th><th scope="col" class="num">Target</th>'
        f'<th scope="col">How</th></tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _method(report: A3Report) -> str:
    rows = "".join(
        f"<dt><code>{esc(i)}</code></dt><dd>{esc(r)}</dd>"
        for i, _, _, r in report.analysis.detectors
    )
    return (
        '<p class="fine">Every classification cites event ids from the <code>ter.event</code> '
        "stream; the JSON output (<code>--json</code>) carries the per-event basis and the "
        "evidence graph (<code>--graph</code>). Detectors and their confidence rules:</p>"
        f"<details><summary>Show the {len(report.analysis.detectors)} detector rules</summary><dl>{rows}</dl></details>"
        + _lean_concepts()
    )


def _lean_concepts() -> str:
    """Each Lean concept and the measures TER computes for it (TER-LEN-006)."""
    rows = "".join(
        f'<tr><th scope="row">{esc(c.label)}</th><td>'
        + "<br>".join(
            f"{esc(m.name)} <code>{esc(m.source)}</code><small>{esc(m.meaning)}</small>"
            for m in measures
        )
        + "</td></tr>"
        for c, measures in LEAN_MEASURES.items()
    )
    return (
        "<details><summary>Show the Lean concepts and their measures</summary>"
        '<div class="chart"><table><thead><tr><th scope="col">Concept</th>'
        f'<th scope="col">Measures</th></tr></thead><tbody>{rows}</tbody></table></div></details>'
    )


def render_a3_html(report: A3Report) -> str:
    """Render ``report`` as one self-contained HTML document.

    It opens with a summary (the problem, four headline measures and the
    countermeasures to act on first), then follows Toyota A3 order. Findings
    and countermeasures link to each other by in-page anchors, so the page
    needs no script to navigate.
    """
    title = esc(report.title)
    ranked = _ranked(report)
    repository = _repository(report)
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">',
        '<meta name="color-scheme" content="light dark">',
        f"<title>TER A3: {title}</title>",
        f"<style>\n{stylesheet('.ter-chart')}\n{_PAGE_VARS}\n{_CSS}</style>",
        "</head>",
        '<body><a class="skip" href="#summary">Skip to the summary</a>',
        '<main class="ter-chart">',
        '<header class="a3"><p class="eyebrow">TER A3 · Lean analysis of an agent session · '
        f"schema {A3_SCHEMA}</p>"
        f"<h1>{title}</h1>{_chips(report)}</header>",
        _nav(report),
        '<div class="sheet">',
        _box(None, "At a glance", _summary(report, ranked), "full summary", "summary"),
        _box(1, "Background", _background(report), "full"),
        _box(None, "Scorecard", _scorecard(report), "full", "s-scorecard"),
        *(
            [_box(None, "Outcome", _outcome(report), "full", "s-outcome")]
            if report.outcome
            else []
        ),
        _box(2, "Current state", _current_state(report), "full"),
        _box(3, "Analysis", _analysis(report)),
        _box(4, "Root causes", _root_causes(report, ranked)),
        *(
            [_box(None, "Repository evidence", repository, "full", "s-repository")]
            if repository
            else []
        ),
        _box(5, "Countermeasures", _countermeasures(report, ranked), "full"),
        _box(6, "Follow-up", _follow_up(report)),
        _box(None, "Evidence and method", _method(report), "", "s-method"),
        "</div>",
        "</main></body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"
