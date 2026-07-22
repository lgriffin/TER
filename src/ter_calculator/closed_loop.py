"""Closed-loop project memory, session lessons, and trend analysis.

The module stays stdlib-only so it can run inside latency-sensitive Claude Code
hooks.  It connects live events to repository memory, records durable lessons,
and aggregates recurring patterns across sessions.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .repository_memory import DEFAULT_INDEX, search_index


@dataclass(frozen=True)
class SessionLesson:
    timestamp: str
    session_id: str
    repository: str
    pattern_type: str
    severity: str
    summary: str
    details: dict[str, Any]
    outcome: str = "observed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_root(event_data: dict[str, Any]) -> Path:
    for key in ("cwd", "project_dir", "repository", "repo_root"):
        value = event_data.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()
    return Path.cwd().resolve()


def build_memory_guidance(
    event_data: dict[str, Any],
    *,
    index_path: str | Path | None = None,
    limit: int = 4,
    minimum_score: float = 0.18,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve project-specific context for a prompt or impending tool call."""
    query = _event_query(event_data)
    if not query:
        return "", []
    root = resolve_project_root(event_data)
    path = Path(index_path) if index_path else root / DEFAULT_INDEX
    if not path.exists():
        return "", []
    try:
        result = search_index(path, query, limit=limit, minimum_score=minimum_score)
    except (OSError, ValueError, json.JSONDecodeError):
        return "", []
    matches = result.get("matches", [])
    if not matches:
        return "", []
    lines = ["[TER Project Memory] Review before acting:"]
    for match in matches:
        location = str(match["path"])
        if match.get("start_line"):
            location += f":{match['start_line']}-{match['end_line']}"
        first_line = re.sub(r"\s+", " ", match.get("excerpt", "")).strip()[:180]
        lines.append(f"- {location} ({match['score']:.2f}): {first_line}")
    for flag in result.get("risk_flags", [])[:3]:
        if flag["type"] in {"duplicate_pattern", "semantic_duplicate_pattern"}:
            lines.append(
                f"- Risk: similar implementation already appears in {flag['path']}; reuse or consolidate it."
            )
        elif flag["type"] == "prior_defect_or_fix":
            lines.append(
                f"- Risk: {flag['path']} contains a related defect/fix history; inspect it before changing behavior."
            )
    return "\n".join(lines), matches


def _event_query(event_data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("prompt", "message", "text", "assistant_message"):
        value = event_data.get(key)
        if isinstance(value, str):
            chunks.append(value)
    tool_input = event_data.get("tool_input")
    if isinstance(tool_input, dict):
        for key in (
            "command",
            "file_path",
            "query",
            "pattern",
            "content",
            "new_string",
        ):
            value = tool_input.get(key)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks).strip()[:8000]


def append_lessons(
    path: str | Path,
    *,
    session_id: str,
    repository: str,
    alerts: list[Any],
    outcome: str = "observed",
) -> int:
    """Append deduplicated alert-derived lessons to a JSONL store."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = _recent_keys(destination)
    rows: list[str] = []
    for alert in alerts:
        key = (session_id, alert.pattern_type, alert.message)
        if key in existing:
            continue
        lesson = SessionLesson(
            timestamp=_utc_now(),
            session_id=session_id,
            repository=repository,
            pattern_type=alert.pattern_type,
            severity=alert.severity,
            summary=alert.message,
            details=dict(alert.details),
            outcome=outcome,
        )
        rows.append(json.dumps(asdict(lesson), sort_keys=True))
        existing.add(key)
    if rows:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    return len(rows)


def _recent_keys(path: Path, limit: int = 1000) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in lines:
        try:
            row = json.loads(line)
            keys.add(
                (str(row["session_id"]), str(row["pattern_type"]), str(row["summary"]))
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return keys


def record_outcome(
    path: str | Path,
    *,
    session_id: str,
    intervention_type: str,
    outcome: str,
    details: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _utc_now(),
        "session_id": session_id,
        "intervention_type": intervention_type,
        "outcome": outcome,
        "details": details or {},
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def analyze_trends(
    path: str | Path,
    *,
    minimum_occurrences: int = 2,
    outcome_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate recurring alert patterns and repository scenarios."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except json.JSONDecodeError:
                continue
    pattern_counts = Counter(str(row.get("pattern_type", "unknown")) for row in rows)
    repository_counts = Counter(str(row.get("repository", "unknown")) for row in rows)
    scenarios = [
        {
            "pattern_type": pattern,
            "occurrences": count,
            "message": f"Watch out for {pattern.replace('_', ' ')}; observed {count} times across recorded sessions.",
        }
        for pattern, count in pattern_counts.most_common()
        if count >= minimum_occurrences
    ]
    outcomes: list[dict[str, Any]] = []
    if outcome_path and Path(outcome_path).exists():
        for line in Path(outcome_path).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and (row.get("effect") or row.get("outcome")):
                    outcomes.append(row)
            except json.JSONDecodeError:
                continue
    effectiveness: dict[str, dict[str, Any]] = {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes:
        by_type.setdefault(str(row.get("intervention_type", "unknown")), []).append(row)
    for kind, group in by_type.items():

        def effect_of(row: dict[str, Any]) -> str:
            effect = str(row.get("effect", ""))
            if effect:
                return effect
            outcome = str(row.get("outcome", ""))
            return {
                "acknowledged": "improved",
                "overridden": "ignored",
                "fired": "issued",
                "no_match": "neutral",
            }.get(outcome, outcome)

        issued_rows = [
            row for row in group if str(row.get("outcome", "")) != "no_match"
        ]
        issued = len(issued_rows)
        followed = sum(
            bool(row.get("followed")) or str(row.get("outcome", "")) == "acknowledged"
            for row in issued_rows
        )
        improved = sum(effect_of(row) == "improved" for row in issued_rows)
        overrides = sum(
            effect_of(row) in {"ignored", "acknowledged_not_followed"}
            for row in issued_rows
        )
        ter_deltas = [
            float(row.get("deltas", {}).get("ter", 0.0)) for row in issued_rows
        ]
        waste_deltas = [
            float(row.get("deltas", {}).get("waste_ratio", 0.0)) for row in issued_rows
        ]
        cost_deltas = [
            float(row.get("deltas", {}).get("estimated_cost_waste_usd", 0.0))
            for row in issued_rows
        ]
        saved = sum(
            max(v, 0.0)
            for row, v in zip(issued_rows, cost_deltas)
            if effect_of(row) == "improved"
        )
        wasted = sum(
            abs(min(v, 0.0))
            for row, v in zip(issued_rows, cost_deltas)
            if effect_of(row) in {"regressed", "ignored", "acknowledged_not_followed"}
        )
        effectiveness[kind] = {
            "issued": issued,
            "compliance_rate": followed / issued if issued else 0.0,
            "improvement_rate": improved / issued if issued else 0.0,
            "override_rate": overrides / issued if issued else 0.0,
            "mean_ter_delta": sum(ter_deltas) / issued if issued else 0.0,
            "mean_waste_delta": sum(waste_deltas) / issued if issued else 0.0,
            "mean_cost_delta_usd": sum(cost_deltas) / issued if issued else 0.0,
            "total_cost_saved_usd": saved,
            "total_cost_wasted_usd": wasted,
        }
    total_saved = sum(v["total_cost_saved_usd"] for v in effectiveness.values())
    total_wasted = sum(v["total_cost_wasted_usd"] for v in effectiveness.values())
    return {
        "lesson_count": len(rows),
        "pattern_counts": dict(pattern_counts),
        "repository_counts": dict(repository_counts),
        "scenarios": scenarios,
        "outcome_count": len(outcomes),
        "intervention_effectiveness": effectiveness,
        "total_estimated_cost_saved_usd": total_saved,
        "total_estimated_cost_wasted_usd": total_wasted,
        "outcome_rows": outcomes,
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _build_improvement_bar_chart(effectiveness: dict[str, dict[str, Any]]) -> str:
    """Return an inline SVG improvement-rate chart or a friendly empty state."""
    import html

    if not effectiveness:
        return '<p class="empty-state">No intervention data yet.</p>'
    items = sorted(effectiveness.items())
    row_height = 42
    width = 760
    label_width = 190
    chart_width = 500
    height = 35 + row_height * len(items)
    parts = [
        f'<svg class="chart" role="img" aria-label="Improvement rate by intervention type" viewBox="0 0 {width} {height}">'
    ]
    for index, (kind, metrics) in enumerate(items):
        rate = max(0.0, min(1.0, float(metrics.get("improvement_rate", 0.0))))
        y = 24 + index * row_height
        bar_width = chart_width * rate
        css_class = "good" if rate >= 0.7 else "warn" if rate >= 0.4 else "bad"
        parts.append(
            f'<text x="0" y="{y + 15}" class="chart-label">{html.escape(kind)}</text>'
            f'<rect x="{label_width}" y="{y}" width="{chart_width}" height="20" class="track" rx="4" />'
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" height="20" class="bar {css_class}" rx="4" />'
            f'<text x="{label_width + chart_width + 8}" y="{y + 15}" class="chart-value">{rate:.0%}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _weekly_cost_buckets(
    outcome_rows: list[dict[str, Any]],
) -> list[tuple[str, float, float]]:
    """Bucket estimated saved and wasted cost by ISO week."""
    buckets: dict[str, list[float]] = {}
    for row in outcome_rows:
        raw = row.get("evaluated_at") or row.get("issued_at") or row.get("timestamp")
        if not isinstance(raw, str):
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        year, week, _ = dt.isocalendar()
        key = f"{year}-W{week:02d}"
        delta = float(row.get("deltas", {}).get("estimated_cost_waste_usd", 0.0))
        effect = str(row.get("effect", ""))
        saved = max(delta, 0.0) if effect == "improved" else 0.0
        wasted = (
            abs(min(delta, 0.0))
            if effect in {"regressed", "ignored", "acknowledged_not_followed"}
            else 0.0
        )
        current = buckets.setdefault(key, [0.0, 0.0])
        current[0] += saved
        current[1] += wasted
    return [(key, values[0], values[1]) for key, values in sorted(buckets.items())]


def _build_cost_trend_chart(outcome_rows: list[dict[str, Any]]) -> str:
    """Return an inline SVG weekly saved-vs-wasted cost chart."""
    buckets = _weekly_cost_buckets(outcome_rows)
    if not buckets:
        return '<p class="empty-state">No dated cost outcomes yet.</p>'
    width, height = 760, 280
    left, top, bottom = 55, 25, 48
    plot_h = height - top - bottom
    max_value = max(max(saved, wasted) for _, saved, wasted in buckets) or 1.0
    if len(buckets) == 1:
        label, saved, wasted = buckets[0]
        scale = plot_h / max_value
        bar_w = 120
        base_y = top + plot_h
        return (
            f'<svg class="chart" role="img" aria-label="Estimated cost saved versus wasted" viewBox="0 0 {width} {height}">'
            f'<line x1="{left}" y1="{base_y}" x2="720" y2="{base_y}" class="axis" />'
            f'<rect x="210" y="{base_y - saved * scale:.1f}" width="{bar_w}" height="{saved * scale:.1f}" class="bar good" />'
            f'<rect x="410" y="{base_y - wasted * scale:.1f}" width="{bar_w}" height="{wasted * scale:.1f}" class="bar bad" />'
            f'<text x="270" y="{base_y + 22}" text-anchor="middle" class="chart-label">saved</text>'
            f'<text x="470" y="{base_y + 22}" text-anchor="middle" class="chart-label">wasted</text>'
            f'<text x="370" y="{height - 8}" text-anchor="middle" class="chart-value">{label}</text>'
            "</svg>"
        )
    step = (width - left - 35) / (len(buckets) - 1)
    scale = plot_h / max_value
    base_y = top + plot_h
    saved_points = []
    wasted_points = []
    labels = []
    for index, (label, saved, wasted) in enumerate(buckets):
        x = left + index * step
        saved_points.append(f"{x:.1f},{base_y - saved * scale:.1f}")
        wasted_points.append(f"{x:.1f},{base_y - wasted * scale:.1f}")
        labels.append(
            f'<text x="{x:.1f}" y="{base_y + 22}" text-anchor="middle" class="chart-label">{label}</text>'
        )
    return (
        f'<svg class="chart" role="img" aria-label="Estimated weekly cost saved versus wasted" viewBox="0 0 {width} {height}">'
        f'<line x1="{left}" y1="{base_y}" x2="725" y2="{base_y}" class="axis" />'
        f'<polyline points="{" ".join(saved_points)}" class="line good-stroke" fill="none" />'
        f'<polyline points="{" ".join(wasted_points)}" class="line bad-stroke" fill="none" />'
        + "".join(labels)
        + '<text x="60" y="16" class="chart-value">saved</text><text x="140" y="16" class="chart-value">wasted</text></svg>'
    )


def build_effectiveness_dashboard_html(
    trends: dict[str, Any],
    *,
    title: str = "TER Intervention Effectiveness",
    tuning_preview: dict[str, Any] | None = None,
) -> str:
    """Build a dependency-free static effectiveness dashboard."""
    import html

    effectiveness = trends.get("intervention_effectiveness", {})
    issued = sum(int(v.get("issued", 0)) for v in effectiveness.values())
    improved = sum(
        int(v.get("issued", 0)) * float(v.get("improvement_rate", 0.0))
        for v in effectiveness.values()
    )
    overall = improved / issued if issued else 0.0
    rows = []
    for kind, m in sorted(effectiveness.items()):
        rows.append(
            f"<tr><td>{html.escape(kind)}</td><td>{int(m.get('issued', 0))}</td><td>{float(m.get('compliance_rate', 0)):.0%}</td><td>{float(m.get('improvement_rate', 0)):.0%}</td><td>{float(m.get('override_rate', 0)):.0%}</td><td>{float(m.get('mean_ter_delta', 0)):+.3f}</td><td>{float(m.get('mean_waste_delta', 0)):+.3f}</td><td>~${float(m.get('mean_cost_delta_usd', 0)):.4f}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="8">No intervention outcome data yet.</td></tr>')
    scenarios = "".join(
        f"<li><strong>{html.escape(str(x.get('pattern_type', '')))}</strong>: {html.escape(str(x.get('message', '')))}</li>"
        for x in trends.get("scenarios", [])
    )
    tuning_html = '<p class="empty-state">No threshold changes recommended.</p>'
    if tuning_preview:
        applied = tuning_preview.get("applied_config")
        changes = tuning_preview.get("changes", [])
        blocks = []
        if applied:
            applied_rows = "".join(
                f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                for k, v in sorted(applied.items())
            )
            blocks.append(
                f'<h3>Applied <span class="badge">Applied</span></h3><table><tbody>{applied_rows}</tbody></table>'
            )
        if changes:
            change_rows = "".join(
                f"<tr><td>{html.escape(str(c['field']))}</td><td>{c['old_value']} → {c['new_value']}</td><td>{html.escape(str(c['reason']))}</td></tr>"
                for c in changes
            )
            blocks.append(
                f"<h3>Pending preview (not yet applied)</h3><table><thead><tr><th>Field</th><th>Change</th><th>Evidence</th></tr></thead><tbody>{change_rows}</tbody></table>"
            )
        elif not applied:
            blocks.append('<p class="empty-state">No changes recommended.</p>')
        tuning_html = "".join(blocks)
    improvement_chart = _build_improvement_bar_chart(effectiveness)
    cost_chart = _build_cost_trend_chart(list(trends.get("outcome_rows", [])))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>:root{{--text:#172033;--border:#d8dee9;--muted:#f4f6f8;--good:#2e8b57;--warn:#c58a00;--bad:#c64242}}body{{font-family:system-ui;margin:2rem;color:var(--text)}}.cards{{display:flex;gap:1rem;flex-wrap:wrap}}.card{{padding:1rem;border:1px solid var(--border);border-radius:10px;min-width:180px}}.chart{{width:100%;max-width:900px;height:auto;border:1px solid var(--border);border-radius:10px;padding:.75rem;box-sizing:border-box}}.track{{fill:var(--muted)}}.good{{fill:var(--good)}}.warn{{fill:var(--warn)}}.bad{{fill:var(--bad)}}.good-stroke{{stroke:var(--good)}}.bad-stroke{{stroke:var(--bad)}}.line{{stroke-width:3}}.axis{{stroke:#8b95a5}}.chart-label,.chart-value{{font-size:12px;fill:var(--text)}}.badge{{font-size:.75rem;background:var(--good);color:white;padding:.15rem .45rem;border-radius:999px}}.empty-state{{padding:1rem;background:var(--muted);border-radius:8px}}table{{border-collapse:collapse;width:100%;margin-top:1rem}}th,td{{padding:.65rem;border-bottom:1px solid #ddd;text-align:left}}th{{background:var(--muted)}}section{{margin-top:2rem}}</style></head><body><h1>{html.escape(title)}</h1><div class='cards'><div class='card'><b>{issued}</b><br>interventions</div><div class='card'><b>{overall:.0%}</b><br>improvement rate</div><div class='card'><b>~${float(trends.get("total_estimated_cost_saved_usd", 0)):.4f}</b><br>estimated saved</div><div class='card'><b>~${float(trends.get("total_estimated_cost_wasted_usd", 0)):.4f}</b><br>estimated wasted</div></div><section><h2>Improvement rate</h2>{improvement_chart}</section><section><h2>Estimated cost over time</h2>{cost_chart}</section><section><h2>Intervention effectiveness</h2><table><thead><tr><th>Type</th><th>Issued</th><th>Compliance</th><th>Improved</th><th>Override</th><th>Mean TER Δ</th><th>Mean waste Δ</th><th>Mean cost Δ</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section><section><h2>Threshold tuning</h2>{tuning_html}</section><section><h2>Recurring scenarios</h2><ul>{scenarios or "<li>No recurring scenarios yet.</li>"}</ul></section></body></html>"""
