"""Batch ingestion, aggregation, validation, and static HTML reporting."""

from __future__ import annotations

import concurrent.futures
import html
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .analyze_pipeline import analyze_session, default_analyze_args
from .formatter_json import ter_result_to_dict
from .phase2_signals import analyze_session_signals
from .portfolio_dashboard import make_dashboard


@dataclass(frozen=True)
class BatchItem:
    input_path: str
    output_path: str
    status: str
    error: str | None = None


def discover_sessions(input_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def _analyze_one(input_path: str, output_path: str, force: bool) -> BatchItem:
    source = Path(input_path)
    target = Path(output_path)
    if target.exists() and target.stat().st_size > 0 and not force:
        try:
            json.loads(target.read_text(encoding="utf-8"))
            return BatchItem(input_path, output_path, "skipped")
        except (OSError, json.JSONDecodeError):
            pass

    try:
        args = default_analyze_args(str(source))
        result = analyze_session(args)
        payload = ter_result_to_dict(result)
        payload["phase2_analysis"] = analyze_session_signals(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)
        return BatchItem(input_path, output_path, "completed")
    except Exception as exc:  # worker boundary: capture all failures in manifest
        return BatchItem(input_path, output_path, "failed", str(exc))


def validate_result(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["result is not a JSON object"]
    required = {
        "session_id": str,
        "aggregate_ter": (int, float),
        "phase_scores": dict,
        "total_tokens": (int, float),
        "aligned_tokens": (int, float),
        "waste_tokens": (int, float),
    }
    for key, expected in required.items():
        if key not in payload:
            errors.append(f"missing field: {key}")
        elif not isinstance(payload[key], expected):
            errors.append(f"invalid type for {key}")
    total = payload.get("total_tokens")
    aligned = payload.get("aligned_tokens")
    waste = payload.get("waste_tokens")
    if all(isinstance(v, (int, float)) for v in (total, aligned, waste)):
        if total < 0 or aligned < 0 or waste < 0:
            errors.append("token counts must be non-negative")
        if aligned + waste != total:
            errors.append("aligned_tokens + waste_tokens != total_tokens")
    score = payload.get("aggregate_ter")
    if isinstance(score, (int, float)) and not 0 <= score <= 1:
        errors.append("aggregate_ter outside [0, 1]")
    return errors


def load_results(result_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("**/*.ter.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_result(payload)
            if errors:
                invalid.append({"path": str(path), "errors": errors})
            else:
                payload["_result_path"] = str(path)
                results.append(payload)
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "errors": [str(exc)]})
    return results, invalid


def write_combined_jsonl(results: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            clean = {k: v for k, v in result.items() if not k.startswith("_")}
            handle.write(json.dumps(clean, separators=(",", ":")) + "\n")


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    total_tokens = sum(float(r.get("total_tokens", 0)) for r in results)
    aligned_tokens = sum(float(r.get("aligned_tokens", 0)) for r in results)
    waste_tokens = sum(float(r.get("waste_tokens", 0)) for r in results)
    ters = [float(r.get("aggregate_ter", 0)) for r in results]
    category_tokens: Counter[str] = Counter()
    category_sessions: Counter[str] = Counter()
    phase_waste: Counter[str] = Counter()
    phase_values: dict[str, list[float]] = {"reasoning": [], "tool_use": [], "generation": []}
    for result in results:
        summary = result.get("waste_summary") or {}
        for category, value in (summary.get("waste_by_category") or {}).items():
            numeric = float(value)
            category_tokens[str(category)] += numeric
            if numeric > 0:
                category_sessions[str(category)] += 1
        for phase, value in (summary.get("waste_by_phase") or {}).items():
            phase_waste[str(phase)] += float(value)
        scores = result.get("phase_scores") or {}
        for phase in phase_values:
            value = scores.get(phase)
            if isinstance(value, (int, float)):
                phase_values[phase].append(float(value))

    signal_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    sessions_with_findings = 0
    total_findings = 0
    for result in results:
        analysis = result.get("phase2_analysis") or {}
        finding_count = int(analysis.get("finding_count", 0) or 0)
        total_findings += finding_count
        sessions_with_findings += finding_count > 0
        signal_counts.update(analysis.get("signal_counts") or {})
        severity_counts.update(analysis.get("severity_counts") or {})

    sorted_ters = sorted(ters)
    def percentile(fraction: float) -> float:
        if not sorted_ters:
            return 0.0
        position = fraction * (len(sorted_ters) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return sorted_ters[lower]
        return sorted_ters[lower] + (sorted_ters[upper] - sorted_ters[lower]) * (position - lower)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions": count,
        "total_tokens": int(total_tokens),
        "aligned_tokens": int(aligned_tokens),
        "waste_tokens": int(waste_tokens),
        "weighted_ter": aligned_tokens / total_tokens if total_tokens else 0.0,
        "average_session_ter": sum(ters) / count if count else 0.0,
        "median_ter": percentile(0.5),
        "p10_ter": percentile(0.1),
        "p90_ter": percentile(0.9),
        "sessions_with_waste": sum(float(r.get("waste_tokens", 0)) > 0 for r in results),
        "perfect_sessions": sum(float(r.get("aggregate_ter", 0)) == 1.0 for r in results),
        "phase_averages": {
            phase: sum(values) / len(values) if values else 0.0
            for phase, values in phase_values.items()
        },
        "waste_by_category": dict(category_tokens.most_common()),
        "affected_sessions_by_category": dict(category_sessions.most_common()),
        "waste_by_phase": dict(phase_waste.most_common()),
        "phase2": {"total_findings": total_findings, "sessions_with_findings": sessions_with_findings, "signal_counts": dict(signal_counts.most_common()), "severity_counts": dict(severity_counts)},
    }


def _bar_chart_svg(labels: list[str], values: list[float], title: str, height: int = 320) -> str:
    width, left, right, top, bottom = 960, 70, 20, 42, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(values, default=1) or 1
    gap = 6
    bar_w = max(2, (plot_w - gap * max(0, len(values) - 1)) / max(1, len(values)))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    parts.append(f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * (bar_w + gap)
        bar_h = (value / maximum) * plot_h
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" class="bar"><title>{html.escape(label)}: {value:,.0f}</title></rect>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 16}" transform="rotate(45 {x + bar_w / 2:.1f} {top + plot_h + 16})" class="tick">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_dashboard_html(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    bucket_count: int = 20,
) -> str:
    """Build the rich self-contained Plotly portfolio dashboard.

    ``summary`` remains part of the public signature for compatibility with
    v2.0.1/v2.0.2 callers; the dashboard recomputes display metrics directly
    from the validated result records.
    """
    del summary
    return make_dashboard(results, ter_bucket_count=bucket_count)


def run_batch(input_dir: Path, output_dir: Path, workers: int | None = None, recursive: bool = True, force: bool = False, bucket_count: int = 20) -> dict[str, Any]:
    sessions = discover_sessions(input_dir, recursive=recursive)
    if not sessions:
        raise ValueError(f"No .jsonl session files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for session in sessions:
        relative = session.relative_to(input_dir)
        target = output_dir / relative.parent / f"{relative.stem}.ter.json"
        jobs.append((str(session), str(target), force))
    worker_count = workers or min(8, os.cpu_count() or 1)
    items: list[BatchItem] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_analyze_one, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            items.append(future.result())
    items.sort(key=lambda item: item.input_path)
    results, invalid = load_results(output_dir)
    write_combined_jsonl(results, output_dir / "all-results.jsonl")
    summary = aggregate_results(results)
    summary["inputs_discovered"] = len(sessions)
    summary["completed"] = sum(item.status == "completed" for item in items)
    summary["skipped"] = sum(item.status == "skipped" for item in items)
    summary["failed"] = sum(item.status == "failed" for item in items)
    summary["invalid_outputs"] = len(invalid)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "generated_at": summary["generated_at"],
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "workers": worker_count,
        "items": [asdict(item) for item in items],
        "invalid_outputs": invalid,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "ter-dashboard.html").write_text(build_dashboard_html(results, summary, bucket_count=bucket_count), encoding="utf-8")
    return summary
