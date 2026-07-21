"""Batch-analysis CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from ..batch_analysis import (
    aggregate_results,
    build_dashboard_html,
    load_results,
    run_batch,
    write_combined_jsonl,
)


def _cmd_batch(args) -> int:
    summary = run_batch(
        Path(args.input_dir),
        Path(args.output_dir),
        workers=args.workers,
        recursive=not args.no_recursive,
        force=args.force,
        bucket_count=args.ter_buckets,
    )
    if args.output_format == "json":
        print(json.dumps(summary, indent=2))
    elif not args.quiet:
        print(
            f"Analyzed {summary['sessions']} sessions "
            f"({summary['completed']} completed, {summary['skipped']} skipped, "
            f"{summary['failed']} failed)."
        )
        print(f"Dashboard: {Path(args.output_dir) / 'ter-dashboard.html'}")
        print(f"Combined JSONL: {Path(args.output_dir) / 'all-results.jsonl'}")
    return 1 if summary["failed"] or summary["invalid_outputs"] else 0


def _cmd_dashboard(args) -> int:
    result_dir = Path(args.result_dir)
    results, invalid = load_results(result_dir)
    if not results:
        raise ValueError(f"No valid *.ter.json results found under {result_dir}")
    write_combined_jsonl(results, result_dir / "all-results.jsonl")
    summary = aggregate_results(results)
    summary["invalid_outputs"] = len(invalid)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (result_dir / "ter-dashboard.html").write_text(
        build_dashboard_html(results, summary, bucket_count=args.ter_buckets),
        encoding="utf-8",
    )
    if args.output_format == "json":
        print(json.dumps(summary, indent=2))
    elif not args.quiet:
        print(f"Dashboard: {result_dir / 'ter-dashboard.html'}")
        print(f"Sessions included: {len(results)}; invalid outputs: {len(invalid)}")
    return 1 if invalid else 0
