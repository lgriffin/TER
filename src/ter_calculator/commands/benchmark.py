"""Benchmark command implementation."""

from __future__ import annotations

from pathlib import Path

from ..evaluation import (
    dump_report_json,
    evaluate_benchmark,
    format_benchmark_report,
    load_benchmark,
)


def _cmd_benchmark(args) -> int:
    records = load_benchmark(args.benchmark_path)
    report = evaluate_benchmark(
        records,
        threshold=args.threshold,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        beta=args.beta,
        minimum_precision=args.minimum_precision,
    )
    rendered = (
        dump_report_json(report)
        if args.output_format == "json"
        else format_benchmark_report(report)
    )
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0
