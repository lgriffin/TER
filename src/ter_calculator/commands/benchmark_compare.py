"""Release benchmark comparison command."""

from __future__ import annotations

from pathlib import Path

from ..evaluation import evaluate_benchmark, load_benchmark
from ..regression import (
    compare_benchmark_reports,
    dump_regression_json,
    format_regression_report,
)


def _cmd_benchmark_compare(args) -> int:
    baseline = evaluate_benchmark(
        load_benchmark(args.baseline_path),
        threshold=args.baseline_threshold,
        bootstrap_samples=0,
        seed=args.seed,
    )
    candidate = evaluate_benchmark(
        load_benchmark(args.candidate_path),
        threshold=args.candidate_threshold,
        bootstrap_samples=0,
        seed=args.seed,
    )
    report = compare_benchmark_reports(
        baseline,
        candidate,
        minimum_precision=args.minimum_precision,
        maximum_precision_drop=args.maximum_precision_drop,
        maximum_recall_drop=args.maximum_recall_drop,
        maximum_f0_5_drop=args.maximum_f0_5_drop,
        maximum_accuracy_drop=args.maximum_accuracy_drop,
        maximum_false_positive_increase=args.maximum_false_positive_increase,
    )
    rendered = (
        dump_regression_json(report)
        if args.output_format == "json"
        else format_regression_report(report)
    )
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report.passed else 2
