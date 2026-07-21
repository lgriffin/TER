"""Phase 7 CI/CD and ecosystem integration command."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..batch_analysis import load_results
from ..integrations import (
    IntegrationGate,
    atomic_write_json,
    atomic_write_text,
    build_github_annotations,
    build_sarif,
    build_step_summary,
    evaluate_gate,
)


def _cmd_integrate(args) -> int:
    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        raise ValueError(f"Integration result directory does not exist: {result_dir}")
    results, invalid = load_results(result_dir)
    if not results:
        raise ValueError(f"No valid .ter.json result files found under {result_dir}")
    gate = IntegrationGate(args.minimum_ter, args.maximum_waste_ratio)
    gate_result = evaluate_gate(results, gate)

    output = (
        Path(args.output) if args.output else _default_output(result_dir, args.format)
    )
    if args.format == "sarif":
        atomic_write_json(output, build_sarif(results, gate_result))
    elif args.format == "github":
        atomic_write_text(output, build_github_annotations(gate_result) + "\n")
    elif args.format == "summary":
        summary = build_step_summary(gate_result, str(result_dir))
        atomic_write_text(output, summary)
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            with Path(step_summary).open("a", encoding="utf-8") as handle:
                handle.write(summary + "\n")
    else:
        atomic_write_json(
            output,
            {
                "version": "2.0.8",
                "source": str(result_dir),
                "invalid_outputs": len(invalid),
                "gate": gate_result.to_dict(),
            },
        )

    if not args.quiet:
        print(f"Integration artifact written to {output}")
        print(json.dumps(gate_result.to_dict(), sort_keys=True))
    return 0 if gate_result.passed else 2


def _default_output(result_dir: Path, output_format: str) -> Path:
    names = {
        "json": "ter-integration.json",
        "sarif": "ter-results.sarif",
        "github": "ter-github-annotations.txt",
        "summary": "ter-step-summary.md",
    }
    return result_dir / names[output_format]
