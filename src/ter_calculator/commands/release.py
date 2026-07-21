"""Phase 8 release validation command."""

from __future__ import annotations

import json
from pathlib import Path

from .. import __version__
from ..batch_analysis import load_results
from ..integrations import atomic_write_json, atomic_write_text
from ..release_validation import (
    ReleaseGate,
    build_file_checksums,
    build_release_snapshot,
    build_release_summary,
    evaluate_release,
)


def _cmd_release(args) -> int:
    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        raise ValueError(f"Release result directory does not exist: {result_dir}")
    results, invalid = load_results(result_dir)
    if not results:
        raise ValueError(f"No valid .ter.json result files found under {result_dir}")

    baseline = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = baseline.get("snapshot", baseline)

    snapshot = build_release_snapshot(
        results, version=__version__, source=str(result_dir)
    )
    gate = ReleaseGate(
        minimum_sessions=args.minimum_sessions,
        minimum_weighted_ter=args.minimum_ter,
        maximum_waste_ratio=args.maximum_waste_ratio,
        maximum_weighted_ter_drop=args.maximum_ter_drop,
        maximum_waste_ratio_increase=args.maximum_waste_increase,
    )
    assessment = evaluate_release(snapshot, gate, baseline)
    manifest = {
        "schema_version": 1,
        "snapshot": snapshot,
        "assessment": assessment.to_dict(),
        "invalid_outputs": len(invalid),
        "files": build_file_checksums(result_dir),
    }
    output = (
        Path(args.output) if args.output else result_dir / "ter-release-manifest.json"
    )
    if args.format == "summary":
        atomic_write_text(output, build_release_summary(snapshot, assessment))
    else:
        atomic_write_json(output, manifest)

    if not args.quiet:
        print(f"Release artifact written to {output}")
        print(json.dumps(assessment.to_dict(), sort_keys=True))
    return 0 if assessment.passed else 2
