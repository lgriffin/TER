"""Rank sessions by how *inconsistent* / threshold-sensitive their TER classification is.

Idea: a session that gets basically the same waste ratio no matter which
reasonable threshold you pick is a session the current heuristics are
confident about (whether right or wrong, it's stable). A session whose waste
ratio swings wildly when you nudge similarity_threshold / confidence_threshold
a little is a session sitting right on the decision boundary - exactly the
kind of case where the model's calibration is actually being tested, and
therefore the highest-value case to hand-label first.

This does NOT tell you if the classifier is *correct*. It tells you where it
is *unstable*, which is the cheapest available proxy for "worth a human's
time" when you have 200 unlabeled sessions and can't label all of them.

Usage:
    python scripts/labeling_priority.py /path/to/sessions_dir --top 20
    python scripts/labeling_priority.py /path/to/sessions_dir --out priority.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from ter_calculator.analyze_pipeline import analyze_session, default_analyze_args

# Threshold grid to probe. Centered on the project defaults
# (similarity=0.40, confidence=0.75) with +/- perturbations.
SIMILARITY_GRID = [0.30, 0.35, 0.40, 0.45, 0.50]
CONFIDENCE_GRID = [0.65, 0.70, 0.75, 0.80, 0.85]


@dataclass
class SessionResult:
    path: Path
    n_spans: int
    waste_ratios: list[float]
    error: str | None = None

    @property
    def mean_waste_ratio(self) -> float:
        return statistics.mean(self.waste_ratios) if self.waste_ratios else 0.0

    @property
    def spread(self) -> float:
        """Max - min waste ratio across the threshold grid. The instability signal."""
        if len(self.waste_ratios) < 2:
            return 0.0
        return max(self.waste_ratios) - min(self.waste_ratios)

    @property
    def stdev(self) -> float:
        if len(self.waste_ratios) < 2:
            return 0.0
        return statistics.pstdev(self.waste_ratios)


def waste_ratio_for(session_path: str, similarity_threshold: float, confidence_threshold: float) -> tuple[float, int]:
    args = default_analyze_args(session_path)
    args.similarity_threshold = similarity_threshold
    args.confidence_threshold = confidence_threshold
    result = analyze_session(args)
    total = result.total_tokens or 0
    waste = result.waste_tokens or 0
    ratio = (waste / total) if total else 0.0
    return ratio, len(result.classified_spans)


def evaluate_session(path: Path) -> SessionResult:
    ratios: list[float] = []
    n_spans = 0
    try:
        for sim_t in SIMILARITY_GRID:
            for conf_t in CONFIDENCE_GRID:
                ratio, n_spans = waste_ratio_for(str(path), sim_t, conf_t)
                ratios.append(ratio)
    except Exception as exc:  # noqa: BLE001 - we want to keep going and report the failure
        return SessionResult(path=path, n_spans=0, waste_ratios=[], error=f"{type(exc).__name__}: {exc}")
    return SessionResult(path=path, n_spans=n_spans, waste_ratios=ratios)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions_dir", help="Directory containing .jsonl session files")
    parser.add_argument("--top", type=int, default=20, help="How many sessions to print (default 20)")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path for full results")
    args = parser.parse_args(argv)

    sessions_dir = Path(args.sessions_dir)
    files = sorted(sessions_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {sessions_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(files)} session file(s). Probing {len(SIMILARITY_GRID) * len(CONFIDENCE_GRID)} "
          f"threshold combinations each...\n")

    results: list[SessionResult] = []
    for i, f in enumerate(files, 1):
        r = evaluate_session(f)
        results.append(r)
        status = "ERROR" if r.error else f"spread={r.spread:.3f}"
        print(f"[{i}/{len(files)}] {f.name}: {status}")

    ok_results = [r for r in results if r.error is None]
    error_results = [r for r in results if r.error is not None]

    ok_results.sort(key=lambda r: r.spread, reverse=True)

    print("\n=== Top sessions to hand-label first (highest threshold-sensitivity) ===")
    print(f"{'session':45s}  {'spans':>6s}  {'mean_waste':>11s}  {'spread':>8s}  {'stdev':>7s}")
    for r in ok_results[: args.top]:
        print(f"{r.path.name:45s}  {r.n_spans:6d}  {r.mean_waste_ratio:11.3f}  {r.spread:8.3f}  {r.stdev:7.3f}")

    if error_results:
        print(f"\n=== {len(error_results)} session(s) failed to parse/classify (fix or exclude before labeling) ===")
        for r in error_results:
            print(f"{r.path.name}: {r.error}")

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["session", "n_spans", "mean_waste_ratio", "spread", "stdev", "error"])
            for r in results:
                writer.writerow([r.path.name, r.n_spans, f"{r.mean_waste_ratio:.4f}",
                                  f"{r.spread:.4f}", f"{r.stdev:.4f}", r.error or ""])
        print(f"\nFull results written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
