#!/usr/bin/env python3
"""Export TER session spans to a CSV file for human annotation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ter_calculator.analyze_pipeline import analyze_session, default_analyze_args


def export_session(session_path: Path, output_path: Path) -> None:
    args = default_analyze_args(str(session_path))

    # These optional analyses are unnecessary for annotation export.
    args.no_waste_patterns = True
    args.no_input_analysis = True

    result = analyze_session(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "session_id",
                "phase",
                "tokens",
                "text",
                "predicted_label",
                "score",
                "gold_label",
                "annotator",
                "notes",
                "source_message_uuid",
                "block_type",
            ],
        )
        writer.writeheader()

        for index, classified in enumerate(result.classified_spans):
            span = classified.span

            writer.writerow(
                {
                    "id": f"{result.session_id}-{index:04d}",
                    "session_id": result.session_id,
                    "phase": span.phase.value,
                    "tokens": span.token_count,
                    "text": span.text.replace("\n", "\\n"),
                    "predicted_label": classified.label.value,
                    # Confidence is not a calibrated waste probability, but it
                    # can still be retained for later exploratory analysis.
                    "score": f"{classified.confidence:.6f}",
                    "gold_label": "",
                    "annotator": "",
                    "notes": "",
                    "source_message_uuid": span.source_message_uuid,
                    "block_type": span.block_type,
                }
            )

    print(f"Exported {len(result.classified_spans)} units to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path, help="Claude Code JSONL session")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output annotation CSV",
    )
    args = parser.parse_args()

    export_session(args.session, args.output)


if __name__ == "__main__":
    main()
