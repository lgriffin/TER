#!/usr/bin/env python3
"""Convert reviewed annotation CSV files into TER benchmark JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VALID_LABELS = {
    "aligned_reasoning",
    "redundant_reasoning",
    "aligned_tool_call",
    "unnecessary_tool_call",
    "aligned_response",
    "over_explanation",
}


def load_reviewed_csv(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for line_number, row in enumerate(reader, start=2):
            gold_label = (row.get("gold_label") or "").strip()

            if not gold_label:
                continue

            if gold_label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{line_number}: invalid gold_label "
                    f"{gold_label!r}"
                )

            predicted_label = (row.get("predicted_label") or "").strip()
            if predicted_label not in VALID_LABELS:
                raise ValueError(
                    f"{path}:{line_number}: invalid predicted_label "
                    f"{predicted_label!r}"
                )

            record: dict[str, object] = {
                "id": row["id"],
                "session_id": row["session_id"],
                "phase": row["phase"],
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "tokens": int(row.get("tokens") or 1),
                "text": (row.get("text") or "").replace("\\n", "\n"),
            }

            score = (row.get("score") or "").strip()
            if score:
                record["score"] = float(score)

            annotator = (row.get("annotator") or "").strip()
            if annotator:
                record["annotator"] = annotator

            notes = (row.get("notes") or "").strip()
            if notes:
                record["notes"] = notes

            source_uuid = (row.get("source_message_uuid") or "").strip()
            if source_uuid:
                record["source_message_uuid"] = source_uuid

            block_type = (row.get("block_type") or "").strip()
            if block_type:
                record["block_type"] = block_type

            records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Reviewed annotation CSV files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output benchmark JSONL",
    )
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for input_path in args.inputs:
        for record in load_reviewed_csv(input_path):
            record_id = str(record["id"])

            if record_id in seen_ids:
                raise ValueError(f"Duplicate benchmark ID: {record_id}")

            seen_ids.add(record_id)
            records.append(record)

    if not records:
        raise SystemExit("No reviewed records were found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
