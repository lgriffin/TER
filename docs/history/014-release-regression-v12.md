# TER v12 — Release Regression Gates

TER v12 adds a CI-oriented comparison layer for frozen benchmark predictions.
It compares a baseline release with a candidate release on the same annotated
records and exits non-zero when configured quality gates fail.

## Command

```bash
ter benchmark-compare \
  benchmarks/example_annotations.jsonl \
  benchmarks/example_annotations_candidate.jsonl \
  --minimum-precision 0.90 \
  --maximum-precision-drop 0.00 \
  --maximum-f0-5-drop 0.00 \
  --maximum-false-positive-increase 0
```

Exit codes:

- `0`: every release gate passed
- `2`: at least one quality gate failed
- `1`: invalid input or another command error

## Compared metrics

- Record-level precision, recall, F0.5, and accuracy
- Token-weighted precision, recall, and F0.5
- False-positive and false-negative counts

## Available gates

- Minimum candidate precision
- Maximum precision regression
- Maximum recall regression
- Maximum F0.5 regression
- Maximum accuracy regression
- Maximum false-positive increase

Text output is intended for local review. JSON output is suitable for CI
artifacts and automated release checks.

The command does not alter production thresholds. Real-data gates should only
become mandatory after the benchmark dataset and annotation process are frozen.
