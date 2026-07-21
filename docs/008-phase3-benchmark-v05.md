# Phase 3 Benchmark and Calibration Layer — TER v05

TER v05 introduces an empirical evaluation layer without changing production classifier thresholds.

## Benchmark unit

Each JSONL line represents the smallest independently defensible annotation unit: a sentence group, paragraph, tool call, tool result, or action transition.

Required fields:

- `id`: globally unique record identifier
- `session_id`: source session identifier
- `phase`: `reasoning`, `tool_use`, or `generation`
- `gold_label`: one of the TER `SpanLabel` values

Prediction fields:

- `predicted_label`: classifier output, or
- `score`: numeric waste-evidence score in `[0, 1]`

Recommended optional fields:

- `tokens`
- `category`
- `text`
- `annotator`
- source provenance and model/version metadata

## Annotation guidance

Annotators should label observable work, not writing style. Corrective iteration is aligned when it responds to new evidence. Similar-looking tool calls are not duplicates when parameters, inputs, or expected results materially change. Waste labels require a concrete signal such as duplicated work, abandoned work with no information gain, or unnecessary elaboration.

Use at least two annotators for the frozen benchmark. Resolve disagreements separately and preserve the original annotations for agreement analysis.

## CLI

```bash
ter benchmark benchmarks/example_annotations.jsonl
ter benchmark benchmarks/example_annotations.jsonl --format json
ter benchmark benchmark.jsonl --threshold 0.85 --bootstrap-samples 5000
```

The report includes binary waste detection, token-weighted metrics, per-label metrics, a multiclass confusion matrix, bootstrap 95% intervals, and an advisory F0.5 threshold recommendation.

Threshold recommendations do not modify production defaults. A threshold change should require a frozen benchmark, before/after results, false-positive review, and release approval.
