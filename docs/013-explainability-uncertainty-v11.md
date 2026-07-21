# TER v11 — Explainability and Uncertainty

TER v11 makes classifier decisions inspectable and reports the uncertainty that
was previously hidden behind a single TER number.

## Per-span explanations

Every classified span may now include:

- a stable reason code and human-readable summary;
- intent and repetition scores;
- semantic, lexical, entity, and action similarities;
- parameter novelty;
- the applicable repetition threshold;
- the position and excerpt of the strongest matched prior span.

JSON output includes the complete evidence record. Text output shows a compact
review list containing waste and low-confidence classifications.

## Session uncertainty

`compute_ter()` now attaches a deterministic uncertainty report containing:

- mean and token-weighted classification confidence;
- low-confidence token count and share;
- a reproducible 95% span-bootstrap TER interval;
- span count, sample count, method, and reliability level.

The interval measures sensitivity to the observed classified spans. It does not
replace uncertainty from annotation disagreement, model choice, or threshold
calibration. Small sessions are therefore marked as low reliability.

## Compatibility

Existing `ClassifiedSpan` construction remains valid because explanation data is
optional. Existing TER APIs remain unchanged. Production repetition thresholds
are not modified in v11.
