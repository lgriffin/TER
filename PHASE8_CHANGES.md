# Phase 8 — Reproducible benchmarking and release validation (v2.0.8)

Phase 8 closes the v2.0 roadmap with deterministic release evidence and regression gates.

## Release validation

```bash
ter release-check ter-results \
  --minimum-sessions 100 \
  --minimum-ter 0.90 \
  --maximum-waste-ratio 0.10 \
  --output ter-release-manifest.json
```

The manifest records aggregate metrics, TER distribution percentiles, a canonical results fingerprint, and SHA-256 checksums for every `*.ter.json` input.

## Baseline regression gates

```bash
ter release-check ter-results \
  --baseline previous-release-manifest.json \
  --maximum-ter-drop 0.01 \
  --maximum-waste-increase 0.01
```

The command exits with code `2` when an absolute quality threshold or baseline regression limit is violated.

## Reproducibility

Input records are normalized and sorted before hashing, so the results fingerprint is stable regardless of filesystem traversal or JSON file ordering. The manifest intentionally omits wall-clock timestamps.
