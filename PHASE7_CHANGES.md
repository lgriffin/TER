# Phase 7 — Ecosystem integrations (v2.0.7)

Phase 7 turns TER analysis output into portable CI/CD artifacts and machine-readable quality gates.

## Integration command

```bash
ter integrate ter-results --format json
ter integrate ter-results --format sarif --output ter-results.sarif
ter integrate ter-results --format github
ter integrate ter-results --format summary
```

The command reads validated `*.ter.json` files, computes portfolio-level metrics, writes an artifact atomically, and exits with code `2` when a configured gate fails.

## Quality gates

```bash
ter integrate ter-results \
  --minimum-ter 0.80 \
  --maximum-waste-ratio 0.20 \
  --format sarif
```

Supported outputs:

- `json`: compact integration payload for automation and dashboards.
- `sarif`: SARIF 2.1.0 output suitable for code-scanning ingestion.
- `github`: GitHub Actions workflow annotations.
- `summary`: Markdown report, also appended to `GITHUB_STEP_SUMMARY` when available.

## Privacy and portability

Integration artifacts contain aggregate metrics, session identifiers, and waste counts only; raw prompts and session transcripts are never exported.
