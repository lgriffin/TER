#
### Dashboard restoration
- Restored the original rich, self-contained Plotly portfolio dashboard.
- Preserved all Phase 1 charts and tables.
- Added Phase 2 findings below the original visuals rather than replacing them.
- Added configurable 5%/10% TER distribution buckets through `--ter-buckets`.
 Changelog

All notable public changes to TER are documented in this file. The project follows Semantic Versioning.

## [2.0.0] - 2026-07-21

### Added

- Standalone interactive HTML reports with scorecards, token composition, phase distribution, span timeline, alignment-confidence visualization, diagnostics, span inspection, and embedded JSON export.
- Public release hygiene documentation and reproducible package validation.

### Fixed

- User-origin prompt content is excluded from TER output scoring.
- Claude queue and metadata records are not treated as generated model output.
- Embedded report data is escaped to prevent script-termination injection.

### Changed

- Promoted the internally developed v16 codebase to the first cleaned public release, version `2.0.0`.
- TER scoring applies to assistant-origin output spans; user messages remain available for intent and input analysis.

### Known limitations

- TER classifications are heuristic estimates and should not be treated as human-validated ground truth.
- Embedding-based analysis requires the optional `embeddings` dependency.
- Low-confidence classifications should be reviewed alongside the report diagnostics.

## 2.0.1

- Added `ter batch` for recursive, parallel folder analysis.
- Added resumable per-session outputs with atomic writes.
- Added result-schema and token-invariant validation.
- Added consolidated `all-results.jsonl`, `summary.json`, and `manifest.json` artifacts.
- Added a dependency-free, self-contained HTML portfolio dashboard with configurable TER buckets.

## 2.0.2

- Added Phase 2 explainable static pattern detection.
- Integrated Phase 2 findings into batch result JSON, summary aggregation, and HTML dashboards.
- Added findings-by-signal and findings-by-severity charts and a searchable evidence table.
- Included the standalone dashboard generation script under `scripts/`.
- Updated README and release documentation.

## 2.0.2.1

- Complete Phase 2 hook intervention with rolling live-efficiency degradation,
  waste acceleration, repeated-failure detection, cooldowns, and mandatory
  replan guidance.
