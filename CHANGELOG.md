# Changelog

All notable public changes to TER are documented in this file. The project follows Semantic Versioning.

## Unreleased

### Added

- Added recursive, parallel batch analysis with resumable outputs.
- Added portfolio metrics, manifests, schema validation, and HTML dashboards.
- Added deterministic Phase 2 pattern detection with inspectable evidence.
- Added rolling live-efficiency and waste-acceleration detection.
- Added repeated-failure detection, intervention cooldowns, and replan guidance.
- Added Phase 3 and Phase 4 analysis capabilities.

### Changed

- Extended the existing portfolio dashboard with later-phase findings.
- Added configurable TER distribution buckets.

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
