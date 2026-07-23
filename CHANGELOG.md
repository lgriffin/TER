# Changelog

All notable public changes to TER are documented in this file. The project follows Semantic Versioning.

## [3.0.0] - 2026-07-22

### Added

- Expanded context orchestration with fragment storage, dependency graphs, adaptive budgeting, and delta composition.
- Acceleration, evaluation, regression, uncertainty, and real-time monitoring capabilities.
- Broader CLI, dashboard, reporting, benchmark, annotation, and project-analysis workflows.
- Additional automated coverage across unit, integration, and feature-level behavior.

### Changed

- Promoted the expanded TER codebase to major release `3.0.0`.
- Updated public documentation and package/runtime version metadata for the v3 release line.
- Consolidated the current architecture and release guidance around the broader TER analysis platform.

### Compatibility

- Python support remains `>=3.11,<3.14`.
- TER scores remain heuristic decision-support signals rather than ground-truth judgments.
- Optional embedding and LLM functionality continues to require the corresponding extras and external model/API access.

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
