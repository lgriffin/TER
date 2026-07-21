# 2.0.4.2

- Restore missing Phase 2 batch and dashboard modules in the release archive.
- Resolve strict mypy errors in dashboard rendering, signal analysis, and batch validation.

# 2.0.4.1

- Fix mypy typing for project waste-source selection.
- Add targeted Plotly missing-stub override for dashboard modules.
- Preserve strict typing for the rest of the codebase.

# Changelog

## 2.0.4 — Phase 4 Cross-Session Intelligence

- Added an opt-in SQLite TER history store at `~/.claude/ter/history.db`.
- Added `ter history record`, `list`, `profile`, and `predict`.
- Added `ter dashboard` for TER, token, waste, and cost trends.
- Added privacy-preserving prompt fingerprints; raw prompts are not persisted.
- Added `data/` immediately below `Backup/` in `.gitignore`.
- Updated README with Phase 4 workflows and privacy guidance.

All notable public changes to TER are documented in this file. The project follows Semantic Versioning.

## [2.0.3] - 2026-07-21

### Added

- Phase 3 Claude Code intervention engine spanning `SessionStart`, `PreToolUse`, `PostToolUse`, `PermissionRequest`, `PostToolUseFailure`, and assistant-stop events.
- Adaptive task-complexity budget hints at session start.
- Pre-execution blocking for exact duplicate tool calls with previous-result summaries.
- Reasoning-loop breaker for highly repetitive consecutive assistant messages.
- Permission-denial circuit breaker after repeated denied requests for the same tool.
- Persistent intervention counters and per-session hook metadata.

### Changed

- `ter hook monitor` is now a unified Phase 3 hook handler while remaining backward compatible with existing PostToolUse configurations.
- Hook thresholds can now configure denial counts, reasoning-loop counts, and similarity sensitivity.

### Validation

- Added focused Phase 3 intervention tests covering state persistence, blocking output, budget hints, reasoning loops, and permission loops.

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
