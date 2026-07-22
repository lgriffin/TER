## 3.1.0

- Added opt-in pre-send duplicate/pattern checks for Claude Code `UserPromptSubmit` hooks.
- Added explicit human acknowledgement and override paths, outcome tracking, CLI configuration, tests, and documentation.

# Changelog

All notable public changes to TER are documented in this file. The project follows Semantic Versioning.

## [3.0.0] - 2026-07-22

### Added

- Productionized repository-aware static intelligence, live degradation policies, and feed-forward session learning as the stable TER 3.0 product surface.
- Inline SVG effectiveness charts for intervention improvement rates and weekly estimated cost saved versus wasted.
- Auditable dashboard visibility for applied and pending repository threshold tuning.
- Transparent CLI precedence: explicit threshold flags override tuned values, which override built-in defaults.
- Separate minimal-install and full-extras CI validation paths.
- TER 3.0 migration and release documentation.

### Changed

- Promoted the package and CLI version to `3.0.0`.
- Updated repository-memory index schema marker to `3.0.0`; indexes should be rebuilt after upgrade.
- Consolidated v2 phase functionality into the documented closed-loop workflow: observe → detect → retrieve → intervene → measure → tune.

### Validation

- Ruff formatting and linting passed.
- Mypy passed across 84 source files.
- Full extras-enabled suite: 1,162 tests passed.
- Branch coverage remains enforced at 90% or higher.

## [2.0.16] - 2026-07-22

### Added

- Inline SVG improvement-rate and weekly estimated-cost charts in the effectiveness dashboard.
- Applied and pending threshold-tuning visibility backed by shared change descriptions.
- Full-extras and minimal-install CI coverage.

### Fixed

- Hook threshold precedence now uses explicit `None` sentinels so user-supplied values always override tuned configuration.

## [2.0.15] - 2026-07-22

### Added

- Estimated intervention cost deltas and cost-framed trend reporting.
- Transparent, dry-run-by-default per-repository threshold tuning.
- Static intervention effectiveness dashboard and supporting documentation.

## [2.0.14] - 2026-07-21

### Fixed

- Native Claude Code hooks now derive rolling TER metrics incrementally from `transcript_path` or `transcript` instead of requiring external metric injection.
- Transcript byte offsets and lightweight rolling counters are persisted in `HookSessionState`; malformed or unavailable transcripts fail silently.
- `observe` policy mode now consumes and records pending interventions without injecting guidance or system messages.
- Outcome classification now uses `neutral` for followed interventions without a meaningful metric change.

### Added

- Regression tests for incremental transcript processing, transcript failures, cross-mode pending-intervention delivery, and all effect-classification branches and boundaries.

### Validation

- Full project validation is documented with the release artifact.

## [2.0.11] - 2026-07-21

### Added

- Sustained TER/waste policy evaluation with configurable context-refresh and replanning thresholds.
- Policy persistence windows and refresh/replan cooldowns to avoid reacting to transient metric noise.
- One-time pending-intervention delivery through Claude Code hooks.
- Intervention identifiers, baseline snapshots, acknowledgement/compliance detection, and effect classification.
- Before/after TER and waste-ratio measurements for intervention evaluation.
- Cross-session effectiveness metrics, including compliance rate, improvement rate, and median metric deltas.
- Direct policy evaluation from the live `SessionMonitor` signal path.
- Regression coverage for transient dips, sustained degradation, cooldowns, pending intervention consumption, policy modes, and outcome classification.

### Changed

- `ter hook monitor` now exposes policy-mode, TER-drop, waste-ratio, persistence-window, and cooldown options.
- Repository-memory trend reports now aggregate intervention outcomes as well as recurring lesson patterns.
- Documentation now describes the full detection → decision → delivery → evaluation loop.

### Validation

- Ruff formatting and linting passed.
- Mypy passed across 83 source files.
- Full test suite: 1,115 tests passed.

## [2.0.10] - 2026-07-21

### Added

- Project-scoped repository-memory retrieval in `SessionStart` and `UserPromptSubmit` hooks.
- Pre-action guidance for similar implementations, prior fixes, and duplicate patterns.
- Lightweight semantic duplicate grouping alongside exact fingerprints.
- Durable session-lesson and intervention-outcome JSONL stores.
- Cross-session `ter memory trends` scenario aggregation.
- Hook controls for memory paths, retrieval thresholds, policy mode, and persistence paths.

## [2.0.9] - 2026-07-21

### Added

- Local repository-memory indexing for source, documentation, and Git history.
- Semantic retrieval with source paths, line provenance, and confidence scores.
- Duplicate-pattern and prior defect/fix risk flags.
- `ter memory index`, `ter memory search`, and `ter memory inspect`.
- Phase 9 documentation and regression tests.

## [2.0.8] - 2026-07-21

### Added

- `ter release-check` for deterministic release manifests and regression gates.
- Canonical result fingerprints and per-file SHA-256 checksums.
- Absolute quality thresholds plus baseline TER-drop and waste-increase limits.
- JSON and Markdown release artifacts, documentation, and regression tests.

## [2.0.7] - 2026-07-21

### Added

- `ter integrate` for CI/CD quality gates.
- JSON, SARIF 2.1.0, GitHub annotation, and Markdown step-summary outputs.
- Weighted TER and waste-ratio release gates with deterministic exit codes.
- Atomic integration artifact writes and Phase 7 regression tests.

## [2.0.6] - 2026-07-21

### Added

- Project-specific learning for TER thresholds, phase weights, token budgets, and intervention policy.
- Optional prompt-neighbor personalization without storing raw prompts.
- Atomic JSON policy export through `ter optimize`.
- Bounded recommendations, confidence tiers, documentation, and regression tests.

## [2.0.5] - 2026-07-21

### Added

- Validated environment-driven runtime configuration.
- SQLite schema versioning, WAL mode, busy timeouts, and integrity checks.
- Restrictive POSIX permissions for local TER state directories and database files.
- Atomic history backup and integrity-checked restore operations.
- `ter doctor` production-readiness diagnostics.

## [2.0.4.2] - 2026-07-21

### Fixed

- Restored missing Phase 2 batch and dashboard modules in the release archive.
- Resolved strict mypy errors in dashboard rendering, signal analysis, and batch validation.

## [2.0.4.1] - 2026-07-21

### Fixed

- Corrected mypy typing for project waste-source selection.
- Added a targeted Plotly missing-stub override for dashboard modules while preserving strict typing elsewhere.

## [2.0.4] - 2026-07-21

### Added

- Opt-in SQLite TER history at `~/.claude/ter/history.db`.
- `ter history record`, `list`, `profile`, and `predict`.
- `ter dashboard` for TER, token, waste, and cost trends.
- Privacy-preserving prompt fingerprints; raw prompts are not persisted.

## [2.0.3] - 2026-07-21

### Added

- Claude Code intervention handling for `SessionStart`, `PreToolUse`, `PostToolUse`, `PermissionRequest`, `PostToolUseFailure`, and assistant-stop events.
- Adaptive task-complexity budget hints at session start.
- Pre-execution blocking for exact duplicate tool calls with previous-result summaries.
- Reasoning-loop and permission-denial circuit breakers.
- Persistent intervention counters and per-session hook metadata.

### Changed

- `ter hook monitor` became the unified hook handler while remaining backward compatible with existing PostToolUse configurations.

## [2.0.0] - 2026-07-21

### Added

- Standalone interactive HTML reports with scorecards, token composition, phase distribution, span timeline, diagnostics, span inspection, and embedded JSON export.
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
- Low-confidence classifications should be reviewed alongside report diagnostics.
