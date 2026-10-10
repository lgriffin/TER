# Changelog

All notable public changes to TER are documented in this file. The project follows Semantic Versioning.

## [Unreleased]

### Added

- TER 4 foundation (maturity level L0): a new `ter` package laid out as a hexagon of domain, ports, application, adapters and bootstrap, with import-linter contracts enforcing inward-only dependencies.
- Provider-neutral `ter.event/0.1` event model with stable event ids, provenance, tool kinds and coverage of unmapped records, plus a Claude Code JSONL adapter.
- Deterministic offline tokenizer and embedder adapters.
- Golden snapshots that freeze TER 3 analysis on a six-session corpus, port contract tests, and architecture tests.
- TER scoring (phase scores, weighted aggregate, raw ratio, aligned and waste accounting) in the pure domain, `ter.domain.scoring`, with property-style tests.
- A `PriceBook` port, a domain `Rates`/`PriceSchedule` model with cost arithmetic, a dated price book shipped as package data (`ter/data/price_book.json`), a JSON adapter, an in-memory fake, and a contract suite for both.
- EARS requirements control: a YAML requirement catalogue per maturity level under `requirements/`, an EARS grammar linter (six templates, one "shall", banned vague words, optional controlled vocabulary), a `--req-trace` pytest option recording which tests verify which requirement, and the `ter-req` command (`lint`, `trace`, `points`, `report`) with a CI gate that fails when a verified L0 requirement lacks a passing test or a test cites an unknown id.
- Vision point navigation: `requirements/points.yaml` gives each of the 200 TER 4 vision points a definition of done, enforcing EARS rules and verification; `ter-req lint` checks the links both ways and that done points are proven; `ter-req points` generates `docs/ter4/points.md`, checked for staleness in CI. Points that need real session data link their GitHub issue and cannot be marked done until `real_data_verified` is set. The catalogue grows to 116 requirements covering every point, reusing the requirement ids proposed in issues #34 to #46.
- TER 4 maturity level L1 (Observed): an incremental, idempotent `AnalysisEngine` whose batch analysis is the fold of its live analysis, with a `StreamReport` of event, tool and token counts, usage, duplicate tool calls, repeated reads, orphan results, open requests, edits without validation and a timeline.
- `EventIngest` driving port, `EventLog` driven port with JSONL and in-memory adapters, and `ObserveEvent` / `RecordEvent` / `AnalyseTrace` / `AnalyseEventLog` use cases. The hook path only appends, to a private (0700/0600) log under the user cache directory.
- Claude Code hooks driving adapter that fails open, and `python -m ter observe` / `python -m ter hook` commands. See `docs/ter4/l1-observed.md`.
- Equivalence, hypothesis property, hook payload contract and StreamReport golden tests; `hypothesis` joins the dev extra.
- TER 4 visual report layer: a `SessionReport` view-model in `ter.domain.report`, and a driving-side reports adapter (`ter.adapters.driving.reports`) with accessible SVG chart primitives (stat tiles, stacked and horizontal bars, span timeline, waste Pareto, positional sparkline) and one colour palette for light and dark themes.
- `ter report --html FILE` writes a self-contained HTML report (no scripts or network requests) with KPIs, all charts, a waste-pattern table, an uncertainty note and a how-to-read guide; see `docs/ter4/reports.md`.
- Golden snapshots of the report view-model, HTML and SVG output, and a `report-renderers` import contract.
- TER 4 maturity level L2 (Explained): a pure Lean model in `ter.domain.lean` (activity classes, eight Lean wastes, a six-stage agentic value stream, flow states), eleven plugin waste detectors behind a `WasteDetector` protocol and registry, each finding citing its evidence event ids with a published confidence rule and uncertain findings kept apart, productive iteration distinguished from rework, a typed session evidence graph (`ter.evidence/0.1`), and a multidimensional scorecard with agentic flow efficiency. See `docs/ter4/l2-explained.md` and ADR 0004.
- `ter a3 SESSION --html FILE --json [FILE] --graph FILE` (also `python -m ter a3`) writes a one-page Toyota A3 report: background, value stream map, waste Pareto, activity and flow bars, root causes with evidence, countermeasures (CLAUDE.md lines, Claude Code hooks, settings) and follow-up measures. `ter explain` prints the findings.
- `AnalysisEngine.explain()` and `explain_batch`: the L2 explanation of the live fold equals the batch one. `TerScorer` driven port with a TER 3 adapter (offline, pinned mode by default in the A3) and a fixed fake.
- Two synthetic golden sessions (`lean_mix`, `iteration_converges`) and golden snapshots of findings, scorecard, evidence graph, A3 JSON and A3 HTML for the whole corpus.
- Requirements catalogue: the L2 tests cite catalogue ids (the provisional TER-LEAN and TER-A3 ids map onto TER-LEN, TER-DET, TER-SCR, TER-FLW, TER-GRF, TER-ANL and new TER-RPT-003 to 005, TER-LEN-008 and TER-ANL-012); CI gates traceability at L0, L1 and L2. The Lean model ADR is numbered 0004.

### Changed

- `ter_calculator.compute.compute_ter` delegates to `ter.domain.scoring`, and TER 3's model rates (`CostModel` defaults, `--cost-model sonnet`, cost-weighted pricing tiers) are read from the price book. Scores and costs are unchanged on the golden corpus.
- CI pins ruff below 0.16, whose wider default rule set fails the existing code base.
- PyYAML is now a runtime dependency (catalogue loading).
- The Claude Code tool map moved to `ter.adapters.claude_code_tools`, shared by the JSONL source and the hooks adapter; `ter.adapters.driven.claude_code.tool_map` re-exports it.
- `ter.ports.driven` imports numpy for type checking only, so hook processes start faster.
- `ter visualize` and `ter present` charts now draw through the TER 4 report primitives: each SVG gains a `<title>` and `<desc>` (via `aria-labelledby`), legends wrap, and in-bar labels use a contrast-checked text colour. Chart names and the `ter_calculator.charts` API are unchanged.
- The A3 page leads with *At a glance*: the problem, four headline measures with bars and the three countermeasures to act on first. Countermeasures are numbered in the order to act on them; root causes are cards that link to the action answering them and back. With `--repo` a Repository evidence section shows reads later used, unused reads with their context tokens, files explored against changed and outcome value (TER-RPT-006, TER-RPT-007). A sticky section bar, header chips (maturity, cost, verdict), labelled snippet targets and a phone layout with no sideways scroll. The A3 JSON is unchanged.

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
