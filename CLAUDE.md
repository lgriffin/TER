# TER Development Guidelines

Last updated: 2026-05-15

## Active Technologies

- Python 3.11+ + sentence-transformers (embeddings), numpy (similarity computation), rich (terminal formatting), sqlite3 (fragment storage)

## Project Structure

```text
src/ter/               # TER 4 hexagon: domain/ ports/ application/ adapters/ bootstrap/
src/ter_calculator/    # TER 3 modules (wrapped by ter adapters during the rebuild)
tests/unit/            # Unit tests
tests/features/        # BDD feature files and step definitions
tests/integration/     # Integration tests
tests/golden/          # Golden snapshots freezing TER 3 scores (TER_UPDATE_GOLDEN=1 to regenerate)
tests/contract/        # One suite per port; real adapters and fakes must both pass
tests/architecture/    # Import-contract fitness tests
tests/requirements/    # EARS catalogue lint, trace and ter-req tooling tests
tests/docs/            # Doc checks: relative links, command examples, points index
requirements/          # EARS requirement catalogue (YAML per maturity level) + points.yaml (200 vision points)
tests/equivalence/     # Live (incremental) analysis and explanation == batch on the golden corpus
tests/fixtures/hooks/  # Example Claude Code hook payloads pinned by contract tests
docs/                  # Architecture, user guide, context orchestrator reference
docs/guides/           # Practical guides: testing, lean, hooks, ears, definition of done, A3, architecture, contributing
sample_sessions/       # Sample JSONL files for testing
```

## Commands

```bash
python dev.py check                       # Everything CI checks (lint + L0-L3 gates); `python dev.py` lists tasks
python dev.py fast                        # Quick loop: -x, last failures first, no embedding model
pytest                                    # Run all tests
pytest tests/unit/test_fragment_store.py  # Run specific module tests
ruff check src/                           # Lint
lint-imports                              # Hexagon dependency rules
pytest tests/golden tests/contract tests/architecture  # L0 gate
ter-req lint --tests tests                # EARS grammar + every req marker cites a known id
pytest --req-trace=req-trace.json && ter-req trace --results req-trace.json --gate L0  # traceability gate
ter-req report --results req-trace.json   # Markdown coverage per level
ter-req points                            # regenerate docs/ter4/points.md (CI runs --check)
pytest tests/equivalence tests/contract                # L1 gate (plus tests/unit/test_ter4_*)
pytest tests/unit/test_ter4_lean_* tests/unit/test_ter4_a3.py tests/golden/test_lean_snapshots.py  # L2 gate
python -m ter observe <session.jsonl> --timeline      # TER 4 L1 report
ter a3 <session.jsonl> --html a3.html --json a3.json   # TER 4 L2 Lean A3 (also python -m ter a3)
python -m ter explain <session.jsonl>                 # L2 findings as text
```

## Code Style

Python 3.11+: Follow standard conventions. Dataclasses for models, enums for domain constants, lazy imports in CLI handlers.

## TER 4 rules

- Dependencies point inward: bootstrap → adapters → application → ports → domain. `ter.domain` never imports `ter_calculator`, vendor SDKs or IO modules.
- TER scoring arithmetic lives in `ter.domain.scoring`; `ter_calculator.compute` delegates to it.
- Model prices are data in `src/ter/data/price_book.json`, read through the `PriceBook` port (ADR 0003). Never hard-code rates.
- Scoring changes must show up as a golden snapshot diff, committed on purpose.
- Tag tests with `@pytest.mark.req("TER-XXX-NNN")` for the requirement they verify. Every behaviour is an EARS requirement in `requirements/*.yaml` (see `docs/ter4/requirements.md`); new ones start `status: planned` and become `verified` once a passing test cites them.
- Every TER 4 change names the vision point ids it advances (`P044`, …) in its commit message and PR body, and updates those points' `status` and `verification` in `requirements/points.yaml` in the same PR (then `ter-req points`). A point is `done` only when its rules are verified by tests. Index: `docs/ter4/points.md`.
- New `ter` code is strictly typed (mypy overrides in pyproject.toml).
- Analysis is a fold over `ter.event`: `AnalysisEngine.apply` must stay O(1) amortised and idempotent by event id, and batch must equal incremental (`docs/ter4/l1-observed.md`).
- Hook entry points fail open: never raise out of `ter.adapters.driving.claude_hooks`.
- Run TER 4 tools with `PYTHONPATH=src` when the venv's editable install may point at another checkout.
- Lean model (L2): detectors are plugins in `ter.domain.lean.detectors` (register in `DEFAULT_REGISTRY`, add a countermeasure and a follow-up measure, add unit tests with positive, negative and boundary cases). Every finding cites evidence event ids and a published `confidence_rule`; below 0.70 it is uncertain and never counted as avoidable. Thresholds are structural, never token counts; iteration that converges is never rework. See `docs/ter4/l2-explained.md` and ADR 0004.
- See `docs/ter4/architecture.md` and `docs/decisions/`.
- Visual reports: renderers in `ter.adapters.driving.reports` (`svg.py`, `html.py`, colours only in `palette.py`) read the `ter.domain.report.SessionReport` view-model; `reports/ter3.py` (`from_ter_result`) is the only piece that reads `TERResult`. `ter_calculator.charts` delegates to these primitives. Rendered output is frozen in `tests/golden/snapshots/report/`. See `docs/ter4/reports.md`.

## Key Modules

### Core Pipeline
`models.py` `loader.py` `intent.py` `classifier.py` `compute.py` `waste.py` `economics.py` `formatter.py` `cli.py` `analyze_pipeline.py`

### Context Orchestrator
`fragment_store.py` `context_graph.py` `budget_optimizer.py` `delta_composer.py` `consistency.py`

### Real-Time & Adaptive
`real_time.py` `adaptive_budget.py` `cost_model.py` `overthinking.py`

### TER 4 Lean (L2, `src/ter/domain/lean/`)
`model.py` `facts.py` `steps.py` `detectors.py` `graph.py` `analysis.py` `countermeasures.py` `a3.py`; renderer `src/ter/adapters/driving/reports/a3.py`; use case `src/ter/application/explain.py`

## CLI Subcommands

`ter analyze` `ter report [--html FILE]` `ter a3` `ter explain` `ter visualize` `ter present` `ter compare` `ter list` `ter watch` `ter budget` `ter context {store|graph|optimize|delta|check}`

TER 4 (`python -m ter`): `observe` `hook` `explain` `a3` `route` `context {bundle|report}` `corpus`
