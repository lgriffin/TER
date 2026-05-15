# TER Development Guidelines

Last updated: 2026-05-15

## Active Technologies

- Python 3.11+ + sentence-transformers (embeddings), numpy (similarity computation), rich (terminal formatting), sqlite3 (fragment storage)

## Project Structure

```text
src/ter_calculator/    # All source modules
tests/unit/            # Unit tests
tests/features/        # BDD feature files and step definitions
tests/integration/     # Integration tests
docs/                  # Architecture, user guide, context orchestrator reference
sample_sessions/       # Sample JSONL files for testing
```

## Commands

```bash
pytest                                    # Run all tests
pytest tests/unit/test_fragment_store.py  # Run specific module tests
ruff check src/                           # Lint
```

## Code Style

Python 3.11+: Follow standard conventions. Dataclasses for models, enums for domain constants, lazy imports in CLI handlers.

## Key Modules

### Core Pipeline
`models.py` `loader.py` `intent.py` `classifier.py` `compute.py` `waste.py` `economics.py` `formatter.py` `cli.py` `analyze_pipeline.py`

### Context Orchestrator
`fragment_store.py` `context_graph.py` `budget_optimizer.py` `delta_composer.py` `consistency.py`

### Real-Time & Adaptive
`real_time.py` `adaptive_budget.py` `cost_model.py` `overthinking.py`

## CLI Subcommands

`ter analyze` `ter report` `ter compare` `ter list` `ter watch` `ter budget` `ter context {store|graph|optimize|delta|check}`
