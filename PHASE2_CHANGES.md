# Phase 2 Implementation — TER v04

TER v04 builds on the v03 command and acceleration module split and completes the remaining low-risk Phase 2 packaging and reliability work.

## Implemented

- Lightweight default installation without `sentence-transformers` or PyTorch.
- `embeddings` optional extra for semantic analysis.
- `llm` optional extra for Anthropic-assisted intent extraction.
- Tested dependency ranges in `constraints/dev.txt` and `constraints/ci.txt`.
- Explicit Python support range: 3.11 through 3.13.
- GitHub Actions test matrix for Python 3.11, 3.12, and 3.13.
- Branch-coverage collection, XML artifact publishing, and a 90% regression floor.
- Narrower expected-exception handling for tiktoken fallback and cache deserialization.
- Documentation updates for installation, development, and validation.
- Package version bump to 0.4.0.

## Deliberately retained broad boundaries

Broad catches remain where isolation or graceful degradation is intentional:

- Top-level CLI command boundaries
- Plugin execution and loading boundaries
- Filesystem watcher loops
- Multiprocessing fallback
- Optional Anthropic API and telemetry paths

These boundaries log, report, or degrade gracefully rather than silently masking errors.

## Validation performed in the build environment

```text
Python compilation: passed
Focused packaging/compatibility tests: 36 passed
Non-ML unit suite: 773 passed
Editable core install without dependencies: passed
Package metadata version: 0.4.0
CLI parser smoke test: passed
```

The embedding-dependent and BDD suites require the complete development environment:

```bash
python -m pip install -c constraints/dev.txt -e ".[dev,embeddings]"
python -m pytest --cov=ter_calculator --cov-branch --cov-report=term-missing
```
