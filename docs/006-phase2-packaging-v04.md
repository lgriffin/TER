# TER v04 — Phase 2 Packaging and Reliability

TER v04 completes the remaining low-risk Phase 2 packaging work on top of the v03 module split.

## Optional integrations

The default installation now contains only the core runtime dependencies. Semantic embeddings are installed with the `embeddings` extra, and Anthropic-assisted intent extraction is available through the `llm` extra.

```bash
python -m pip install -e .
python -m pip install -e ".[embeddings]"
python -m pip install -e ".[llm]"
```

Development and full test environments should use:

```bash
python -m pip install -c constraints/dev.txt -e ".[dev,embeddings]"
```

## Reproducibility

`constraints/dev.txt` and `constraints/ci.txt` define the tested dependency ranges. They are constraints rather than frozen lock files so the package can still be tested against supported dependency updates.

## Python support

The declared and tested Python range is 3.11 through 3.13. CI uses a three-version matrix and no longer implicitly promises compatibility with untested future Python releases.

## Coverage enforcement

Coverage configuration enables branch coverage and enforces a 90% project floor, below the previously verified 92% baseline. CI publishes `coverage.xml` from Python 3.11.

## Exception audit

Broad exception catches remain at deliberate process and isolation boundaries: CLI dispatch, plugin isolation, watcher loops, optional telemetry, and multiprocessing fallback. Internal cache deserialization and token-estimation paths now catch narrower expected exception families.
