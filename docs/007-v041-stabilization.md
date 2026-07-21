# TER v0.4.1 Stabilization

## Scope

This release stabilizes v0.4.0 after the Phase 2 packaging and module-split work.

## Static analysis

- Ruff passes for `src` and `tests`.
- Production source remains strict.
- Tests use narrowly scoped per-file ignores for compact legacy fixture style:
  `E401`, `E701`, `E702`, `F401`, `F403`, `F405`, and `F841`.
- Mypy passes across all 52 source files.
- Optional third-party modules (`anthropic`, `sentence_transformers`, and
  `tiktoken`) are explicitly treated as optional imports by mypy.

## Corrections

- Removed stale imports introduced during the acceleration package split.
- Restored the historical `multiprocessing` and `time` compatibility exports.
- Added explicit PluginRegistry attribute types.
- Corrected formatter loop-variable type inference.
- Added typed session-listing records.
- Corrected optional API, embedding-cache, dashboard, intent, and real-time
  return types.
- Added Rich type-only imports.
- Preserved runtime behavior while replacing implicit `Any` returns with
  validated or converted values.

## Verification

The canonical Python 3.11 environment reported 1,010 tests passing for v0.4.0.
The v0.4.1 build environment verified:

- `ruff check src tests`: passed
- `mypy src`: passed, 52 files checked
- Non-network regression subset: 772 passed, 37 deselected

A complete embedding-enabled coverage run requires access to the configured
sentence-transformers model. The canonical command remains:

```bash
python -m pip install -c constraints/dev.txt -e ".[dev,embeddings]"
python -m pytest --cov=ter_calculator --cov-branch --cov-report=term-missing
```
