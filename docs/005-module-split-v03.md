# v03 Module Split

## CLI

Command implementations were extracted from `cli.py` into focused modules under
`src/ter_calculator/commands/`. The root module now contains parser construction,
entry-point dispatch, top-level error handling, and thin compatibility facades.

## Acceleration

The former monolithic `acceleration.py` was converted into a package:

- `hashing.py`
- `cache.py`
- `quick_analyser.py`
- `session_watcher.py`
- `parallel.py`

`ter_calculator.acceleration` continues to export the historical public API.
