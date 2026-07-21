# Migrating to TER 3.0

TER 3.0 promotes the closed-loop capabilities developed throughout the v2 series into the stable product surface. The command name remains `ter`, and normal analysis/reporting workflows remain compatible.

## Upgrade

```bash
python -m pip install -U -e ".[dev,embeddings]"
ter --version
```

Expected version:

```text
3.0.0
```

## Required repository-memory refresh

TER 3.0 marks repository-memory indexes with schema version `3.0.0`. Rebuild the local index after upgrading:

```bash
ter memory index --root .
```

Existing lesson, outcome, history, and tuned-policy files under `.ter/` remain repository-local. Keep a backup before upgrading when those files contain production evidence.

## Hook configuration

Threshold precedence is explicit in TER 3.0:

1. an explicitly supplied hook CLI flag;
2. a value loaded from `.ter/tuned-policy-config.json`;
3. the built-in `PolicyConfig` default.

Omitted threshold arguments use an internal `None` sentinel, so explicitly supplying a value equal to the built-in default is still respected.

## Recommended release verification

```bash
ruff format --check src tests
ruff check src tests
mypy src/
python -m pytest
python -m pytest --cov=ter_calculator --cov-branch --cov-report=term-missing
```

The validated v3.0 release baseline is 1,162 passing tests with branch coverage enforced at 90% or higher.

## New stable capabilities

- project-scoped repository memory and semantic duplicate detection;
- live TER/waste degradation policies with context refresh and replanning;
- durable intervention outcomes and recurring-scenario analysis;
- estimated cost saved and wasted by intervention type;
- dry-run-by-default, opt-in repository threshold tuning;
- static effectiveness dashboard with inline SVG charts and tuning visibility;
- minimal-install and full-extras CI coverage.
