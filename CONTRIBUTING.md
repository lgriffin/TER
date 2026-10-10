# Contributing to TER

TER is being rebuilt as TER 4: a Lean analysis platform for agentic software
engineering, grown inside a hexagon around the TER 3 calculator. This page is
the short version of how to contribute. The full workflow, with examples, is
the [contributing guide](docs/guides/contributing.md); the other
[guides](docs/guides/README.md) cover testing, EARS requirements, the
definition of done, Lean, hooks and the A3.

## Setup

```bash
git clone https://github.com/lgriffin/TER.git
cd TER
python -m venv .venv && source .venv/bin/activate
python dev.py setup      # installs .[dev] and the pre-commit hooks
```

On Windows, activate with `.venv\Scripts\activate`. Every other command on
this page is the same on every platform.

Python 3.11, 3.12 and 3.13 are supported. With `.[dev]` alone the suite is
green: the 39 TER 3 tests marked `embeddings` are skipped, each with the
install command as its reason. Add `embeddings` to the extras to run them; they
then need network access to the Hugging Face host for the model. Everything
else, including every golden test, runs offline. CI installs the extra and sets
`TER_REQUIRE_EMBEDDINGS=1`, so there they always run.

## The hexagon and its dependency rule

New behaviour goes in `src/ter`, in one of five layers. Dependencies point
inward only:

```text
ter.bootstrap → ter.adapters → ter.application → ter.ports → ter.domain
```

- `ter.domain` is pure: no TER 3 (`ter_calculator`), no vendor SDK, no IO.
- `ter.ports` and `ter.application` know no vendors.
- Driven adapters do not import each other; only `ter.bootstrap` picks
  concrete adapters.
- TER 3 changes only to delegate inward (the strangler plan in
  [ADR 0001](docs/decisions/0001-hexagonal-strangler-rebuild.md)).

`lint-imports` enforces these contracts from `pyproject.toml`, and
`tests/architecture` checks the same rules. See the
[architecture guide](docs/guides/architecture.md).

## Capability packs

Work from another project (GARE is the first) enters TER as a **capability
pack**, never as a dependency or vendored code
([ADR 0005](docs/decisions/0005-admitting-external-capabilities.md)):

- a **port** in `ter.ports` with its obligations in the docstring;
- a **contract suite** in `tests/contract/` that every adapter and the fake
  pass;
- **EARS requirements** in `requirements/`, each linked to a vision point;
- an **in-memory fake**;
- one **reference adapter**, registered through an entry point and shipped as
  an optional extra. It reads the outside system's files by schema name (for
  example `gare.ter.usage.v2`) and never imports its package.

The domain, ports and use cases may not import `gare`, `pydantic` or `httpx`
(TER-ARC-003). The
[contributing guide](docs/guides/contributing.md#adding-a-capability-pack)
walks through adding one.

## Requirements and vision points

- **Every behaviour is an EARS requirement** in `requirements/*.yaml`, one
  file per maturity level, written in one of six templates. A new one starts
  `status: planned` and becomes `verified` once a passing test cites it with
  `@pytest.mark.req("TER-XXX-NNN")`. See the [EARS guide](docs/guides/ears.md).
- **Every requirement serves a vision point** in `requirements/points.yaml`,
  linked both ways (`source_points` on the requirement, `rules` on the
  point). P001 to P200 are Leigh's vision, verbatim. A point contributed from
  elsewhere is numbered past P200 and records its `origin` (`source`, `ref`,
  `author`). A point is `done` only when its rules are verified by tests, and
  never on synthetic data alone when it needs real sessions. See the
  [definition of done](docs/guides/definition-of-done.md); the living index
  is [docs/ter4/points.md](docs/ter4/points.md).

```bash
ter-req lint --strict --tests tests       # EARS grammar, points, links, test markers
ter-req points                            # regenerate docs/ter4/points.md
ter-req report --results req-trace.json   # coverage per maturity level
```

## Gates to run before pushing

`dev.py` runs every check CI runs, with the tools from your active
environment and `src` first on the path, so it tests this checkout:

```bash
python dev.py            # list the tasks
python dev.py fast       # quick loop: stop at first failure, no model
python dev.py fmt        # format and auto-fix lint
python dev.py check      # the CI lint and test jobs: lint, coverage, L0-L3 gates
```

A test (`tests/docs/test_dev_tasks.py`) keeps `dev.py` and CI in step: every
command `python dev.py check` runs is also a CI step.

Two rules the gates rely on you to keep:

- **Golden snapshot diffs are on purpose.** A scoring change shows up as a
  diff in `tests/golden`, regenerated with `TER_UPDATE_GOLDEN=1` and
  explained in the commit. See the
  [testing guide](docs/guides/testing.md#changing-a-snapshot-on-purpose).
- **`ter` code is strictly typed.** `mypy src/` applies the strict overrides
  in `pyproject.toml` to every `ter` module.

Model prices are data in `src/ter/data/price_book.json`
([ADR 0003](docs/decisions/0003-price-book-as-data.md)); never hard-code a
rate.

## Commits and pull requests

A TER 4 commit message says what changed and why, then names:

- the **vision point ids** it advances (`P044`, …) and how their status
  changed, with their `status` and `verification` updated in
  `requirements/points.yaml` in the same change;
- the **requirement ids** it adds, verifies or changes;
- any **golden snapshot diff** and why the numbers moved, or that snapshots
  are unchanged.

The PR body repeats the point and requirement ids and lists the gates you
ran. Keep one feature or fix per PR. Branch names: `feature/…`, `fix/…`,
`docs/…`, `refactor/…`, `test/…`.

## Reporting bugs and requesting features

Open a [GitHub issue](https://github.com/lgriffin/TER/issues) with steps to
reproduce, expected and actual behaviour, Python version and OS, and a
redacted session file if one is involved. Work that needs real session data
belongs under the tracker issue #47.

## License

By contributing you agree your contributions are licensed under the
[Apache License 2.0](LICENSE).
