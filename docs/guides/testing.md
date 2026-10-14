# Testing guide

TER's tests do more than catch regressions. They freeze TER 3's behaviour
while it is rebuilt, hold every adapter to its port's contract, keep the
architecture's dependency rule, and prove each EARS requirement. This guide
walks through each layer, the gates CI runs, and how to write a new test from
requirement to passing gate.

## The pyramid in this repository

```mermaid
flowchart TB
    G["Gates<br/>ter-req trace --gate L0/L1/L2 · coverage ≥ 90% · lint-imports"]
    A["Architecture fitness<br/>tests/architecture"]
    GO["Golden snapshots<br/>tests/golden"]
    EQ["Equivalence<br/>tests/equivalence"]
    C["Contract suites<br/>tests/contract"]
    S["Schema contracts<br/>tests/schema"]
    SM["Smoke<br/>tests/smoke, from the wheel"]
    I["Integration and BDD<br/>tests/integration · tests/features"]
    P["Property tests<br/>hypothesis, in tests/unit"]
    U["Unit tests<br/>tests/unit · tests/requirements · tests/docs"]
    G --- SM --- A --- GO --- EQ --- C --- S --- I --- P --- U
```

| Layer | Where | Roughly how many | What it proves |
|---|---|---:|---|
| Unit | `tests/unit/` | 1,400 | One module's behaviour: TER 3 modules (`test_classifier.py`, `test_waste.py`, …) and TER 4 (`test_ter4_*.py`) |
| Property | `tests/unit/test_ter4_lean_properties.py`, `test_ter4_stream_properties.py` | (in unit) | Invariants over generated sessions (hypothesis) |
| BDD | `tests/features/**/*.feature` with steps in `tests/features/steps/*_steps.py` | 200 | TER 3 user stories in Gherkin (pytest-bdd) |
| Integration | `tests/integration/test_cli.py` | 12 | The `ter` CLI end to end on real files |
| Contract | `tests/contract/` | 100 | Every adapter of a port, real and fake, meets the same obligations |
| Schema | `tests/schema/` | 50 | Every versioned document format has a JSON Schema; writers conform, readers accept what it allows |
| Smoke | `tests/smoke/` | 90 | Every `ter` and `python -m ter` subcommand runs end to end in a fresh process; bad input is refused without a traceback; CI runs it against the built wheel |
| Equivalence | `tests/equivalence/test_live_static.py` | 24 | Live, event-by-event analysis equals batch analysis |
| Golden | `tests/golden/` | 110 | TER 3 scores, event streams, reports, Lean findings and A3s are frozen |
| Architecture | `tests/architecture/` | 2 | The import-linter contracts hold |
| Requirements tooling | `tests/requirements/` | 140 | The EARS lint, trace gate, points lint and `ter-req` CLI |
| Docs | `tests/docs/` | 40+ | README and docs links resolve, shown commands exist, the points index is fresh |

## Running tests

```bash
python -m pytest                                   # everything
python -m pytest tests/unit/test_ter4_lean_detectors.py -v
python -m pytest tests/golden tests/contract tests/architecture -q   # the L0 gate
python -m pytest -k "rework" -q                    # by name
python -m pytest tests/features/steps/core_steps.py -q               # one BDD module
```

`python -m pytest` uses the interpreter you installed into, which avoids
picking up a `pytest` from another environment. `pyproject.toml` sets
`pythonpath = ["src"]` and `--strict-markers`, so a misspelt marker fails.

### Tests that need the network

Two TER 3 dependencies download data on first use: tiktoken's `cl100k_base`
encoding and the sentence-transformers embedding model.

Tests that run the real embedding model carry the `embeddings` marker (in a
`.feature` file, the `@embeddings` tag): today 39 tests in
`tests/unit/test_input_analysis.py`, `tests/integration/test_cli.py` and the
intent scenarios under `tests/features`. Without the `embeddings` extra they
are skipped, each with the install command as its reason (`pytest -rs` lists
them). With the extra installed they run, and need the Hugging Face host for
the model. CI sets `TER_REQUIRE_EMBEDDINGS=1`, which turns the skip off, so a
missing model fails CI instead of hiding behind a skip. Mark any new test that
needs the real model the same way.

tiktoken falls back to a character estimate when it cannot download its
encoding, so those tests pass offline, slowly. Everything TER 4 does,
including every golden test, runs offline because it pins the deterministic
`RegexTokenizer` and `HashingEmbedder` adapters.

## Unit tests

Unit tests are plain pytest functions or classes, one file per module. TER 4
unit tests build event streams in code rather than reading files. The L2
detector tests use a small builder, `tests/unit/ter4_lean_builder.py`:

```python
@pytest.mark.req("TER-DET-002", "TER-DET-010")
class TestRepeatedToolCall:
    def test_identical_call_and_output_is_waste(self) -> None:
        s = Script()
        s.prompt("list things")
        s.bash("ls src", "a.py")
        _, first = s.bash("ls src", "a.py")
        [f] = found(s, "repeated_tool_call")
        assert f.confidence == 0.9 and not f.uncertain
        assert f.waste is LeanWaste.OVER_PROCESSING and f.kind is FindingKind.WASTE
```

Each detector has three kinds of cases: a positive one, a negative one, and
the boundary that separates them (for `rework_cycle`, a failure that *moves*
after a fix is iteration, not rework). Write all three for any new detector.

## Property tests

Where an invariant must hold for every session, not just the examples you
thought of, use hypothesis. `test_ter4_lean_properties.py` generates random
scripts of prompts, reads, edits, shell runs and handoffs, then checks that
every finding cites events that exist, confidence is bounded, uncertain
findings are exactly those below 0.70, and token sums add up:

```python
@settings(max_examples=150, deadline=None)
@given(scripts())
def test_findings_are_traceable_and_bounded(script: Script) -> None:
    a = explain(script.events, RegexTokenizer())
    ids = {e.id for e in script.events}
    for f in a.findings:
        assert f.evidence and set(f.evidence) <= ids
        assert f.uncertain == (f.confidence < UNCERTAIN_BELOW)
```

Keep `deadline=None` for anything that runs the full explainer, and keep
`max_examples` modest so the suite stays quick.

## BDD features

TER 3's user stories are Gherkin files under `tests/features/<area>/`, bound
to step definitions in `tests/features/steps/*_steps.py` with
`scenarios(...)`. `pyproject.toml` collects `*_steps.py` as test modules.

```gherkin
Scenario: Mixed alignment session produces valid TER with phase breakdown
  Given a completed session with the following spans:
    | phase      | tokens | aligned |
    | reasoning  | 200    | 160     |
  When the TER is calculated
```

Use BDD when a behaviour is best read by a non-developer as a scenario. New
TER 4 behaviour is usually better served by a unit or property test tagged
with its requirement.

## Golden snapshots

Golden tests freeze what TER computes on a fixed corpus: the sample session,
the fixture session and six synthetic sessions under `tests/golden/sessions/`
(see [tests/golden/README.md](../../tests/golden/README.md) for what each
exercises). For each session they snapshot:

| Snapshot | Content |
|---|---|
| `<name>.default.json`, `<name>.fine.json` | TER 3 analysis with default and fine span segmentation |
| `<name>.events.json` | The normalised `ter.event` stream |
| `<name>.stream.json` | The L1 `StreamReport` |
| `<name>.report.json`, `report/<name>.html` | The report view-model and HTML |
| `<name>.lean.json`, `<name>.a3.json`, `report/<name>.a3.html` | L2 findings, scorecard, evidence graph, the A3 view-model and page |

The tests pin TER 3's tokenizer and embedder to deterministic offline
adapters with the `pinned_models` fixture
([ADR 0002](../decisions/0002-hermetic-golden-characterisation.md)), so a
diff can only mean analysis logic changed.

### Changing a snapshot on purpose

A golden diff is a behaviour change. When you mean it:

```bash
TER_UPDATE_GOLDEN=1 python -m pytest tests/golden -q
git diff --stat tests/golden/snapshots
git diff tests/golden/snapshots/lean_mix.lean.json
python -m pytest tests/golden -q
```

Then review every changed number before committing:

- Does each change follow from your code change? A detector tweak should
  move that detector's findings and the scorecard, not TER 3 scores.
- Did anything change that should not have? A `*.default.json` diff from a
  report change means you touched analysis, not presentation.
- Commit the snapshot diff with the code that caused it, and say in the
  commit message which numbers moved and why. Reviewers read the diff.

To add a session, drop the `.jsonl` in `tests/golden/sessions/`, add its name
to `SYNTHETIC_SESSIONS` in `tests/golden/corpus.py`, and regenerate.
`test_corpus_integrity.py` fails on a missing snapshot or an orphaned one.

## Contract suites

A port's obligations are one test module under `tests/contract/`, run against
every adapter, real and fake, through a parametrised fixture. This keeps
fakes honest: a fake that drifts from the real adapter fails the same suite.

```python
@pytest.fixture(params=[_json, _memory], ids=["json", "in-memory"])
def book(request: pytest.FixtureRequest) -> PriceBook:
    factory: Factory = request.param
    return factory()


def test_unknown_model_raises(book: PriceBook) -> None:
    with pytest.raises(UnknownModelError):
        book.rate("no-such-model")
```

| Suite | Port | Adapters |
|---|---|---|
| `test_session_source.py` | `SessionSource` | `ClaudeCodeJsonlSource`, `GareRunSource` (mock runs and a real recorded run), `InMemorySessionSource` |
| `test_price_book.py` | `PriceBook` | `JsonPriceBook`, `InMemoryPriceBook` |
| `test_event_log.py` | `EventLog` | `JsonlEventLog`, `InMemoryEventLog` |
| `test_ter_scorer.py` | `TerScorer` | `Ter3Scorer` (offline), `FixedTerScorer` |
| `test_event_ingest.py` | `EventIngest` | `ObserveEvent` (no log, in-memory, JSONL), `RecordEvent` (the hook's append-only ingest) |
| `test_ingest_wiring.py` | `EventIngest` wiring | every CLI path (transcript, GARE run, event log, explain, hook) applies its events through the composition root's `EventIngest` |
| `test_hook_payloads.py` | hook payload shapes | fixtures in `tests/fixtures/hooks/` |

When you add an adapter, add its factory to the fixture's `params`. When you
add an obligation to a port, write it in the port's docstring and as a test
here.

## Schema contracts

Every file TER writes and later reads (event logs, ledgers, policy files,
control limits and measures) names its format in a `schema` field, such as
`ter.intervention-policies/1`. The JSON Schema (draft-07) for each format
ships in the package under `src/ter/data/schemas/<name>/<version>.json`, so a
tool in any language can check a file before TER reads it. This is TER's
answer to schema libraries such as zod: the schema is published data, and
`tests/schema` holds the Python writers and readers to it from both sides.

| Test module | Requirement | What it proves |
|---|---|---|
| `test_catalogue.py` | TER-TST-001, TER-TST-004 | every `ter.*/N` id in `src/ter` ships a schema or is on the `PENDING` list (which only shrinks); each schema is valid and canonically formatted; every enumeration equals the domain enumeration it names in `x-ter-enum` |
| `test_writers.py` | TER-TST-002, TER-TST-005 | real output conforms: events of every golden and GARE session, the frozen control limits and measures, a draft and an approved policy file, a shadow-run ledger with an observation. Dropping any field or adding an unknown one is rejected |
| `test_readers.py` | TER-TST-003, TER-TST-002 | hypothesis-jsonschema generates documents from each schema; the reader reads them (events and ledger lines round-trip exactly) or refuses them with its own error for a rule the schema cannot state, and what it reads it writes back conforming |

```python
@given(documents("ter.event/0.5"))
def test_any_conforming_event_record_round_trips_exactly(record: Any) -> None:
    event = event_from_record(record)
    assert event_to_record(event) == record
```

An enumeration copies a domain one and says which:

```json
"outcome": {
  "type": "string",
  "x-ter-enum": "ter.domain.intervention:Outcome",
  "enum": ["advised", "below_ceiling", "no_evidence", "ineligible", "..."]
}
```

When you change a format:

- **Add a field or an enum value**: update the schema in the same change. The
  writer test fails until you do, because schemas forbid unknown fields.
  Readers of the old version notice, so bump the version (a new file beside
  the old one) when a reader would.
- **Add a new format**: ship `src/ter/data/schemas/<name>/<version>.json` and
  add its writer to `test_writers.py` and its reader to `test_readers.py`. A
  new `ter.<name>/<version>` id without a schema fails `test_catalogue.py`.
- **Ship a pending format's schema**: remove it from `PENDING` in
  `test_catalogue.py`; the test fails if you forget.

Load a schema from code with
`ter.adapters.driven.json_schemas.schema("ter.control-limits/1")`. TER itself
never validates at runtime, so `jsonschema` is a dev dependency only.

## Smoke tests

Unit tests call functions; people run commands. `tests/smoke` starts the real
console scripts (`ter`, `python -m ter`, `ter-req`) in fresh processes, in an
empty working directory, the way a developer or Claude Code does, and checks
exit codes, the files written and, where a format has a JSON Schema, that
they conform. Each journey chains commands as a user would: measure, limits
and chart; check, approve, shadow and effectiveness; hook payloads on stdin
into the event log, then `observe` and `hooks check`.

```bash
python dev.py smoke                                     # this checkout
TER_SMOKE_BIN=/tmp/ter-wheel-test/bin python -m pytest tests/smoke -q   # an install
```

With `TER_SMOKE_BIN` the suite runs that environment's scripts with no
`PYTHONPATH`; CI's package job points it at a clean venv holding only the
built wheel, so a file left out of the package fails there. Without it, the
scripts beside the current interpreter run with `src` first.

| Module | Requirement | What it proves |
|---|---|---|
| `test_install.py` | TER-TST-006 | the scripts start and report versions, the right package is the one running, every subcommand has `--help`, and every subcommand appears in `TER4_COMMANDS` or `TER3_COMMANDS` with the journey that covers it (a new subcommand fails until it has one) |
| `test_ter4_analysis.py` | TER-TST-006, TER-TST-007 | `explain`, `a3`, `route`, `context`, `corpus import`; missing and malformed sessions are refused with a reason |
| `test_ter4_control_and_policy.py` | TER-TST-006, TER-TST-007 | control charts and the L4 policy lifecycle; every measures, limits, policy and ledger file conforms to its schema; an edited policy file is refused until approved again |
| `test_ter4_hooks.py` | TER-TST-006, TER-TST-008 | hook payloads become schema-valid events; advice on Stop is only a `systemMessage`; a broken payload or an unwritable log still exits 0 with `{}` |
| `test_ter3.py` | TER-TST-006, TER-TST-007 | TER 3's `ter`: commands that need the embedding model work when the extra is installed and otherwise refuse with the install command |

The first runs of this suite found three bugs no unit test had: a file that
was not a session crashed `observe`, `explain`, `a3` and `route` with a
traceback, and `ter context store|graph|delta|check` and `ter watch` (with
the embeddings extra) recursed until Python gave up. The unit tests called
the handlers directly, never through the dispatcher.

## Resilience: fuzzing and fault injection

`tests/resilience` attacks every reader of input TER does not control, and
every file a dying process can leave half written.

The fuzzers (Hypothesis) start from real documents: corpus sessions, a GARE
run, the hook payload fixtures, the shipped policy file, the frozen control
limits and measures, an event log and a ledger written by a shadow run. They
remove, retype, stringify or replace a few values, tear and duplicate lines,
and mix in arbitrary JSON and bytes. A mutated real document gets past the
first checks and reaches the field that is wrong, which pure random JSON
rarely does. Each property names what a reader may do with bad input:

| Module | Requirement | Property |
|---|---|---|
| `test_fuzz_sources.py` | TER-TST-009 | a session source returns a well-formed trace (unique ids, contiguous sequences) or raises `SessionFormatError`, nothing else |
| `test_fuzz_hooks.py` | TER-TST-010 | a payload of the wrong shape is refused with a stated reason, never by an exception the fail-open entry point caught; the hook always prints a JSON object |
| `test_fuzz_documents.py` | TER-TST-011, TER-TST-013 | policy, limits and measures files are read or refused with their format's own error; a damaged event log or ledger is read line by line, and what was read can still be analysed |
| `test_faults.py` | TER-TST-013, TER-TST-014 | a write cut short loses only its own record and the next write starts a fresh line; a short `os.write` still writes the whole record; a full disk, a read-only log or an unreadable policy file leaves the hook answering `{}`, and advice the ledger could not record is not shown |
| `test_regressions.py` | TER-TST-009, TER-TST-013 | every input the fuzzers found, pinned as a plain test |

CI runs each fuzzer with a few hundred examples. Before changing a reader,
run them harder:

```bash
python dev.py fuzz                            # 20x the examples
python -m pytest tests/resilience --fuzz-scale=100 -q
```

A failing example is printed as a test case; pin it in
`test_regressions.py` with the fix, so it stays fixed without depending on
Hypothesis finding it again.

The first run found:
- a record repeated with its uuid gave duplicate event ids;
- records with a `null` or text `message`, a non-text role, block type or
  id, a text tool input or text token counts crashed the TER 3 loader;
- event log records with a list as call id or a text sequence decoded into
  events that crashed the analysis;
- a short write was never finished.

## Mutation score

Coverage says a line ran; it does not say a test would notice the line
changing. mutmut makes one small change at a time to `src/ter/domain` (a
`<` for a `<=`, a constant changed, an argument dropped) and runs the tests
that reach the changed function (golden, contract, equivalence and unit, as
`[tool.mutmut]` in `pyproject.toml` sets). A mutant a test fails on is
killed; one every test passes on survived, and is a behaviour no test pins.

`scripts/mutation.py` turns the run into a score per module: the share of
its mutants killed, timed out or caught by the type checker. A mutant no
test reaches counts as missed and a skipped one does not count.
`tests/mutation/baseline.json` holds each module's score, and the check
fails when a module scores lower, or has no baseline (TER-TST-015). Scores
only go up:

```bash
python dev.py mutation                        # the whole domain (about two hours)
python dev.py mutation --changed origin/main  # the modules this branch changes
python scripts/mutation.py ter.domain.scoring # one module or package
python scripts/mutation.py --shard 2/4        # one of the four parts CI runs
python scripts/mutation.py ter.domain.scoring --update  # raise its baseline
```

The `Mutation score` workflow runs the check on the domain modules a pull
request changes, and on the whole domain, in four shards, when main changes, every Monday and
on demand (TER-TST-016). When a change of yours fails it, look at the
survivors (`mutmut results`, then `mutmut show <name>`), add a test that
fails on them, and commit the raised baseline with `--update`. `--update`
refuses to lower a score, so a drop is fixed with tests, never by editing
the baseline. A survivor that cannot be killed (an equivalent mutant, such as
a changed log message) is marked with `# pragma: no mutate` on its line.

## Software bill of materials

`scripts/sbom.py` writes a CycloneDX 1.6 SBOM of an environment holding the
wheel and no pip of its own: each distribution's exact version, package URL,
declared licence and what depends on it. It then checks the SBOM against
that environment and exits 1 on any mismatch. CI's package job runs it on
the environment it smoke-tested, so the SBOM names the versions that were
tested, and publishes it as the `sbom` artifact of every run (TER-SBM-001,
TER-SBM-002). The check names:

- a distribution installed but not listed (direct or transitive), or listed
  but not installed, or listed at another version;
- build or test tooling (pip, setuptools, pytest);
- a component without a `pkg:pypi` package URL or a licence, or with a
  licence its own metadata does not declare. A distribution whose licence
  the SBOM cannot carry is on `LICENCE_NOT_IN_SBOM` with the reason, a list
  that only shrinks, so a new one is reviewed before it ships;
- a dependency outside its range in `pyproject.toml`, a graph entry that is
  not a component, or a component nothing TER installs depends on.

`tests/sbom` holds the check to each of those on small documents, with no
network. With `TER_SBOM` and `TER_SBOM_ENV` set (CI sets them), it also
checks the published SBOM against the tested environment.

`python dev.py sbom` writes the same file to `dist/ter.cdx.json` from this
checkout; it needs network access to install the dependencies and the SBOM
tool, which runs in an environment of its own. The SBOM is a build output,
not a committed file: the dependencies are ranges, so the versions it names
are the ones that resolved on the day it was built.

## Test order

Tests run in a random order (TER-TST-012): modules are shuffled, then the
classes in a module, then the tests in a class, so module and class
fixtures are still set up once. The run header names the seed:

```text
test order: random, seed 903883 (rerun with --test-order-seed=903883)
```

A test that passes in file order but fails shuffled leans on state another
test left behind. Rerun the failing order with `--test-order-seed=N` (or
`TER_TEST_ORDER_SEED=N`), and use `--test-order=declared` to compare with
file order. `--ff` and `--nf` keep their own order, so `python dev.py fast`
still runs the last failures first; the header then says the run is not
shuffled. The ordering lives in the root `conftest.py`, with no plugin to
install.

## Equivalence tests

`tests/equivalence/test_live_static.py` replays every corpus session one event
at a time (as hooks would deliver it, including every event twice) and
asserts the live report and the live Lean explanation equal the batch ones,
including at intermediate prefixes. This is TER-ANL-010: live and batch share
one semantics. Any change to `AnalysisEngine` or the Lean fold must keep it
green.

## Architecture fitness

`tests/architecture/test_import_contracts.py` runs import-linter against the
contracts in `pyproject.toml` and checks the expected contracts are declared.
Run `lint-imports` for a readable report of any broken chain. See the
[architecture guide](architecture.md#the-dependency-rule).

## Requirement tracing

Every TER 4 behaviour is an EARS requirement (see the [EARS guide](ears.md)).
Tests say which requirement they verify:

```python
@pytest.mark.req("TER-ANL-010", "TER-OBS-004")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_live_with_redelivery_through_the_log_equals_batch(name: str) -> None:
    ...
```

The root `conftest.py` registers the `ter.adapters.driving.pytest_req` plugin,
which adds `--req-trace`:

```bash
python -m pytest --req-trace=req-trace.json
ter-req trace --results req-trace.json --gate L0
ter-req trace --results req-trace.json --gate L2
ter-req report --results req-trace.json --out coverage.md
```

`ter-req trace` fails when a `verified` requirement at or below the gate has
no passing citing test (forward trace), or when a test cites an id that is
not in the catalogue (backward trace). It also lists `planned` requirements
that already have a passing test, ready to promote. `ter-req lint --tests
tests` catches unknown ids statically, without running the suite.

## Documentation tests

`tests/docs/test_docs.py` keeps these guides honest, offline and in under a
second:

- every relative link in `README.md` and `docs/**/*.md` names a file that
  exists and, for Markdown targets, a heading anchor that exists
  (TER-REQ-007);
- every `ter`, `ter-req` and `python -m ter` command in a fenced shell block
  in the README, the guides, `docs/ter4/` and the hooks guide names a real
  subcommand and real options, checked against the argparse `--help` of each
  subcommand; nothing in the example is run (TER-REQ-008);
- `docs/ter4/points.md` equals the index generated from
  `requirements/points.yaml` (TER-REQ-005).

If you document a command, write it in a `bash` block so the check sees it.
Use `text` for output.

## Coverage

Branch coverage is on and `pyproject.toml` sets `fail_under = 90` over both
packages:

```bash
python -m pytest --cov=ter_calculator --cov=ter --cov-branch --cov-report=term-missing
```

New `ter` code should be close to fully covered by the tests that verify its
requirements; defensive fallbacks and presentation paths may sit lower.

## What CI runs

`.github/workflows/ci.yml` has three jobs.

| Job | Step | Command |
|---|---|---|
| Lint and type-check | Ruff format | `ruff format --check src tests dev.py` |
| | Ruff lint | `ruff check src tests dev.py` |
| | Mypy | `mypy src/` |
| | Architecture import contracts | `lint-imports` |
| | EARS requirements and vision points lint | `ter-req lint --strict --tests tests` |
| | Vision points index is up to date | `ter-req points --check` |
| Python 3.11, 3.12, 3.13 | L0 gate | `python -m pytest tests/golden tests/contract tests/architecture -q` |
| | Tests with branch coverage | `python -m pytest --cov=ter_calculator --cov=ter --cov-branch --cov-report=term-missing --cov-report=xml --req-trace=req-trace.json` |
| | Traceability gates (3.11 only) | `ter-req trace --results req-trace.json --gate L0` (then L1, L2, L3) |
| Package | Build and smoke-test package | `python -m build`, `twine check`, install the wheel in a clean venv, `ter --help`, the schemas ship |
| | Smoke-test the installed wheel | `TER_SMOKE_BIN=/tmp/ter-wheel-test/bin python -m pytest tests/smoke -q` |
| | SBOM of the tested install | `python scripts/sbom.py --env /tmp/ter-wheel-test`, `TER_SBOM=dist/ter.cdx.json TER_SBOM_ENV=/tmp/ter-wheel-test python -m pytest tests/sbom -q`, upload as the `sbom` artifact |
| Mutation score | Mutation score ratchet | `python scripts/mutation.py --changed origin/main` on a pull request; the whole domain in four shards (`--shard K/4`) on main, weekly and on demand |

Run the same locally before pushing with `dev.py` (the
[contributing guide](contributing.md) lists every task):

```bash
python dev.py lint       # the lint job
python dev.py gates      # the suite with --req-trace, then the L0-L4 gates
python dev.py check      # both
python dev.py smoke      # the smoke suite, against this checkout
```

`tests/docs/test_dev_tasks.py` fails if `dev.py` runs a command CI does not.

`--strict` also fails on warnings (proof that exists only on an unmerged
branch), which CI will hit once that branch merges.

## Writing a new test end to end

Suppose you are adding a rule: an edit to a file the agent then deletes is
wasted. The full path from idea to gate:

1. **Find the vision point** it serves in
   [docs/ter4/points.md](../ter4/points.md) (search the text, or the level's
   section). Note its id, for example `P020`.
2. **Write the requirement** in the file for its level, `status: planned`,
   with `source_points` naming the point, and add the id to that point's
   `rules` in `requirements/points.yaml`:

   ```yaml
   - id: TER-DET-011
     pattern: event-driven
     text: >-
       When the agent deletes a file it edited earlier in the same prompt, the
       waste detector shall report the earlier edits as overproduction.
     level: L2
     source_points: [20]
     rationale: Work that is thrown away did not advance the outcome.
     status: planned
   ```

3. **Lint it**, then regenerate the index:

   ```bash
   ter-req lint --tests tests
   ter-req points
   ```

4. **Write the tests first**: positive, negative and boundary, tagged with
   the id. They fail until the detector exists.

   ```python
   @pytest.mark.req("TER-DET-011")
   class TestDeletedWork:
       def test_edit_then_delete_is_overproduction(self) -> None:
           s = Script()
           s.prompt("tidy up")
           s.edit("src/tmp.py")
           s.bash("rm src/tmp.py", "")
           [f] = found(s, "deleted_work")
           assert f.waste is LeanWaste.OVERPRODUCTION
   ```

5. **Implement** until they pass. Add a property if an invariant is involved,
   and a synthetic golden session if the behaviour is visible in reports;
   regenerate snapshots with `TER_UPDATE_GOLDEN=1` and review the diff.
6. **Promote**: set `status: verified` on the requirement, update the
   point's `status` and `verification` honestly (see the
   [definition of done guide](definition-of-done.md)), and run
   `ter-req points`.
7. **Run the gates** listed above, then commit naming the point and
   requirement ids.

From then on, CI fails if that test stops passing or loses its tag.
