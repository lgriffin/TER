# TER Calculator Improvement Roadmap — v 2.0

## Executive Summary

This roadmap consolidates the main technical, packaging, testing, and measurement-quality improvements identified during review of the project.

A major portion of Phase 1 has now been completed. Through five focused test-development tranches, the project moved from an initial measured coverage baseline of approximately **60%** to a verified **92% branch-aware coverage**, with **1,003 tests passing** under Python 3.11. The work added targeted tests for CLI orchestration, static JSONL parsing, embedding and caching logic, formatters, plugins, acceleration utilities, real-time analysis, intent construction, and optional analysis pipelines.

Phase 2 is now complete in **TER v1.04.1**. The oversized CLI and acceleration modules were split into focused packages while preserving historical imports and test monkeypatch contracts. The root `cli.py` was reduced from approximately 1,207 lines to 482 lines, and the former monolithic `acceleration.py` was replaced by a compatibility-exporting package containing cache, hashing, quick-analysis, session-watching, and parallel-execution modules.

Packaging and maintainability work was completed by making heavy ML dependencies optional, adding reproducible constraints, explicitly supporting Python 3.11–3.13, adding a CI version matrix and 90% branch-coverage floor, narrowing internal exception handling, and stabilizing the codebase until Ruff and mypy passed cleanly.

Phase 3 has now begun and two major algorithmic increments have been completed. TER v1.05 introduced a benchmark and calibration layer with JSONL annotations, binary and multiclass metrics, token-weighted evaluation, confusion matrices, bootstrap confidence intervals, conservative F0.5-oriented threshold recommendations, and a new `ter benchmark` command. TER v1.06 added deterministic structured tool-call fingerprints, and TER v1.07 added blended repetition scoring that combines semantic, lexical, entity, action, and parameter-novelty signals while preserving existing production thresholds.

TER v1.08 now implements improved intent construction. User prompts are embedded independently and combined through explicit information, recency, correction, and operational-message weights. Low-information prompts such as `continue`, `retry`, and `go ahead` retain provenance but receive minimal influence. Semantic topic-shift detection separates unrelated tasks, while the compatibility API represents the latest active topic and a new per-topic API exposes all detected intents.

TER v1.09.1 is canonically validated with **1,051 tests passed**, **Ruff clean**, and **mypy clean across 58 source files**. TER v1.10.1 implements stable JSONL block identities, deterministic sibling merging, exact-duplicate suppression, conflict retention, merge warnings, and source-line provenance propagated through fine-grained spans. The latest explicitly measured branch-aware coverage remains 92%, protected by a 90% regression floor. The remaining priorities are canonical v1.10 verification, real-data annotation and calibration, classifier explainability, and uncertainty reporting.

---

# Phase 1 — Fix Developer Experience and Reproducibility

## 1. Repair README sample-session instructions

### Problem

The README states:

```text
Sample sessions are included in sample_sessions/
```

but the documented command references a missing file:

```text
Error: Session file not found:
sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl
```

### Actions

- Add one or more real `.jsonl` files under `sample_sessions/`.
- Update every README command to reference an existing sample file.
- Add a smoke test that runs the documented example.
- Prefer a stable filename such as:

```text
sample_sessions/example_session.jsonl
```

### Acceptance criteria

- Every README command can be copied and executed successfully.
- CI validates at least one documented CLI example.
- Missing sample files cause a clear, actionable error message.

---

## 2. Make test dependencies explicit

### Problem

The test suite failed with:

```text
ModuleNotFoundError: No module named 'pytest_bdd'
```

because `pytest-bdd` was not consistently installed in the environment used to run tests.

### Actions

- Add `pytest-bdd` to the development dependency group.
- Add all test-only packages to a single reproducible installation target.
- Document the canonical test command:

```bash
python -m pytest
```

- Add a development setup command such as:

```bash
python -m pip install -e ".[dev]"
```

- Ensure CI installs the same development dependency set.

### Acceptance criteria

- A clean checkout can run all tests after one documented install command.
- `python -m pytest` works without manual package installation.
- CI and local development use the same dependency declaration.

---

## 3. Improve measured test coverage

### Status

**Completed and regression-protected.**

The original review measured approximately **60% statement coverage**. Five focused test-development tranches increased the verified full-suite result to:

```text
Tests collected:          1,035
Tests passed:             1,035
Tests failed:                 0
Statements:               6,431
Missed statements:          359
Branches:                 2,160
Partially covered branches: 236
Overall branch-aware coverage: 92%
```

The original 92% coverage verification ran under Python 3.11 with `pytest`, `pytest-bdd`, and `pytest-cov`. TER v1.07 subsequently passed the expanded 1,035-test suite in the same canonical Python 3.11 environment.

### Coverage progression

```text
Initial baseline:  60%
After tranche 1:   78%
After tranche 2:   83%
After tranche 3:   87%
After tranche 4:   91%
After tranche 5:   92%
```

The intermediate figures above are the clean, full-suite measurements recorded after each tranche. Earlier local estimates were superseded by the authoritative Python 3.11 runs.

### Test work added

#### Tranche 1 — Previously untested support modules

Added focused tests for:

- Acceleration cache lifecycle
- Cache hits, misses, expiry, corruption, invalidation, and clearing
- Quick static JSONL parsing
- Request-block merging
- Feedback history, trends, tagging, and threshold checks
- Plugin configuration and registration
- Dashboard and Rich rendering
- Error and edge-case behavior

Major gains included:

```text
acceleration.py       0%  -> approximately 80%
dashboard.py          0%  -> 96%
feedback.py           0%  -> 99%
plugins.py            0%  -> approximately 67%
rich_components.py   11%  -> 94%
```

#### Tranche 2 — Embeddings, parsing helpers, optimizer, and CLI foundations

Added deterministic tests for:

- Embedding-model loading and device selection
- Embedding cache reads, writes, corruption recovery, batching, and reuse
- Fake embedding pipelines that avoid network and hardware dependencies
- Budget optimizer fallback and edge cases
- Comparison sorting and aggregation
- Configuration parsing
- Loader helper functions and session discovery
- Initial CLI dispatch and error handling

Major gains included:

```text
embedding_cache.py   32% -> 98%
budget_optimizer.py  66% -> 99%
compare.py             0% -> 100%
config_parse.py        69% -> 100%
loader.py              75% -> 85%
cli.py                 21% -> 39%
```

#### Tranche 3 — Context commands, watch paths, and formatters

Added tests for:

- Context store, graph, optimize, delta, and consistency commands
- Invalid context subcommands
- Watch-mode path and initialization failures
- Graceful interruption handling
- Text and Rich output matrices
- Economics, cost weighting, overthinking, grouped analysis, and input-analysis output
- Empty and low-information formatting states

Major gains included:

```text
cli.py              39% -> 65%
formatter_rich.py   64% -> 99%
formatter_text.py   66% -> 100%
```

#### Tranche 4 — Plugins, acceleration fallbacks, grouped analysis, and CLI expansion

Added tests for:

- Modern and legacy plugin discovery
- Plugin registration, loading, duplication, and configuration failures
- Grouped parent/subagent analysis
- Baseline and ordinary comparison modes
- Budget-history failures
- Signal logging
- Latest-session and directory watch paths
- Session watcher callbacks
- Parallel embedding success and fallback behavior

Major gains included:

```text
cli.py            65% -> 83%
acceleration.py   80% -> 91%
plugins.py        67% -> 99%
```

#### Tranche 5 — Real-time analysis, intent, and optional pipeline behavior

Added tests for:

- Real-time economics and cache-hit metrics
- Context growth and bloat detection
- Healthy, improving, stable, and degrading signals
- Alignment rules by phase
- Repetition history
- Rolling TER accounting
- Duplicate and malformed JSONL records
- Filesystem failures
- Monitor callbacks and polling loops
- Intent extraction and confidence boundaries
- Full optional analysis pipeline
- Minimal pipeline paths with optional features disabled

Final module results included:

```text
analyze_pipeline.py   98%
intent.py            100%
real_time.py          92%
```

### Final coverage profile

The strongest-covered modules now include:

```text
compute.py              100%
models.py               100%
compare.py              100%
config_parse.py         100%
formatter_text.py       100%
intent.py               100%
plugins.py               99%
formatter_rich.py        99%
budget_optimizer.py      99%
embedding_cache.py       98%
analyze_pipeline.py      98%
adaptive_budget.py       98%
overthinking.py          98%
```

The largest remaining statement gaps are concentrated in:

```text
cli.py                  97 missed statements
real_time.py            28 missed statements
acceleration.py         28 missed statements
intent_extraction.py    28 missed statements
validation.py           27 missed statements
loader.py               22 missed statements
consistency.py          14 missed statements
fragment_store.py       13 missed statements
formatter.py            13 missed statements
```

### Why the work stopped at 92%

The original target was to explore whether coverage could be raised into the 95% range. A genuine 95% branch-aware result would require disproportionately expensive tests for:

- Rare CLI orchestration paths
- Defensive exception branches
- Filesystem and concurrency races
- Hardware-specific execution
- Low-frequency parser corruption cases
- Internal fallback branches with limited product risk

At 92%, the project has strong coverage of core calculations, model behavior, parsing, reporting, caching, plugins, and real-time logic. Further tests should now be driven by defects, feature changes, and risk rather than by the global percentage alone.

### Remaining actions

- Add the authoritative coverage command to CI:

```bash
python -m pytest \
  --cov=ter_calculator \
  --cov-branch \
  --cov-report=term-missing
```

- Publish XML or HTML coverage artifacts in CI.
- Introduce a conservative regression gate, initially around 90%.
- Require focused tests for newly added and modified branches.
- Avoid reducing the measured baseline without an explicit review.
- Resolve the existing `pytest-bdd` deprecation warnings before pytest 10.
- Use diff coverage for pull requests so new code remains highly covered.

### Updated acceptance criteria

- A clean Python 3.11 environment runs all 1,035 tests successfully.
- CI publishes statement and branch coverage.
- The global branch-aware coverage baseline remains at or above 90%.
- Coverage regressions fail CI unless deliberately approved.
- New modules and modified branches receive focused behavioral tests.
- Test design prioritizes meaningful assertions over line-only execution.

---

# Phase 2 — Improve Maintainability and Packaging

## 4. Split oversized modules

### Status

**Implemented in TER v1.03.**

The first structural Phase 2 refactor split the two highest-priority oversized modules while preserving behavior and compatibility.

### CLI implementation

The CLI command implementations now live under:

```text
src/ter_calculator/
    commands/
        __init__.py
        analyze.py
        report.py
        watch.py
        context.py
        budget.py
        listing.py
        hook.py
```

The root `cli.py` was reduced from approximately:

```text
1,207 lines -> 482 lines
```

It now retains:

- Argument parser construction
- Command registration
- Shared CLI-facing utilities
- Top-level exception handling
- Entry-point dispatch
- Thin compatibility wrappers for historical private command functions

The additional `listing.py` and `hook.py` modules were introduced because those responsibilities were distinct from analysis, reporting, watching, context operations, and budgeting.

### Acceleration implementation

The former monolithic module:

```text
src/ter_calculator/acceleration.py
```

was replaced by:

```text
src/ter_calculator/acceleration/
    __init__.py
    cache.py
    quick_analyser.py
    session_watcher.py
    parallel.py
    hashing.py
```

Responsibilities are now separated as follows:

| Module | Responsibility |
|---|---|
| `cache.py` | Analysis cache lifecycle, cache entries, statistics, expiry, invalidation, and persistence |
| `quick_analyser.py` | Lightweight static JSONL analysis and approximate TER calculation |
| `session_watcher.py` | Session-file polling, filesystem events, callbacks, and watch state |
| `parallel.py` | Parallel embedding and fallback execution paths |
| `hashing.py` | Stable file and content hashing helpers |
| `__init__.py` | Compatibility exports for the historical public API |

### Compatibility preserved

Historical imports remain supported:

```python
from ter_calculator.acceleration import (
    AnalysisCache,
    CacheStats,
    QuickAnalyser,
    SessionWatcher,
    WatchEvent,
    WatchEventType,
    parallel_embed,
    hash_file,
)
```

Constants such as the following are also re-exported:

```text
DEFAULT_CACHE_DIR
CACHE_VERSION
EMBEDDING_DIM
```

Existing CLI imports remain valid:

```python
from ter_calculator.cli import main
import ter_calculator.cli as cli
```

The private command names used by the existing test suite and monkeypatch-based tests were retained through thin wrappers.

### Tests added

Explicit architecture and compatibility regression tests were added for:

- Command-module imports
- Root CLI compatibility wrappers
- Historical acceleration imports
- Public constant exports
- Entry-point behavior
- Monkeypatch contracts used by existing tests

### Validation

The refactor produced the following verified results in the available environment:

```text
Python compilation: successful
Focused compatibility suite: 94 passed
Available unit/integration suite: 766 passed
```

The complete BDD and semantic-embedding suite could not be executed in the refactoring environment because `pytest-bdd` and `sentence-transformers` were unavailable there. No available test failure was caused by the structural changes.

The canonical full verification remains:

```bash
python -m pip install -e ".[dev]"

python -m pytest \
  --cov=ter_calculator \
  --cov-branch \
  --cov-report=term-missing
```

### Documentation and versioning

The refactor also:

- Bumped the package version to `0.3.0`
- Added `docs/005-module-split-v03.md`
- Added a v0.3.0 entry to `UPDATES.md`
- Removed repository caches and Git metadata from the distributed v03 archive

### Acceptance criteria status

- [x] Command implementations were extracted from the root CLI module.
- [x] Each extracted command module has a focused responsibility.
- [x] The acceleration module was split by responsibility.
- [x] Existing external acceleration imports remain supported.
- [x] Existing CLI imports and test monkeypatch contracts remain supported.
- [x] Focused architecture and compatibility tests pass.
- [x] Available unit and integration tests pass.
- [x] Re-ran the complete BDD and embedding-enabled suite in the canonical Python 3.11 development environment: 1,035 tests passed in v07.
- [ ] Consider further reduction of the 482-line root `cli.py` only when it can be done without weakening parser readability or compatibility.

### Remaining oversized modules

This item is complete for CLI and acceleration, but the repository still contains other large modules that may benefit from later risk-driven extraction:

```text
real_time.py
validation.py
waste_detectors.py
plugins.py
embedding_cache.py
```

These should not be split solely to reduce line count. Future extraction should be driven by change frequency, coupling, test isolation, and clear responsibility boundaries.

---

## 5. Reduce installation weight

### Status

**Implemented in TER v1.04.**

Heavy semantic-embedding functionality was moved behind an optional installation extra so the core package no longer requires PyTorch or a Hugging Face model download.

### Implementation

The default installation remains lightweight:

```bash
pip install ter-calculator
```

Semantic embedding functionality is installed explicitly:

```bash
pip install "ter-calculator[embeddings]"
```

Anthropic-assisted functionality is also isolated behind an optional extra:

```bash
pip install "ter-calculator[llm]"
```

Development environments can install the required combinations explicitly:

```bash
python -m pip install -e ".[dev,embeddings]"
```

### Behavior

- Core CLI commands install without `sentence-transformers`.
- Embedding-dependent paths provide actionable installation guidance.
- Lazy and optional imports prevent base-install failures.
- Historical error-message compatibility was retained where tests depended on it.
- README and contributor documentation distinguish base, embedding-enabled, and LLM-enabled installations.

### Acceptance criteria status

- [x] Core CLI commands install without PyTorch or model downloads.
- [x] Embedding-dependent commands fail gracefully when the optional dependency is absent.
- [x] README documentation distinguishes base and ML-enabled installations.
- [x] Lightweight editable installation was smoke-tested successfully.

---

## 6. Constrain dependencies for reproducibility

### Status

**Implemented in TER v1.04.**

The repository now includes reproducible constraint files:

```text
constraints/dev.txt
constraints/ci.txt
```

High-risk dependencies use tested upper bounds, including the semantic-embedding stack.

### Implementation

- Development and CI environments install against committed constraints.
- Package metadata remains appropriately flexible while CI uses known-good bounds.
- Optional extras are represented in the documented install commands.
- The canonical development command is:

```bash
python -m pip install -c constraints/dev.txt -e ".[dev,embeddings]"
```

### Acceptance criteria status

- [x] CI installs from a reproducible dependency set.
- [x] Development setup uses committed constraints.
- [x] Optional dependency combinations are documented.
- [ ] Automated dependency-update pull requests remain a future repository-management enhancement.

---

## 7. Clarify supported Python versions

### Status

**Implemented in TER v1.04.**

The package now explicitly supports:

```text
Python 3.11
Python 3.12
Python 3.13
```

The declared range excludes unvalidated future versions:

```toml
requires-python = ">=3.11,<3.14"
```

CI uses a Python-version matrix for all declared versions.

### Acceptance criteria status

- [x] Every declared Python version is represented in CI.
- [x] Unsupported future Python versions are excluded until validated.
- [x] Optional dependency installation is included in the canonical CI setup.
- [x] Python 3.11 canonical validation passed with the full 1,010-test suite.

---

## 8. Narrow broad exception handling

### Status

**Implemented for the Phase 2 target scope in TER v1.04 and stabilized in v04.1.**

Broad exception handling was reviewed and narrowed in core internal paths. Broad catches were retained only where they represent intentional process or isolation boundaries.

### Narrowed areas

- Token-estimation fallback
- Cache deserialization and corruption handling
- Optional dependency imports
- Typed conversion and result handling exposed by mypy
- Internal code paths identified during the v04.1 stabilization pass

### Retained broad boundaries

- CLI entry-point dispatch
- Plugin isolation
- Watcher and monitor loops
- Optional API boundaries
- Multiprocessing fallback
- Best-effort telemetry or rendering boundaries

### Stabilization results

The v04.1 pass resolved all reported source typing failures and source-level Ruff findings.

```text
ruff check src tests
All checks passed

mypy src
Success: no issues found in 56 source files
```

### Test-lint policy

Production source remains strict. Legacy and tranche-style tests use narrowly scoped per-file exceptions for formatting and fixture-density rules:

```text
E401
E701
E702
F401
F403
F405
F841
```

Correctness-oriented Ruff rules remain enabled for the test tree.

### Acceptance criteria status

- [x] Core parsing, caching, and validation code avoids unnecessary unqualified broad catches.
- [x] Remaining broad catches correspond to documented recovery or isolation boundaries.
- [x] Expected failure modes remain covered by tests.
- [x] Ruff passes across `src` and `tests`.
- [x] Mypy passes across all 56 source files.

---

# Phase 3 — Validate TER as a Measurement Product

## 9. Build an empirical validation program

### Status

**Partially implemented in TER v1.05.**

TER now includes the technical foundation required to evaluate annotated data, but the real multi-session, multi-annotator benchmark dataset is not yet available.

### Implemented evaluation layer

TER v1.05 added:

- A validated JSONL benchmark schema
- Binary waste-versus-aligned evaluation
- Multiclass per-label evaluation
- Record-level and token-weighted metrics
- Precision, recall, F1, F0.5, and accuracy
- Confusion matrices
- Deterministic session-level bootstrap confidence intervals
- Conservative threshold calibration
- Optional minimum-precision constraints
- Text and JSON reports
- A new CLI command:

```bash
ter benchmark benchmarks/example_annotations.jsonl
```

### Added implementation assets

```text
src/ter_calculator/evaluation.py
src/ter_calculator/commands/benchmark.py
benchmarks/example_annotations.jsonl
docs/annotation_guidelines.md
tests/unit/test_evaluation.py
```

The benchmark command is advisory. It does not silently modify production thresholds.

### Pilot annotation workflow

The repository also supports a practical export-and-review flow:

1. Export classified spans from a real or fixture JSONL session.
2. Review them in CSV form.
3. Assign independent human `gold_label` values.
4. Convert reviewed CSV files into benchmark JSONL.
5. Run `ter benchmark`.

A synthetic pilot verified the workflow end to end, but synthetic labels are not valid calibration evidence.

### Current empirical limitation

The available pilot contains only one reviewed session. That is sufficient for workflow testing, but not for statistical claims, inter-rater agreement, or production threshold changes.

The following remain outstanding:

- A representative real-session sample
- Independent human labels
- Multiple annotators
- Inter-rater agreement
- A frozen calibration split
- A frozen test split
- Per-category error analysis
- Model and embedding sensitivity studies
- Release-to-release regression thresholds based on real data

### Acceptance criteria status

- [x] Benchmark schema exists.
- [x] Benchmark CLI exists.
- [x] Binary and multiclass metrics are implemented.
- [x] Token-weighted metrics are implemented.
- [x] Bootstrap confidence intervals are implemented.
- [x] Advisory threshold calibration is implemented.
- [x] Synthetic regression fixtures are committed.
- [ ] Build a representative real labeled session dataset.
- [ ] Use multiple independent annotators.
- [ ] Measure inter-rater agreement.
- [ ] Freeze calibration and final test splits.
- [ ] Require benchmark evidence before production threshold changes.

---


---

## 10. Improve intent construction

### Status

**Implemented in TER v1.08. Canonical full-suite verification pending.**

### Previous weakness

The current implementation weights recent prompts by repeating their text:

```python
parts = [prompts[0]]
for prompt in prompts[1:]:
    parts.extend([prompt, prompt])
```

This changes the embedded text instead of combining independently calculated vectors.

### Implemented method

1. Embed each prompt independently in one batch.
2. Assign explicit information, recency, correction, and operational weights.
3. Normalize the weights.
4. Compute an L2-normalized weighted centroid.
5. Detect semantic shifts between adjacent informative prompts.
6. Represent the latest active topic through the compatibility `extract_intent` API.
7. Expose all topic-specific vectors through `extract_intent_topics`.
8. Preserve every original prompt in `source_prompts` for auditability.
9. Exclude low-information operational prompts from display text and strongly down-weight them in the centroid.

The intent vector should be:

\[
I =
\frac{
\sum_i w_i e_i
}{
\left\lVert \sum_i w_i e_i \right\rVert
}
\]

where:

- \(e_i\) is the embedding of prompt \(i\)
- \(w_i\) is its recency or information-content weight

### Candidate weighting factors

- Recency
- Prompt length
- Novel entity introduction
- Explicit goal language
- User correction strength
- Topic-shift probability

### Exclude or down-weight prompts such as

```text
continue
yes
go ahead
retry
do that
```

unless they contain new task information.

### Added assets

```text
src/ter_calculator/intent_construction.py
tests/unit/test_intent_construction.py
docs/010-weighted-intent-construction-v08.md
```

### Acceptance criteria status

- [x] Intent vectors are calculated from independently embedded prompts.
- [x] Weighting is explicit and testable.
- [x] Topic shifts create separate intent clusters.
- [x] The compatibility API follows the latest active topic.
- [x] Operational prompts do not dominate intent representation.
- [x] Corrections receive explicit additional weight.
- [x] Historical prompt provenance is preserved.
- [ ] Validate the complete suite in the canonical embedding-enabled environment.
- [ ] Calibrate topic-shift and prompt-weight parameters on real labeled sessions.

---

## 11. Replace one-dimensional repetition detection

### Status

**Implemented in two increments: structured tool-call fingerprints in TER v1.06 and blended repetition scoring in TER v1.07.**

### TER v1.06 — Structured tool-call fingerprints

The static parser now preserves structured tool metadata on spans:

```text
tool_name
tool_input
```

A deterministic fingerprinting layer normalizes:

- Tool name
- File and directory paths
- Line ranges
- Offsets and limits
- Search queries and patterns
- Shell commands
- Nested arguments

Volatile metadata is excluded, including request IDs, timestamps, tool-use IDs, and descriptive fields.

Behavior now distinguishes:

- Same tool, same path, same range: strong duplicate evidence
- Same file, different range: parameter novelty
- Same search tool, different query: distinct action
- Same shell tool, different command: distinct action
- Missing structured metadata: semantic fallback

Added assets include:

```text
src/ter_calculator/tool_fingerprints.py
tests/unit/test_tool_fingerprints.py
benchmarks/tool_call_adversarial.jsonl
docs/008-structured-tool-fingerprints-v06.md
```

### TER v1.07 — Blended repetition scoring

The classifier now combines:

\[
R =
lpha S_{semantic}
+ eta S_{lexical}
+ \gamma S_{entity}
+ \delta S_{action}
- \lambda N_{parameter}
\]

where \(N_{parameter}\) represents parameter novelty.

The score exposes:

```text
score
semantic
lexical
entity
action
parameter_novelty
exact_duplicate
```

Structured exact duplicates remain authoritative. Parameter novelty prevents identical or near-identical embeddings from overriding meaningful changes in path, range, query, command, identifier, or numeric reference.

Reasoning and generation comparisons now also reduce repetition scores when newer text introduces materially different entities or actions.

Added assets include:

```text
src/ter_calculator/repetition_scoring.py
tests/unit/test_repetition_scoring.py
docs/009-blended-repetition-v07.md
```

### Compatibility and threshold policy

- Existing classifier interfaces remain compatible.
- Existing phase thresholds remain unchanged.
- Exact structured duplicates score as strong repetition.
- Semantic fallback remains available.
- Synthetic adversarial benchmarks are regression fixtures, not real-world accuracy evidence.

### Canonical validation

```text
TER v1.06:
  1,029 tests passed
  Ruff clean
  Mypy clean across 55 source files

TER v1.07:
  1,035 tests passed
  Ruff clean
  Mypy clean across 56 source files
```

### Acceptance criteria status

- [x] Tool calls use structured fingerprints.
- [x] Reasoning and generation use multiple independent similarity signals.
- [x] Parameter novelty is explicitly represented.
- [x] Exact duplicates remain deterministic and explainable.
- [x] Existing production thresholds remain unchanged pending real calibration data.
- [x] Synthetic false-positive and false-negative regression cases are committed.
- [ ] Measure real false positives and false negatives on a labeled dataset.
- [ ] Calibrate signal weights and thresholds empirically.

---


---

## 12. Calibrate thresholds empirically

### Status

**Calibration tooling implemented in TER v1.05; production calibration remains pending real labeled data.**

### Current thresholds

```text
REASONING: 0.88
TOOL_USE: 0.93
GENERATION: 0.88
```

These values appear to come from a small internal gold set and may not generalize.

### Calibration procedure

For each phase and waste category:

- Compute precision
- Compute recall
- Compute F1
- Compute F0.5
- Plot precision-recall curves
- Generate confusion matrices
- Bootstrap confidence intervals
- Perform leave-one-session-out validation

Because false accusations of waste are particularly damaging, optimize for precision.

Use:

\[
F_\beta,\quad \beta < 1
\]

For example:

\[
F_{0.5}
\]

weights precision more heavily than recall.

### Acceptance criteria status

- [x] Calibration scripts are committed to the repository.
- [x] F0.5 and minimum-precision optimization are implemented.
- [x] Bootstrap confidence intervals are implemented.
- [x] Threshold recommendations are advisory and do not alter production defaults.
- [ ] Derive production thresholds from real labeled validation data.
- [ ] Calibrate thresholds separately by phase and evidence type.
- [ ] Require benchmark evidence for future threshold changes.
- [ ] Report stable per-phase confidence intervals from a multi-session dataset.

### Update — real-data attempt (post-v16)

A first real attempt to build toward "derive production thresholds from real
labeled validation data" surfaced a prerequisite this roadmap didn't
anticipate: **not all real session corpora have enough span-level structure
to calibrate against.**

- `scripts/labeling_priority.py` was built to rank unlabeled sessions by
  threshold-sensitivity (how much waste ratio moves across a small grid of
  `similarity_threshold` / `confidence_threshold` values), as a way to pick
  which sessions are worth the cost of hand-labeling first, given a corpus
  too large to label in full.
- Run against ~145 sessions from a real, long-running project, 141 of 143
  parsed sessions collapsed to exactly 2 spans each. This traced to the
  workload being **bursty** (many independent single-shot tasks) rather than
  **long-lived** (one session with many tool-using turns). Verified against
  raw JSONL — correct parsing, not a bug: those sessions have no tool use,
  no retries, no repeated reasoning, so there is nothing for the existing
  waste detectors to find.
- Practical implication: this corpus alone cannot supply the 20–50+
  hand-labeled span gold set this section calls for, because only a small
  handful of its sessions have span counts high enough for the classifier's
  decisions to be meaningfully tested. A gold set drawn only from bursty
  single-shot sessions would under-represent exactly the failure modes
  (duplicate tool calls, retry spirals, restated reasoning) the thresholds
  above are supposed to catch.
- Updated acceptance criterion: before scoring "derive production thresholds
  from real labeled data" as done, confirm the source sessions include a
  meaningful share of long-lived, multi-tool-call sessions, not only bursty
  single-shot ones. Bursty corpora remain useful for a *different*, currently
  unbuilt feature — cross-session redundancy detection (is task N redoing
  work task N-1 already did) — which is out of scope for the phase-level
  waste detectors this section covers.

---

# Phase 4 — Improve Parsing and Scoring Robustness

## 13. Segment spans more precisely

### Status

**Implemented as an opt-in compatibility path in TER v1.09.**

### Implementation

TER v1.09 introduces a dedicated segmentation module:

```text
src/ter_calculator/span_segmentation.py
```

Reasoning and generated-response blocks can now be divided at:

- Paragraph boundaries
- Markdown headings and horizontal rules
- Sentence-group boundaries for oversized units
- Discourse transitions such as `now`, `next`, `again`, `in summary`, `however`, and `finally`

Adjacent fragments below the configured minimum are merged when doing so does not exceed the maximum target size. Tool calls and tool results remain atomic so structured tool fingerprints and argument comparison are preserved.

### Configuration

```bash
ter analyze tests/fixtures/sample_session.jsonl \
  --fine-segmentation \
  --segment-min-tokens 12 \
  --segment-max-tokens 180
```

The feature is opt-in in v1.09. Existing calls to `segment_spans(session)` retain the historical one-content-block-per-span behavior, keeping prior reports comparable.

### Provenance

Every `TokenSpan` can now retain:

```text
parent_block_id
segment_index
char_start
char_end
source_message_uuid
block_type
source_role
phase
```

This allows exact highlighting and reconstruction of each segment from its parent block.

### Added assets

```text
src/ter_calculator/span_segmentation.py
tests/unit/test_span_segmentation.py
docs/011-fine-span-segmentation-v1.09.md
```

### Acceptance criteria status

- [x] Mixed-purpose reasoning and generation blocks can produce multiple spans.
- [x] Minimum and maximum token targets are configurable.
- [x] Segments preserve exact character offsets and parent-block identity.
- [x] Role, phase, source message, and block type remain available.
- [x] Tool calls and results remain atomic.
- [x] Legacy block-level behavior remains available by default.
- [ ] Compare block-level and segment-level accuracy on real labeled sessions.
- [ ] Decide whether fine segmentation should become the default after empirical validation.

---

## 14. Strengthen JSONL merge identity

### Status

**Implemented in TER v1.10.1.**

### Stable identity model

TER now calculates a normalized SHA-256 fingerprint for every content block using its role and semantic block content. Volatile transport metadata is excluded, including timestamps, request IDs, UUIDs, and parent UUIDs. Dictionary order and text whitespace are normalized before hashing.

Sibling records are identified in this order:

1. Shared `requestId` and role
2. Shared explicit `messageId` or `message_id` and role
3. Source-line identity fallback

UUID-only records remain independent to preserve historical loader behavior.

### Deterministic merge behavior

- Preserve first-seen message and block order.
- Emit exact content fingerprints once.
- Retain all source lines contributing to a duplicate block.
- Retain distinct sibling blocks rather than discarding conflicts.
- Record a merge warning when distinct siblings are combined.
- Backfill usage metadata when the first sibling lacks it.
- Preserve malformed-line failure messages with exact source line numbers.

### Provenance fields

`ContentBlock` and `TokenSpan` now retain:

```text
source_line
source_lines
content_fingerprint
source block index
```

`Message` retains source lines and merge warnings. `Session` exposes aggregate merge warnings. Fine segmentation propagates the same block identity to every derived segment.

### Added assets

```text
src/ter_calculator/jsonl_identity.py
tests/unit/test_jsonl_identity_v1.10.py
docs/012-jsonl-identity-provenance-v1.10.md
```

### Compatibility

- Existing model constructors remain valid because provenance fields have defaults.
- `_deduplicate_entries()` remains available as a compatibility wrapper.
- Request-ID sibling merging remains compatible when sibling UUIDs differ.
- UUID-only entries remain separate unless a stronger shared message identifier exists.

### Acceptance criteria status

- [x] Exact duplicate blocks are not emitted twice.
- [x] Distinct sibling blocks are retained.
- [x] Merge order is deterministic.
- [x] Every parsed block and derived span can be traced to source lines.
- [x] Stable normalized block fingerprints are exposed.
- [x] Conflicting sibling merges generate warnings.
- [ ] Add tolerant recovery for a corrupt final line as an explicit opt-in mode.
- [ ] Expose merge warnings in all CLI and JSON report formats.

---

---

## 15. Add uncertainty reporting

### Problem

A score such as:

```text
TER: 0.83
```

appears more precise than the underlying heuristic classification warrants.

### Proposed output

```text
TER estimate: 0.83
Bootstrap 95% interval: 0.77–0.88
Low-confidence tokens: 14.2%
```

### Sources of uncertainty

- Embedding model sensitivity
- Threshold uncertainty
- Segmentation choice
- Classifier confidence
- Annotation disagreement
- Small sample size
- Uncertain intent assignment

### Suggested implementation

- Attach confidence to every classified span.
- Track the token share below a confidence threshold.
- Bootstrap spans or sessions to estimate score intervals.
- Report model/version metadata with the score.
- Mark very small sessions as low-confidence.

### Acceptance criteria

- CLI and JSON outputs expose uncertainty fields.
- Confidence intervals are reproducible.
- Low-confidence classifications are inspectable.
- Reports identify the model, thresholds, and classifier version used.

---


# Current Implementation Snapshot — TER v1.11

The current implementation baseline is **TER v1.11**, extending v08 weighted intent construction with opt-in, provenance-preserving fine span segmentation.

## Completed work

- Approximately 60% to 92% branch-aware coverage improvement, protected by a 90% CI floor
- v08 canonical validation: 1,043 tests passed
- Ruff clean across source and tests
- Mypy clean across 57 v08 source files
- Benchmark and calibration CLI
- Structured tool-call fingerprints
- Blended repetition scoring
- Weighted intent centroids and topic-shift detection
- Fine segmentation for reasoning and generated-response blocks
- Configurable segment token bounds
- Parent-block IDs, segment indexes, and exact character offsets
- Atomic tool-call preservation
- Stable JSONL block fingerprinting and deterministic merge provenance added
- Package version updated to 0.10.0

## Validation status

The authoritative completed baseline supplied for v08 is:

```text
Environment: Python 3.11.15
Tests collected: 1,043
Tests passed: 1,043
Warnings: 34

ruff check src tests:
All checks passed

mypy src:
Success: no issues found in 57 source files
```

TER v1.09 adds seven focused segmentation tests, bringing full-suite collection to 1,050 tests in the packaged source. Local validation completed:

```text
Focused segmentation, loader, and classifier tests: 59 passed
Ruff: all checks passed
Python compilation: passed
Full suite collection observed locally: 1,049 before the final added boundary test
```

The local full suite could not complete because the restricted environment could not access the existing Hugging Face embedding model. Full mypy also exceeded the local execution limit. Canonical Python 3.11 verification remains required for v1.09.

## Warning status

The existing 34 warnings originate from `pytest-bdd` compatibility with APIs scheduled for removal in pytest 10. They are not application failures.

## Phase status

```text
Phase 1: Complete
Phase 2: Complete
Phase 3: In progress
  - Benchmark tooling: complete
  - Structured repetition detection: complete
  - Blended repetition scoring: complete
  - Improved intent construction: complete in v08
  - Real-data calibration: pending
Phase 4: Complete
  - Fine span segmentation: implemented in v1.09
  - Stronger JSONL identity and merge provenance: complete in v1.10
  - Explainability and uncertainty: pending
```

---


## v1.09 compatibility correction

Canonical validation identified a compatibility regression in `analyze_pipeline.py`: the pipeline always passed a second `SegmentationConfig` argument to `segment_spans`, including when fine segmentation was disabled. Existing integrations and tests that replaced `segment_spans` with the historical one-argument callable therefore failed.

The corrected implementation now:

- Calls `segment_spans(session)` on the default compatibility path.
- Passes `SegmentationConfig` only when `--fine-segmentation` is enabled.
- Adds a regression test for the enabled configuration path.
- Uses the real repository fixture `tests/fixtures/sample_session.jsonl` in v1.09 examples instead of the placeholder `session.jsonl`.

Focused correction validation:

```text
Pipeline and segmentation tests: 11 passed
Ruff: all checks passed
Python compilation: passed
```

The user-reported pre-correction canonical run collected 1,050 tests and produced exactly two compatibility failures; Ruff and mypy remained clean across 58 source files. The corrected archive adds one regression test, so the next complete canonical run should collect 1,051 tests.

---

# TER v1.11 Increment — Explainability and Uncertainty

## Status

**Implemented in TER v1.11.**

TER now exposes why each span received its label and quantifies how sensitive the
session score is to the observed classifications.

### Per-span evidence

Each classified span can include:

- Stable reason code and readable summary
- Intent and repetition scores
- Semantic, lexical, entity, and action similarities
- Parameter novelty
- Applicable repetition threshold
- Strongest matched prior span position and excerpt

Complete evidence is available in JSON output. Text reports show a compact list
of waste and low-confidence decisions for review.

### Session-level uncertainty

TER results now include:

- Mean classification confidence
- Token-weighted confidence
- Low-confidence token count and share
- Reproducible 95% span-bootstrap TER interval
- Bootstrap sample count and method
- Session reliability level based on span count
- Classifier version metadata

The interval measures sensitivity to the spans observed in the session. It does
not replace uncertainty from annotation disagreement, threshold calibration, or
embedding-model choice.

### Added assets

```text
src/ter_calculator/uncertainty.py
tests/unit/test_explainability_uncertainty_v1.11.py
docs/013-explainability-uncertainty-v1.11.md
```

### Validation

```text
Focused v1.11 and compatibility tests: 74 passed
Ruff: all checks passed
Mypy: no issues across 60 source files
Full suite collection: 1,061 tests
```

Production thresholds remain unchanged. Real multi-session annotation and
calibration remain necessary before treating the interval or confidence values
as externally validated measurement uncertainty.

---

# Proposed Delivery Sequence

## Milestone 1 — Reproducible development setup

- [x] Repair README sample paths
- [x] Add explicit development dependencies
- [x] Standardize on `python -m pytest`
- [x] Add coverage reporting and enforce the verified 90%+ baseline
- [x] Add Python-version CI matrix

## Milestone 2 — Packaging and maintainability

- [x] Split `cli.py` into focused command modules in v03
- [x] Split `acceleration.py` into a compatibility-preserving package in v03
- [x] Make ML and LLM dependencies optional in v04
- [x] Add reproducible development and CI constraints in v04
- [x] Add explicit Python 3.11–3.13 support and CI matrix
- [x] Narrow internal exception handling
- [x] Make Ruff pass across source and tests in v04.1
- [x] Make mypy pass across all 56 source files in v04.1
- [ ] Reassess other oversized modules only when coupling and change-frequency evidence justify extraction

## Milestone 3 — Measurement validation

- [x] Create initial annotation guidelines
- [x] Add benchmark schema and CLI
- [x] Add precision, recall, F1, F0.5, confusion matrices, and bootstrap intervals
- [x] Add advisory threshold calibration with minimum-precision support
- [x] Add synthetic benchmark and adversarial regression fixtures
- [ ] Build a representative real labeled evaluation dataset
- [ ] Measure inter-rater agreement
- [ ] Freeze calibration and final test splits
- [ ] Calibrate production thresholds using real data

## Milestone 4 — Algorithm improvements

- [x] Replace repeated-text intent weighting with weighted vector centroids in v08
- [x] Add topic-shift detection in v08
- [x] Implement structured tool-call fingerprints in v06
- [x] Add blended repetition scoring in v07
- [x] Introduce finer span segmentation in v1.09

## Milestone 5 — Robustness and uncertainty

- [x] Add stable block fingerprints in v1.10
- [x] Preserve JSONL provenance in v1.10
- Handle partial and corrupt writes
- [x] Add uncertainty estimates and confidence intervals in v1.11
- Publish release-to-release benchmark results

---

# Suggested Priority Matrix

| Priority | Work item | Impact | Effort |
|---|---|---:|---:|
| P0 | Fix README sample paths | High | Low |
| P0 | Add `pytest-bdd` to dev dependencies | High | Low |
| P0 | Standardize test invocation | High | Low |
| P0 | Add CI Python matrix | Completed in v04 | Completed |
| P1 | Add reproducible constraints | Completed in v04 | Completed |
| P1 | Make embeddings optional | Completed in v04 | Completed |
| P1 | Maintain 90%+ branch coverage | 92% verified | Ongoing |
| P1 | Narrow broad exception handling | Completed for Phase 2 | Completed |
| P2 | Split CLI and acceleration modules | Completed in v03 | Completed |
| P2 | Build labeled evaluation dataset | Tooling complete; real data pending | High |
| P2 | Calibrate thresholds empirically | Tooling complete; real data pending | High |
| P2 | Improve intent construction | Completed in v08 | Completed |
| P2 | Structured repetition detection | Completed in v06/v07 | Completed |
| P3 | Finer span segmentation | Completed in v1.09 | Completed |
| P3 | Stronger JSONL identity rules | Completed in v1.10 | Completed |
| P3 | Add uncertainty reporting | Completed in v1.11 | Completed |

---

# Definition of Done

The improvement program should be considered complete when:

- The test suite maintains at least 90% branch-aware coverage; the current verified result is 92%, with 1,035 functional tests passing in v07.

- A clean checkout installs and runs from documented commands.
- All declared Python versions are represented in the CI matrix; canonical Python 3.11 validation passes.
- Dependencies are reproducible.
- Core functionality does not require a heavy ML stack.
- CLI and acceleration modules have been split by responsibility; remaining large modules are reviewed using risk and coupling evidence.
- Broad internal exception handling is substantially reduced, Ruff passes, and mypy reports no issues across 56 source files.
- TER is evaluated against a labeled, multi-annotator benchmark.
- Thresholds are empirically calibrated.
- Repetition detection uses structured and blended evidence. This is implemented in v06/v07; empirical calibration remains pending.
- Intent is represented through weighted embeddings rather than repeated text. This is implemented in v08; empirical weight calibration remains pending.
- Mixed-purpose reasoning and response blocks can be segmented with exact source provenance. This is implemented as an opt-in path in v1.09.
- JSONL merging is deterministic and provenance-preserving.
- TER reports include uncertainty and confidence information. Implemented in v1.11; empirical validation remains pending.


## v1.10.1 Stabilization Note

The first v1.10 canonical run produced:

```text
1,057 tests passed in the latest complete canonical v1.10.1 run; v1.11 collects 1,061 tests pending canonical execution
Ruff clean
One mypy error in loader.py
```

The error was caused by a temporary variable whose inferred type was `dict` in one branch and `str` in another. v1.10.1 preserves the runtime behavior but appends the dictionary and string branches separately, allowing mypy to narrow each branch correctly.

Focused v1.10.1 validation:

```text
29 loader, identity, and pipeline tests passed
Ruff clean
Targeted mypy check for loader.py passed
```

The canonical full-suite expectation remains 1,057 passing tests with Ruff and mypy clean across 60 source files.
