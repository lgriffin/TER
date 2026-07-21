# TER Calculator

> **TER 2.0.14** adds closed-loop project intelligence: repository-aware guidance, sustained efficiency-degradation policies, automatic context refresh/replanning, and measured intervention outcomes. Earlier release milestones are documented in the phase notes and `UPDATES.md`.


[![CI](https://github.com/lgriffin/TER/actions/workflows/ci.yml/badge.svg)](https://github.com/lgriffin/TER/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Token Efficiency Ratio (TER) calculator for Claude Code sessions.

TER measures how efficiently an AI coding agent uses its token budget by classifying token spans as **aligned** with the user’s intent or as potential **waste**, such as redundant reasoning, repeated tool calls, unnecessary restatement, or over-explanation.

The project also provides session economics, real-time monitoring, context optimization, intent construction, repetition analysis, evaluation tooling, and regression checks.

> TER is a heuristic analysis system. Its scores should be interpreted as decision-support signals, not as ground-truth judgments.

---

## Highlights

- **TER scoring** with phase-aware analysis for reasoning, tool use, and generation
- **Waste detection** for repeated reasoning, duplicate tools, repeated reads, retries, fragmented edits, and command repetition
- **Session economics** using API token usage, cache statistics, pricing models, and waste-cost estimates
- **Intent construction** using weighted prompt embeddings rather than simple text repetition
- **Structured repetition scoring** using semantic, lexical, entity, action, and tool-call evidence
- **Tool fingerprints** based on normalized tool names and arguments
- **JSONL identity and merging** with source provenance and content fingerprints
- **Span segmentation** for finer-grained reasoning and generation analysis
- **Real-time monitoring** with rolling TER, drift detection, live warnings, and sustained-degradation policies
- **Context orchestration** with fragment storage, dependency graphs, budget optimization, and delta composition
- **Evaluation and regression tooling** for threshold calibration and release comparison
- **Adaptive optimization** that learns bounded project thresholds, phase weights, token budgets, and intervention settings from aggregate history
- **Repository memory** for local retrieval of similar code, prior fixes, duplicate patterns, and project lessons
- **Closed-loop interventions** with configurable observe/suggest/warn/block modes, context refresh, replanning, and outcome measurement
- **Feed-forward trend analysis** across lessons and intervention effectiveness
- **High automated test coverage** with branch coverage enforced in CI

Current verified release status:

```text
ruff format --check src tests: passed
ruff check src tests: passed
mypy src/: passed (83 source files)
python -m pytest: 1,122 passed
Configured branch-coverage minimum: 90%
```

---


## Scoring scope and JSONL compatibility

TER scores **model output only**: assistant reasoning, tool use, and generated
responses. User prompts remain available for intent construction, prompt-response
alignment, and input analysis, but are excluded from `total_tokens`,
`aligned_tokens`, `waste_tokens`, and phase TER scores.

Claude Code and SDK JSONL exports may contain metadata records such as
`queue-operation`, `last-prompt`, and `ai-title`. The loader ignores these
non-conversation records. Only supported conversation records are converted into
messages, and only messages whose role is `assistant` become scored spans. This
prevents duplicated queued prompts from being misclassified as generated output.

To verify a session after analysis:

```bash
ter analyze path/to/session.jsonl --format json | jq '{
  total_tokens,
  aligned_tokens,
  waste_tokens,
  roles: [.classified_spans[].source_role] | unique
}'
```

The `roles` output should contain only `"assistant"`. User-token totals are
reported separately under `input_analysis.token_breakdown`.

## Installation

### Base installation

```bash
python -m pip install -e .
```

### Development installation

```bash
python -m pip install -e ".[dev]"
```

### Embedding-enabled installation

Some TER features use sentence-transformer embeddings. If embeddings are optional in your installation configuration, install the corresponding extra:

```bash
python -m pip install -e ".[embeddings]"
```

For development with embedding features:

```bash
python -m pip install -e ".[dev,embeddings]"
```

TER requires Python 3.11 or a newer version explicitly supported by the project’s CI configuration.

---

## Quick Start

A deterministic synthetic session is included at:

```text
sample_sessions/example_session.jsonl
```

It contains no real user data, credentials, or proprietary source code.

### Analyze the sample session

```bash
ter analyze sample_sessions/example_session.jsonl
```

### Generate JSON output

```bash
ter analyze sample_sessions/example_session.jsonl --format json
```

### Generate a Markdown report

```bash
ter report sample_sessions/example_session.jsonl
```

Write the report to a file:

```bash
ter report sample_sessions/example_session.jsonl -o example-report.md
```

### Run cost-weighted and overthinking analysis

```bash
ter analyze sample_sessions/example_session.jsonl \
  --cost-weighted \
  --check-overthinking
```

### Store and optimize context fragments

```bash
ter context store sample_sessions/example_session.jsonl

ter context optimize \
  sample_sessions/example_session.jsonl \
  --budget 10000
```

---

## Phase 6 adaptive optimization

After recording at least a few sessions for a project, TER can learn a bounded operating policy from aggregate history:

```bash
ter optimize --project my-project
```

Export the policy for review, CI, or hook configuration:

```bash
ter optimize \
  --project my-project \
  --minimum-samples 5 \
  --output .ter/adaptive-policy.json \
  --format json
```

Optionally personalize token limits using the same privacy-preserving prompt fingerprints used by history prediction:

```bash
ter optimize \
  --project my-project \
  --prompt "refactor parser tests" \
  --neighbors 8
```

The generated policy includes similarity, confidence, and restatement thresholds; phase weights; soft/recommended/hard token budgets; and intervention thresholds. Recommendations are conservatively bounded and carry an `insufficient`, `experimental`, `stable`, or `mature` confidence label. Raw prompts and session content are never written to the policy. See [`PHASE6_CHANGES.md`](PHASE6_CHANGES.md).

---

## Sample Session Contents

The included synthetic session demonstrates:

- A user request with explicit requirements
- Assistant reasoning
- Assistant text output
- Tool calls and tool results
- A repeated file read
- A file write
- A verification command
- Input/output/cache token usage
- Stable session, request, message, and tool identifiers

The sample is intended for:

- README examples
- Manual experimentation
- CLI smoke tests
- Regression tests
- Demonstrations and screenshots

---

## Core Commands

### Analyze a session

```bash
ter analyze path/to/session.jsonl
```

Useful options:

```text
--format text|json
--similarity-threshold FLOAT
--confidence-threshold FLOAT
--restatement-threshold FLOAT
--phase-weights REASONING,TOOL_USE,GENERATION
--no-waste-patterns
--cost-model MODEL
--group
--no-input-analysis
--prompt-similarity-threshold FLOAT
--cost-weighted
--check-overthinking
```

Example:

```bash
ter analyze sample_sessions/example_session.jsonl \
  --format json \
  --cost-weighted
```

---

### Generate a Markdown report

```bash
ter report path/to/session.jsonl
```

Write to a file:

```bash
ter report path/to/session.jsonl -o report.md
```

The report can include:

- Aggregate TER
- Per-phase scores
- Waste percentage
- Token totals
- Cost estimates
- Cache efficiency
- Context growth
- Positional TER
- Waste patterns
- Suggested next steps
- Calibration and uncertainty metadata when available

---

### List sessions

```bash
ter list
```

Specify a location:

```bash
ter list ~/.claude/projects/
```

JSON output:

```bash
ter list ~/.claude/projects/ --format json
```

Limit results:

```bash
ter list ~/.claude/projects/ --limit 20
```

---

### Compare sessions

```bash
ter compare session-a.jsonl session-b.jsonl
```

Directories may also be supplied:

```bash
ter compare ~/.claude/projects/project-a ~/.claude/projects/project-b
```

Sort results:

```bash
ter compare sessions/ --sort ter
ter compare sessions/ --sort tokens
ter compare sessions/ --sort waste
```

Generate a before/after baseline comparison:

```bash
ter compare before.jsonl after.jsonl --baseline
```

---

## Live Monitoring

Monitor active Claude Code sessions in real time:

```bash
ter watch ~/.claude/projects/your-project
```

Watch a specific session file:

```bash
ter watch path/to/session.jsonl
```

Watch the most recently modified session:

```bash
ter watch ~/.claude/projects/your-project --latest
```

Use stream mode for logs or pipelines:

```bash
ter watch --stream ~/.claude/projects/your-project
```

Save monitoring signals:

```bash
ter watch ~/.claude/projects/your-project \
  --stream \
  --log ter-signals.jsonl
```

The live monitor can display:

- Rolling TER
- Reasoning, tool-use, and generation scores
- Output, aligned, and waste token totals
- Input and cache token statistics
- Cost and estimated waste cost
- Session duration
- Token throughput
- Context growth and bloat signals
- Recent TER trend
- Drift warnings

### Closed-loop Claude Code interventions

Index the repository before enabling project-aware guidance:

```bash
ter memory index .
```

Configure Claude Code hooks to invoke the unified handler for events such as
`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`PostToolUseFailure`, `PermissionRequest`, and assistant-stop events:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 15}]
    }],
    "PreToolUse": [{
      "hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 15}]
    }]
  }
}
```

Native Claude Code hook payloads do not contain TER metrics. When a payload
includes `transcript_path` (or `transcript`), `ter hook monitor` reads only the
newly appended JSONL bytes, updates rolling session counters persisted in
`HookSessionState`, and injects the resulting `ter_metrics` mapping before policy
evaluation. Transcript access and parsing failures are ignored so hooks remain
fast and non-blocking. Explicit `ter_metrics`, `metrics`, or `ter_signal` payloads
still take precedence for external integrations.

TER evaluates sustained degradation rather than reacting to a single noisy
measurement. Default policy thresholds are:

- refresh warning: TER drop of `0.12` with waste ratio at least `0.25`;
- replan: TER drop of `0.20` with waste ratio at least `0.40`;
- persistence: three degraded windows;
- refresh/replan cooldowns: 120/180 seconds.

Override them explicitly when needed:

```bash
ter hook monitor \
  --policy-mode warn \
  --ter-drop-warning 0.12 \
  --ter-drop-replan 0.20 \
  --waste-ratio-warning 0.25 \
  --waste-ratio-replan 0.40 \
  --degraded-windows-required 3
```

Policy modes are `observe`, `suggest`, `warn`, and `block`. In `observe` mode,
interventions are consumed, recorded, and evaluated silently: no guidance or
system message is injected. The other three modes surface guidance, while
blocking is intended for high-confidence conditions such as exact duplicate tool
calls. Metric dips normally produce context-refresh or replanning guidance.
Pending interventions are consumed once, and issued interventions retain baseline
snapshots for later compliance and effect evaluation. Effects are classified as
`improved`, `neutral`, `regressed`, `acknowledged_not_followed`, or `ignored`.

---


## Phase 4: Cross-Session Intelligence

TER v2.0.4 can build an **opt-in local history** of aggregate efficiency data.
The default database is `~/.claude/ter/history.db`. TER does not store raw
session content in this database; predictive matching uses a deterministic
hashed prompt fingerprint.

Record a completed session:

```bash
ter history record tests/fixtures/sample_session.jsonl --project TER
```

Review recent records and a project-level profile:

```bash
ter history list --project TER
ter history profile --project TER
```

Estimate likely efficiency before starting similar work:

```bash
ter history predict "add JSON export with tests" --project TER
```

Predictions remain marked experimental until a project has at least 50
recorded sessions. Use the cost dashboard for aggregate trends:

```bash
ter dashboard --project TER
```

Use `--db PATH` on any history or dashboard command to keep the database in a
project-specific or encrypted location. Recording is never automatic.

## Budget Recommendations

Estimate an appropriate token and model budget before starting a task:

```bash
ter budget "Fix the authentication bug in login.py"
```

Use prior history:

```bash
ter budget \
  "Implement an e-commerce checkout with Stripe" \
  --use-history
```

The recommendation can include:

- Complexity classification
- Model tier
- Thinking-token budget
- Estimated total tokens
- Estimated cost
- Historical adjustment

---

## Context Orchestrator

### Store fragments

```bash
ter context store path/to/session.jsonl
```

### Build a context graph

```bash
ter context graph path/to/session.jsonl
```

### Optimize context for a token budget

```bash
ter context optimize path/to/session.jsonl --budget 10000
```

Optional relevance threshold:

```bash
ter context optimize path/to/session.jsonl \
  --budget 10000 \
  --relevance-threshold 0.2
```

### Compose a delta prompt

```bash
ter context delta path/to/session.jsonl
```

### Check cross-session consistency

```bash
ter context check path/to/session.jsonl
```

Include subagents:

```bash
ter context check path/to/session.jsonl --group
```

Select consistency mode:

```bash
ter context check path/to/session.jsonl --mode strict
ter context check path/to/session.jsonl --mode relaxed
```

---

## Grouped Analysis

Analyze a parent session together with subagent sessions:

```bash
ter analyze path/to/session.jsonl --group
```

TER discovers subagent sessions from the supported filesystem layout and reports:

- Parent-session results
- Per-subagent results
- Token-weighted aggregate TER
- Aggregate costs
- Aggregate waste
- Cross-session comparisons

---

## Architecture

```text
src/ter_calculator/
├── __main__.py
├── cli.py
├── commands/
│   ├── analyze.py
│   ├── context.py
│   ├── hook.py
│   ├── listing.py
│   ├── report.py
│   └── watch.py
│
├── acceleration/
│   ├── __init__.py
│   ├── cache.py
│   ├── parallel.py
│   ├── quick_analyser.py
│   └── session_watcher.py
│
├── models.py
├── loader.py
├── jsonl_identity.py
├── span_segmentation.py
├── intent.py
├── intent_construction.py
├── intent_extraction.py
├── classifier.py
├── repetition_scoring.py
├── tool_fingerprints.py
├── compute.py
├── waste.py
├── waste_detectors.py
├── economics.py
├── cost_model.py
├── input_analysis.py
├── overthinking.py
├── real_time.py
├── adaptive_budget.py
├── formatter.py
├── formatter_json.py
├── formatter_rich.py
├── rich_components.py
├── dashboard.py
├── fragment_store.py
├── context_graph.py
├── budget_optimizer.py
├── delta_composer.py
├── consistency.py
├── embedding_cache.py
├── token_counting.py
├── validation.py
├── evaluation.py
├── regression.py
├── feedback.py
├── plugins.py
├── hook_monitor.py
├── intervention.py
├── intervention_policy.py
├── repository_memory.py
├── session_report.py
└── analyze_pipeline.py
```

### Main responsibilities

#### Parsing and identity

- `loader.py` parses session JSONL.
- `jsonl_identity.py` creates stable block identities and fingerprints.
- Sibling entries sharing a request ID can be merged while preserving distinct content.
- Source provenance is retained so merged spans can be traced to source records.
- Partial, duplicated, or malformed records are handled through validation and recovery logic.

#### Segmentation

- `span_segmentation.py` divides large reasoning or generation blocks into finer segments.
- Segmentation can use paragraph, sentence-group, Markdown, or discourse boundaries.
- Small adjacent segments may be merged to avoid unstable micro-spans.

#### Intent construction

- Prompts are embedded independently.
- Explicit weights can represent recency and information content.
- Weighted prompt embeddings are combined into a normalized intent centroid.
- Operational prompts such as “continue” can be down-weighted.
- Topic-shift handling can preserve more than one active intent representation.

#### Repetition analysis

Reasoning repetition can combine:

- Semantic similarity
- Lexical similarity
- Entity overlap
- Action overlap
- Parameter novelty
- Temporal distance

Tool repetition uses structured evidence such as:

- Tool name
- Normalized arguments
- File path
- Line range
- Query
- Command
- Exit state
- Result fingerprint

#### Evaluation and regression

- `evaluation.py` supports metric evaluation and threshold analysis.
- `regression.py` supports release-to-release comparisons.
- Thresholds should be calibrated against labeled data.
- Precision-weighted metrics such as F0.5 are appropriate when false-positive waste labels are especially costly.

---

## How TER Works

A typical analysis pipeline is:

1. **Load**  
   Parse JSONL records and recover valid session messages.

2. **Identify and merge**  
   Assign stable identities, merge sibling records, remove exact duplicate blocks, and preserve provenance.

3. **Segment**  
   Split reasoning, generation, and tool content into analysis spans.

4. **Construct intent**  
   Build one or more weighted intent vectors from user prompts.

5. **Classify**  
   Estimate whether each span is aligned, repetitive, or potentially wasteful.

6. **Score repetition**  
   Combine semantic, lexical, entity, action, and structured tool evidence.

7. **Compute TER**  
   Calculate per-phase efficiency and a weighted aggregate.

8. **Detect structural waste**  
   Detect repeated reads, fragmented edits, failed retries, repeated commands, and related patterns.

9. **Calculate economics**  
   Use API usage, model pricing, cache statistics, and output calibration.

10. **Evaluate confidence**  
    Track classifier confidence and low-confidence token share.

11. **Report**  
    Produce Rich terminal output, JSON, Markdown, monitoring signals, or comparison results.

12. **Optimize context**  
    Optionally store fragments, build dependency graphs, and select context within a token budget.

---

## JSONL Merge Semantics

A `requestId` can represent one logical API response that was serialized across several sibling JSONL lines.

TER therefore preserves distinct content blocks instead of keeping only the sibling with the highest `output_tokens`.

The parser should protect against:

- Exact duplicate content blocks
- Out-of-order sibling entries
- Reused request IDs
- Missing request IDs
- Partial writes
- Interrupted sessions
- Corrupt final lines
- Tool results arriving separately
- Multiple assistant messages sharing a request identifier

Stable content fingerprints and source-line provenance help make merging deterministic and auditable.

---

## Interpretation and Uncertainty

TER is a heuristic metric.

A score such as:

```text
TER: 0.83
```

does not imply perfect certainty. Classification depends on:

- Prompt interpretation
- Embedding behavior
- Similarity thresholds
- Segmentation choices
- Tool normalization
- Session completeness
- Model and tokenizer differences

Where supported, reports should include uncertainty information such as:

```text
TER estimate: 0.83
Bootstrap 95% interval: 0.77–0.88
Low-confidence tokens: 14.2%
```

TER is most useful for:

- Comparing similar sessions
- Detecting regressions
- Identifying repeated work
- Finding high-cost waste patterns
- Guiding investigation

It should not be used as the sole basis for judging a developer, model, or individual session.

---

## Empirical Validation

Functional correctness does not by itself prove that TER matches expert judgment.

A strong validation program should include:

- A labeled evaluation dataset
- Multiple human annotators
- Annotation guidelines
- Inter-rater agreement
- Precision and recall by waste category
- F1 and F0.5
- Precision-recall curves
- Confusion matrices
- Bootstrap confidence intervals
- Leave-one-session-out validation
- False-positive analysis
- Model and embedding sensitivity analysis
- Release-to-release regression benchmarks

Threshold changes should be justified by benchmark results rather than hand tuning alone.

---

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the full test suite:

```bash
python -m pytest
```

Run tests with branch coverage:

```bash
python -m pytest \
  --cov=ter_calculator \
  --cov-branch \
  --cov-report=term-missing
```

Current verified result:

```text
1,065 passed
91.96% branch coverage
```

Lint:

```bash
ruff check src/
```

Apply safe automatic fixes:

```bash
ruff check src/ --fix
```

Type check:

```bash
mypy src/
```

Recommended full local quality check:

```bash
python -m pytest && ruff check src/ && mypy src/
```

Run an individual test module:

```bash
python -m pytest tests/unit/test_loader.py -v
```

Run a single BDD scenario:

```bash
python -m pytest \
  tests/features/steps/performance_steps.py::test_sibling_entries_sharing_a_requestid_preserve_all_content_blocks \
  -vv
```

---

## Coverage Policy

Branch coverage is enabled.

The configured project floor is:

```text
90%
```

The current verified result is:

```text
91.96%
```

Coverage is most important in:

- Parsing and JSONL merging
- Identity and fingerprinting
- Classification
- TER computation
- Cost calculations
- Validation
- Real-time state management
- Context persistence
- Regression detection

Some presentation, platform-specific, and defensive fallback paths may reasonably remain below the repository average.

---

## Testing the README Example

A smoke test should verify that the included sample remains usable:

```python
def test_readme_sample_session_runs(cli_runner):
    result = cli_runner.invoke(
        ["analyze", "sample_sessions/example_session.jsonl"]
    )

    assert result.exit_code == 0
```

At minimum, CI should verify:

```bash
ter analyze sample_sessions/example_session.jsonl
ter analyze sample_sessions/example_session.jsonl --format json
ter report sample_sessions/example_session.jsonl -o /tmp/ter-example-report.md
```

This prevents documentation examples from becoming stale.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pytest_bdd'`

Install the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run pytest through the active interpreter:

```bash
python -m pytest
```

This avoids invoking a `pytest` executable from a different Python environment.

### `Session file not found`

Confirm that the path exists:

```bash
ls -l sample_sessions/example_session.jsonl
```

Run the included example from the repository root:

```bash
ter analyze sample_sessions/example_session.jsonl
```

### Embedding model unavailable

Install embedding dependencies:

```bash
python -m pip install -e ".[embeddings]"
```

If the model must be downloaded, ensure network access is available during first use or configure a local model path.

### Coverage unexpectedly drops

Check for stale or duplicate modules left behind after a package refactor. A file and package with the same import name can cause coverage to count unreachable source files.

Verify the imported module path:

```bash
python -c "import ter_calculator.acceleration as a; print(a.__file__)"
```

---

## Documentation

- [Architecture](docs/architecture.md)
- [Context Orchestrator](docs/context-orchestrator.md)
- [User Guide](docs/user-guide.md)
- [Phase 9: Repository Memory](PHASE9_CHANGES.md)
- [Phase 10: Closed-loop Project Intelligence](PHASE10_CHANGES.md)
- [Phase 11: Metric-driven Interventions](PHASE11_CHANGES.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [License](LICENSE)

---

## Contributing

Contributions are welcome.

Before submitting a pull request:

```bash
python -m pytest
ruff check src/
mypy src/
```

New behavior should include:

- Unit or integration tests
- Regression coverage for bug fixes
- Documentation updates where relevant
- No reduction below the configured coverage floor

---

## Limits of Interpretation

TER should be interpreted carefully:

- Token estimates may differ from provider billing totals.
- Embeddings are approximations of semantic intent.
- Similarity thresholds are model- and dataset-sensitive.
- Repetition is not always waste.
- Corrections and verification can be necessary even when text appears similar.
- Tool calls require structured comparison, not semantic similarity alone.
- Incomplete or corrupt logs reduce confidence.
- A high TER does not prove task success.
- A low TER does not prove poor engineering judgment.
- Waste detection (duplicate tool calls, retries, restated reasoning) assumes
  long-lived, multi-turn sessions. On bursty workloads made of many short,
  single-shot sessions, spans per session can be as low as one `thinking` +
  one `text` block, leaving nothing for these detectors to act on. TER on
  such sessions is not wrong, just largely uninformative — see
  `scripts/labeling_priority.py` for a way to find which sessions in a
  bursty corpus have enough structure to be worth hand-labeling.
- Semantic embeddings (`sentence-transformers`) and token estimation
  (`tiktoken`) both attempt a one-time network download on first use.
  Tiktoken now falls back to a character-based estimate if that download
  fails (see UPDATES.md); the embedding model download has no offline
  fallback yet, so first run on a firewalled or air-gapped machine will fail
  until the model is vendored or cached ahead of time.

Use TER alongside task outcomes, expert review, and session context.

---

## Requirements

Core requirements typically include:

- Python
- NumPy
- Rich
- Standard-library SQLite support

Embedding-enabled features may additionally require:

- sentence-transformers
- A compatible ML backend
- Access to a local or downloadable embedding model

See `pyproject.toml` for the authoritative dependency and Python-version declarations.

## Standalone HTML reports

TER can generate a portable, interactive HTML report for a session. The report embeds its CSS, JavaScript, charts, and analysis data, so it opens directly in a browser and does not require a web server or network access.

```bash
ter analyze sample_sessions/example_session.jsonl --format html
```

The default output is written beside the input file as:

```text
sample_sessions/example_session.ter-report.html
```

Choose an explicit destination with `--output`:

```bash
ter analyze session.jsonl \
  --format html \
  --output reports/session-report.html
```

The report includes:

- an executive scorecard for TER, aligned and waste tokens, cost, and reliability;
- token composition and phase distribution charts;
- a token-weighted span timeline;
- an alignment-versus-confidence scatter plot;
- an interactive span inspector with text and classification evidence;
- consistency diagnostics for low confidence, low alignment, and invalid source roles;
- an embedded JSON download for downstream analysis.

HTML reports score only assistant-origin spans. User prompts remain available for intent construction and input analysis but are excluded from TER output scoring.

## Phase 5 production hardening

TER v2.0.5 adds production-readiness diagnostics and durable history operations:

```bash
ter doctor --format json
ter history backup ~/.claude/ter/backups/history.db
ter history restore ~/.claude/ter/backups/history.db --force
```

Runtime settings can be controlled with `TER_DB_PATH`, `TER_LOG_LEVEL`,
`TER_BUSY_TIMEOUT_MS`, and `TER_BACKUP_RETENTION`. The history database uses
WAL journaling, a bounded busy timeout, schema-version tracking, integrity
checks, and restrictive POSIX permissions.

## Phase 7: CI/CD and ecosystem integrations

TER v2.0.7 can enforce portfolio quality gates and export artifacts for automation platforms:

```bash
ter integrate ter-results \
  --minimum-ter 0.80 \
  --maximum-waste-ratio 0.20 \
  --format sarif \
  --output ter-results.sarif
```

Use `--format github` for GitHub Actions annotations, `--format summary` for a step summary, or `--format json` for external dashboards and telemetry pipelines. A failed gate exits with status `2`, making the command suitable for CI enforcement. See `PHASE7_CHANGES.md` for details.

## Phase 8: reproducible release validation

Build a deterministic release manifest and enforce final quality gates:

```bash
ter release-check ter-results \
  --minimum-sessions 100 \
  --minimum-ter 0.90 \
  --maximum-waste-ratio 0.10
```

Compare a candidate against a prior manifest:

```bash
ter release-check ter-results \
  --baseline previous-release-manifest.json \
  --maximum-ter-drop 0.01 \
  --maximum-waste-increase 0.01
```

The manifest includes canonical aggregate metrics, distribution percentiles, a stable results fingerprint, and SHA-256 checksums for every input result file. See `PHASE8_CHANGES.md`.

## Repository memory and feed-forward intelligence (v2.0.9–v2.0.14)

Build a private project index and retrieve similar code, prior failures, fixes,
and duplicate patterns before coding:

```bash
ter memory index .
ter memory search "authentication retry loop"
ter memory inspect
```

The deterministic local index is stored at `.ter/memory-index.json` by default.
Results retain source paths, line ranges, excerpts, confidence scores, duplicate
flags, and defect/fix indicators. Claude Code hooks can automatically retrieve
this context on session start and prompt submission.

Session alerts and intervention results are persisted separately, by default, as:

```text
.ter/session-lessons.jsonl
.ter/intervention-outcomes.jsonl
```

Inspect recurring scenarios and measured intervention performance:

```bash
ter memory trends --minimum-occurrences 2
ter memory trends --format json
```

Trend output can include issuance counts, acknowledgement and compliance rates,
improvement rates, median TER change, and median waste-ratio change by
intervention type. Raw before/after measurements remain available so policy
thresholds can be recalibrated without losing evidence.

See [`PHASE9_CHANGES.md`](PHASE9_CHANGES.md),
[`PHASE10_CHANGES.md`](PHASE10_CHANGES.md), and
[`PHASE11_CHANGES.md`](PHASE11_CHANGES.md).
