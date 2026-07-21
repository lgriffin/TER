# TER 2.0.0 — Public release

- Promoted the internally validated v16 feature set to semantic version `2.0.0`.
- Standardized package, CLI, documentation, and release metadata.
- Cleaned repository and distribution hygiene.
- Preserved standalone HTML reports and assistant-only TER scoring.
- Added public release, security, and build-validation documentation.

# Unreleased — Offline fallback fix + labeling-priority triage tool

- **Fixed:** `embedding_cache.py::estimate_tokens` caught only local exception
  types (`ImportError`, `AttributeError`, `LookupError`, `UnicodeError`,
  `ValueError`, `RuntimeError`) around the tiktoken call. Network errors from
  tiktoken's first-run download of `cl100k_base.tiktoken`
  (`requests.exceptions.HTTPError`, `ConnectionError`, timeouts) were not
  caught, so the documented character-count fallback never actually fired on
  firewalled or offline machines — the whole analysis crashed instead of
  degrading gracefully. Now catches broadly (`except Exception`) since the
  fallback is safe regardless of *why* tiktoken failed.
- **Known, not yet fixed:** `sentence-transformers` still needs a one-time
  network download of `all-MiniLM-L12-v2` from `huggingface.co` with no
  offline/vendored path. Same class of risk as the tiktoken issue; worth
  vendoring the model or documenting as a hard deployment requirement before
  this runs anywhere firewalled.
- **Added:** `scripts/labeling_priority.py` — runs the classifier over a
  directory of session files across a small grid of `similarity_threshold` /
  `confidence_threshold` combinations and ranks sessions by how much their
  waste ratio swings (`spread`). High spread = sitting on the classifier's
  decision boundary = highest-value session to hand-label first when building
  a real gold set. Does not judge correctness, only threshold-sensitivity —
  it's a triage step for limited annotation time, not a substitute for
  annotation.
- **Finding (not a code change):** running this against ~145 real sessions
  from a long-running but *bursty* (single-shot, not long-lived multi-turn)
  workload showed 141/143 sessions collapsing to exactly 2 spans (one
  `thinking` block, one `text` block). Confirmed against raw JSONL that this
  is correct parsing, not a bug — those sessions genuinely have no tool use,
  no retries, nothing for the waste detectors to act on. This is a real
  scope gap: the classifier's waste signals (duplicate tool calls, retries,
  restated reasoning) assume long-lived, multi-tool sessions and currently
  have nothing to say about bursty single-shot workloads. See
  `ter_calculator_improvement_roadmap_v11.md` § 12 for the follow-on
  implication for threshold calibration.

# v0.9.0 — Fine-grained span segmentation

- Added opt-in semantic segmentation for reasoning and generated-response blocks.
- Added paragraph, heading, sentence-group, and discourse-transition boundaries.
- Added configurable minimum and maximum segment token targets.
- Preserved parent block identity and exact character offsets on every segment.
- Kept tool calls and tool results atomic for structured fingerprinting.
- Preserved legacy block-level behavior unless `--fine-segmentation` is enabled.
- Added focused segmentation, provenance, compatibility, and atomic-tool tests.

# v0.4.0


## v0.4.1 — Static-analysis stabilization

- Cleared all source Ruff and mypy findings.
- Added a pragmatic test-only Ruff policy while retaining strict source linting.
- Restored acceleration compatibility exports used by downstream tests.
- Added explicit typing for plugin registries, Rich renderables, optional APIs, cache values, command records, and formatter loops.
- Added `docs/007-v041-stabilization.md`.

- Made semantic embeddings an optional `embeddings` extra.
- Added an optional `llm` extra for Anthropic-assisted intent extraction.
- Added development and CI constraints files.
- Declared Python support as 3.11–3.13 and added a CI matrix.
- Enabled branch coverage with a 90% regression floor and XML artifact.
- Narrowed internal token-estimation and cache-deserialization exception handling.
- Added Phase 2 packaging documentation.

# v0.3.0 — Module Split

- Extracted CLI command implementations into `ter_calculator.commands`.
- Converted `ter_calculator.acceleration` into a focused package.
- Preserved historical imports through compatibility facades.
- Added architecture and compatibility regression tests.

# TER Calculator — update log and design notes

This document records a focused pass on **measurement accuracy** (especially dollars and input vs output context) and **classifier ergonomics**. It is meant for maintainers and for anyone calibrating the tool against real Claude Code sessions.

---

## What changed (summary)

| Area | Change |
|------|--------|
| **Spans** | Each `TokenSpan` now has `source_role` (`assistant` \| `user`), set when segmenting the JSONL. |
| **Economics** | `estimated_waste_cost_usd` counts **assistant-origin** waste only and **scales** to API `output_tokens` when usage data exists. New field: `waste_output_calibration_ratio`. |
| **Waste breakdown $** | Rows are tagged **output-priced** vs **input-priced**; totals use the right rate from `CostModel`. JSON includes `pricing` per source. |
| **Double-counting (breakdown table)** | Pattern rows for `duplicate_tool_call` and `context_restatement` are skipped when the same bucket already appears from classified spans (same idea as `reasoning_loop` ↔ redundant reasoning). See § below. |
| **Classifier** | `--confidence-threshold` now gates **self-repetition** waste; `--similarity-threshold` shapes filler / verbose thresholds (bounded so defaults stay near legacy behavior). |
| **Repo hygiene** | `.hf_cache/` added to `.gitignore` for local Hugging Face cache dirs. |

---

## Thought process

### 1. Why `source_role` on spans?

Heuristic span tokens are built from **both** assistant blocks (thinking, text, tool_use) and user blocks (notably **tool_result**). API **`output_tokens`** only reflect what the **assistant** generated. Mixing user-side spans into “output waste $” overstated or misattributed cost, and made calibration meaningless.

**Decision:** Tag spans at load time and use `assistant` vs `user` anywhere we tie waste to **billed output** or to **input-side context cost**.

### 2. Why calibrate waste $ to `output_tokens`?

Span `token_count` uses `len(text) // 4`, which rarely matches Anthropic’s tokenizer. The **ratio** of waste to total is still useful for TER; the **dollar** line should track what you actually pay for generation when usage is present.

**Decision:**  
`calibration_ratio = billed_output_tokens / sum(assistant span tokens)`  
`estimated_waste_cost_usd = assistant_waste_tokens × ratio × output_rate / 1e6`

Expose `waste_output_calibration_ratio` so JSON/UI consumers can see when the heuristic is tight or loose (ratios far from 1.0 deserve a glance).

### 3. Why split output vs input pricing in the waste breakdown?

Some detectors measure waste that shows up as **re-injected context** (e.g. repeated Read results, bash stderr/stdout, failed tool results). Pricing that at **output** $/MTok was systematically wrong vs Sonnet-style **input** $/MTok.

**Decision:** Classified assistant waste and most “behavioral” patterns stay **output**-priced; tool-result-heavy patterns use **input**-priced rows. The headline “Waste $” from `_compute_waste_cost` sums both with the appropriate rates and applies output calibration only to output-priced rows.

### 4. Why extend pattern overlap for duplicates / restatement?

If the classifier already counts tokens under “Unnecessary Tool Calls” or “Over-Explanation”, adding the same mass again from **duplicate_tool_call** or **context_restatement** patterns **double-counted** in the breakdown table and inflated “Waste $”.

**Decision:** Mirror the existing `reasoning_loop` ↔ “Redundant Reasoning” overlap for those pattern types when the corresponding category already has tokens from classification.

#### What we fixed in practice (double-counting and related mistakes)

Before this pass, several issues compounded:

1. **Same waste counted twice in the UI/table** — Classifier buckets (e.g. unnecessary tool calls, over-explanation) already attributed tokens to waste. Structural patterns (`duplicate_tool_call`, `context_restatement`) could **add the same phenomenon again** as separate rows. Totals looked worse than the underlying session because **two mechanisms described one failure mode**.

2. **Wrong pricing for some of that mass** — Even after fixing (1), **re-injected context** (tool results, repeated reads) was easy to price like **output**; it is often billed **input**-side. So dollars could be wrong even when token *labels* were right.

3. **“Output waste $” mixed assistant and user-origin spans** — Heuristic spans include **user** `tool_result` text. Billed **`output_tokens`** are **assistant-only**. Mixing roles misattributed cost and broke **calibration** to the API.

The **overlap rules** in `formatter._build_waste_breakdown` address **(1)** directly. **`source_role`**, **input- vs output-priced rows**, and **calibration to `output_tokens`** address **(2)** and **(3)**. Together, the breakdown and headline waste dollar line should **not** inflate the same work twice, and **should** separate invoice-relevant output waste from context-heavy input pricing where the model says so.

### 5. Why wire CLI thresholds into the classifier?

`--similarity-threshold` and `--confidence-threshold` were documented but not fully driving behavior; tuning sessions required changing code.

**Decision:**  
- **Confidence:** only treat high self-similarity as repetition-waste if `repetition_similarity >= confidence_threshold` (reduces borderline false “duplicate work”).  
- **Similarity:** map `similarity_threshold` into bounded bands for filler reasoning and long low-intent generation, preserving roughly the old defaults when threshold ≈ 0.40.

---

## How to verify locally

```bash
cd TER  # project root containing pyproject.toml
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Optional: keep HF cache inside the repo (sandbox-friendly)
export HF_HOME="$(pwd)/.hf_cache"

pytest tests/unit/test_economics.py tests/unit/test_classifier.py tests/unit/test_formatter.py \
  tests/unit/test_waste.py tests/unit/test_loader.py tests/unit/test_compute.py -q

ter analyze tests/fixtures/sample_session.jsonl --format json
```

**Note:** Full integration tests and anything that loads `sentence-transformers` need a **writable** Hugging Face cache (or `HF_HOME` pointing inside the repo). Failures from `PermissionError` on `~/.cache/huggingface` are environment-related, not logic errors in the tests themselves.

---

## Before / after (illustrative)

**Scenario:** A few tokens flagged as waste live on a **user** `tool_result` block (edge case in fixtures).

| | Before | After |
|---|--------|--------|
| That row in the waste table | Priced at **output** rate | Priced at **input** rate when appropriate |
| `estimated_waste_cost_usd` | Could mix user + assistant heuristic waste at output $ | **Assistant output waste only**, scaled to **`output_tokens`** when available |

When assistant waste exists and heuristic totals differ from `output_tokens`, **dollar estimates move toward invoice-reality** via `waste_output_calibration_ratio`.

---

## Files touched (reference)

- `src/ter_calculator/models.py` — `TokenSpan.source_role`, `SessionEconomics.waste_output_calibration_ratio`
- `src/ter_calculator/loader.py` — set `source_role` when building spans
- `src/ter_calculator/economics.py` — calibration + assistant-only waste cost helpers
- `src/ter_calculator/classifier.py` — confidence / similarity wiring
- `src/ter_calculator/formatter.py` — breakdown pricing kinds, overlap rules, JSON fields
- `src/ter_calculator/analyze_pipeline.py`, `config_parse.py` — shared `analyze` path for CLI consistency
- `src/ter_calculator/session_report.py`, `cli.py` — `ter report`, `ter compare --baseline`, **`-o`** on report
- `tests/unit/test_economics.py`, `test_classifier.py`, `test_session_report.py` — extended coverage
- `.gitignore` — `.hf_cache/`

---

## Future ideas (prioritized)

1. ~~**`ter report`**~~ — **Done (initial):** `ter report <session.jsonl>` emits Markdown; **`-o FILE`** writes `report.md` (or any path) instead of stdout. Optional: grouped run + richer `CLAUDE.md` bullets.

2. **“Lite” mode without embeddings** — Usage + structural detectors only for CI or live hooks; full TER when analyzing offline. Cuts cold-start and flakiness from model download.

3. **Gold set + metrics** — 20–50 hand-labeled spans/snippet pairs; report precision/recall per detector; tune thresholds with data instead of intuition.

4. **Per-phase calibration** — If logs ever expose reasoning vs generation usage separately, split calibration (today only aggregate `output_tokens` exists).

5. **Hooks / live nudge** — Cursor hook every N turns or after estimated $\Delta$: append a short hint file or stderr line (“same file Read 3×”). Pairs with lite mode.

6. **`ter compare --baseline`** — **Done (initial):** two session files → Markdown delta. **Open:** whether to expose full `analyze` flags on compare (today: default thresholds only); TBD.

7. **Export for RL / preferences** — `(turn_index, features, label, cost_proxy)` parquet or JSONL for DPO / offline RL on tool choice (bash vs Read) using existing anti-pattern signal as a reward channel.

8. **Documentation** — Link this file from the main `README.md` in a single sentence when you want discoverability without duplicating content.

---

## Maintenance

When adding a new **waste pattern**:

1. Decide if its `tokens_wasted` is mainly **assistant generation / tool JSON** (output-priced) or **tool results / context** (input-priced) and extend `_pattern_pricing` in `formatter.py` if needed.

2. If the pattern **duplicates** classified categories, add an entry to `pattern_overlap` in `_build_waste_breakdown` so tables and totals stay consistent.

---


*Last updated: 2026-04-17*

## v0.5.0 — Phase 3 benchmark foundation

- Added a validated JSONL benchmark schema for labeled TER units.
- Added binary and multiclass classifier evaluation.
- Added token-weighted precision, recall, F1, and F0.5 metrics.
- Added confusion matrices and deterministic bootstrap confidence intervals.
- Added conservative F-beta threshold calibration with optional minimum precision.
- Added the `ter benchmark` command and JSON/text reports.
- Added annotation guidance and an example benchmark dataset.
- Production scoring thresholds remain unchanged pending a real frozen benchmark.

## v0.6.0 — Structured tool-call repetition

- Added deterministic structured fingerprints for tool calls.
- Normalized paths, line ranges, queries, commands, nested arguments, and volatile metadata.
- Exact repeated tool actions are now distinguished from legitimate parameter changes.
- Static `TokenSpan` objects preserve tool name and structured input.
- Added adversarial regression tests and a synthetic benchmark fixture.
- Left production intent and non-tool repetition thresholds unchanged.

## v0.7.0 — Blended repetition scoring

- Added explainable semantic, lexical, entity, action, and parameter-novelty signals.
- Integrated blended scoring into reasoning, generation, and structured tool repetition checks.
- Exact tool duplicates remain strong waste evidence.
- Changed paths, ranges, queries, commands, and newly introduced entities reduce repetition scores.
- Retained existing production thresholds and semantic fallback behavior.

## 0.8.0

- Replaced repeated-text intent weighting with independently embedded weighted centroids.
- Added low-information operational prompt filtering.
- Added correction and constraint weighting.
- Added semantic topic-shift detection and per-topic intent extraction.
- Preserved the legacy intent API and prompt provenance.

## 0.10.1

- Fixed a mypy regression in JSONL block annotation by explicitly typing the dict-or-string temporary value.
- Preserved all v10 runtime behavior and compatibility contracts.

## 0.10.0

- Added stable normalized SHA-256 identities for JSONL content blocks.
- Deduplicated exact sibling blocks without losing contributing source lines.
- Retained distinct sibling blocks deterministically and surfaced merge warnings.
- Added source-line and fingerprint provenance to blocks and spans.
- Preserved request-ID and constructor compatibility.

## v0.11.0 — Explainability and uncertainty

- Added per-span classification explanations with reason codes, contributing
  signals, thresholds, and strongest matched-prior evidence.
- Added deterministic session-level uncertainty with token-weighted confidence,
  low-confidence token share, reliability labels, and 95% span-bootstrap TER
  intervals.
- Exposed explanations and uncertainty in JSON output and compact evidence in
  text reports.
- Kept production classification thresholds unchanged.

## v0.12.0

- Added `ter benchmark-compare` for baseline-versus-candidate evaluation.
- Added configurable precision-first release quality gates and CI exit codes.
- Added text and JSON regression reports with record-level, token-weighted, and error-count deltas.
- Added a paired example candidate benchmark and focused regression tests.
- Production classifier thresholds remain unchanged.

## v16 — Standalone HTML analysis reports

- Added `ter analyze SESSION --format html [--output FILE]`.
- Added a portable, dependency-free report with executive metrics, token and phase charts, span timeline, alignment/confidence scatter plot, diagnostics, interactive span inspection, and JSON export.
- Added safe embedded-JSON handling and HTML renderer regression tests.
- Preserved v15 assistant-only scoring behavior.
- Used internal development version `0.16.0` and documented the workflow in README.
