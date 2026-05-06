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
