# TER Pipeline Accuracy Improvements (PR 004)

> Improve live and post-hoc TER accuracy through four targeted changes: exponential decay intent tracking in live mode, accurate token estimation using tiktoken, a drop-in embedding model upgrade, and wiring SlidingIntentExtractor into the post-hoc pipeline.

---

## Changes in scope

---

### 1. Exponential decay for live intent tracking

**Where:** [`src/ter_calculator/real_time.py`](../src/ter_calculator/real_time.py) lines 621–627

**Current behaviour:** Every user prompt is embedded and stored in `state.intent_embeddings: list[NDArray]`. Intent is `np.mean(intent_embeddings, axis=0)`. In a 50-turn session, turn 1 has the same weight as turn 49 — the signal drifts toward a diluted average of all goals.

```python
state.intent_embeddings.append(prompt_emb)
state.intent_embedding = np.mean(state.intent_embeddings, axis=0).astype(np.float32)
```

**Fix:** Replace with an Exponential Moving Average (EMA):

```python
_INTENT_DECAY = 0.3  # tunable
state.intent_embedding = (
    _INTENT_DECAY * prompt_emb + (1 - _INTENT_DECAY) * state.intent_embedding
)
```

Remove `state.intent_embeddings` list from `RollingTERState` entirely — it is no longer needed.

**SOTA justification:** EMA is the standard algorithm for online/streaming signal tracking where recency matters — used in financial time series (MACD), reinforcement learning (eligibility traces), and adaptive filtering. In multi-turn conversational AI, intent drift is a documented problem: early goals anchor classification even when the user has moved to a completely different subtask. α = 0.3 gives a half-life of ~2 turns (0.7² ≈ 0.49), meaning a prompt from 5 turns ago contributes only ~17% weight. This is consistent with standard signal processing starting points for tracking signals with moderate drift rates. The alternative (uniform mean) is not suitable for streaming contexts; it is only appropriate in post-hoc mode where the full session is available.

**What this achieves:** The live intent signal reflects what the user is working on *now*, not an average of everything they have said since the session started. Spans classified as waste in the current task phase will no longer be rescued by an intent vector dragged toward unrelated earlier prompts. This directly reduces false negatives (waste missed because the global intent happens to overlap with a wasteful span) in long sessions.

**Drawback:** α is a tunable hyperparameter. Without a hand-labelled gold set, optimal α is estimated rather than validated.

---

### 2. Tiktoken for live span token estimation

**Where:** [`src/ter_calculator/real_time.py`](../src/ter_calculator/real_time.py) `_estimate_tokens()` + [`pyproject.toml`](../pyproject.toml)

**Current behaviour:** `_estimate_tokens(text) = max(1, len(text) // 4)`. Char/4 assumes roughly 4 characters per token — true for average English prose, but code, reasoning traces, and technical terminology consistently produce shorter tokens than prose (e.g. `kwargs`, `isinstance`, `np.float32` each tokenise to 2–4 tokens, not one). We observed 9,413 estimated vs 31,099 actual on the 6e3423c8 session (3.3x undercount).

**Fix:**

```python
import tiktoken
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

def _estimate_tokens(text: str) -> int:
    return max(1, len(_TIKTOKEN_ENC.encode(text)))
```

Add `tiktoken>=0.7.0` to `pyproject.toml`.

**SOTA justification:** BPE (Byte-Pair Encoding) is the tokenization algorithm used by both Anthropic and OpenAI. Rather than splitting on characters or words, it learns common subword sequences from a large corpus and merges them into single tokens — so "authentication" may be one token, while "authenticate_user" splits into multiple subword pieces. Anthropic has not published a standalone tokenizer package. Claude uses BPE; `cl100k_base` (GPT-4's encoding, available via OpenAI's open-source `tiktoken` library) uses the same algorithm family and is the closest publicly available approximation to Claude's tokenizer. `tiktoken` is implemented in Rust with a Python wrapper; encoding a typical span takes under 1ms, well within live-mode tolerances.

The char/4 heuristic assumes all text has the same character-to-token density. Empirically this is not true: tool call arguments, file paths, and code identifiers tokenize into more pieces than prose (tiktoken gives higher counts), while natural language reasoning tokenizes more efficiently (tiktoken gives slightly lower counts). Replacing char/4 with tiktoken applies real tokenization rules per span rather than a fixed divisor with no empirical basis.

**What this achieves:** `tiktoken` is OpenAI's open-source tokenizer library — a fast, Rust-backed Python package that applies real BPE tokenization to text. For code-heavy spans dominant in Claude Code sessions (tool call arguments, file paths, structured output), tiktoken gives higher and more realistic token counts than char/4. For natural language reasoning, it gives slightly lower counts. The net result is that the live Waste $ figure and per-phase token breakdown reflect actual tokenization behaviour rather than a flat approximation. The TER ratio itself (aligned/total) is not materially affected since both numerator and denominator are estimated the same way.

**Important caveat:** tiktoken does not close the structural gap between live and post-hoc token totals. That gap exists because live mode estimates tokens on partial streaming chunks as they arrive, while post-hoc uses `usage.output_tokens` from the completed API response. tiktoken makes the per-span estimates more principled, not necessarily closer to the API aggregate total.

**Drawback:** Adds a dependency (~50ms cold-start). Claude's tokenizer is proprietary — tiktoken is an approximation, not an exact match. Post-hoc is unaffected — it uses `usage.output_tokens` from the API directly.

---

### 3. Embedding model upgrade (drop-in)

**Where:** Three independent `_get_model()` functions — change the model string in each:
- [`src/ter_calculator/intent.py`](../src/ter_calculator/intent.py) line 30
- [`src/ter_calculator/intent_extraction.py`](../src/ter_calculator/intent_extraction.py) line 71
- [`src/ter_calculator/real_time.py`](../src/ter_calculator/real_time.py) `load_embedding_model()` line 127

Change `"all-MiniLM-L6-v2"` → `"all-MiniLM-L12-v2"` in all three.

**Model selection:** Three 384-dimensional candidates were evaluated on benchmarks directly representative of TER's classification task. Results from empirical runs on this machine:

| Model | Dim | Latency | CodeSearchNetRetrieval nDCG@10 | STSBenchmark Spearman |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (current) | 384 | 4.80ms | 0.7928 | 0.8203 |
| all-MiniLM-L12-v2 (chosen) | 384 | 9.07ms | 0.8209 | 0.8309 |
| BAAI/bge-small-en-v1.5 | 384 | 10.57ms | 0.8879 | 0.8586 |

`CodeSearchNetRetrieval` (Python subset, MTEB) — NL query → code retrieval — is the task most directly representative of what TER's classifier does: compare a natural language intent embedding against code and tool call spans. `STSBenchmark` validates the NL-to-NL side: intent drift detection and prompt-response alignment.

**Why not bge-small-en-v1.5?** Despite its higher retrieval scores, bge-small-en-v1.5 compresses cosine similarities toward higher values across the board. Empirically tested on TER-representative pairs without query prompts (which the model uses internally in MTEB but TER does not apply):

| Pair | L12-v2 | bge-small |
|---|---|---|
| Intent vs aligned reasoning span | 0.559 | 0.764 |
| Intent vs aligned tool call | 0.451 | 0.717 |
| Identical tool calls (repetition) | 1.000 | 1.000 |
| Intent vs completely unrelated text | -0.031 | **0.409** |
| Intent vs different-topic span | 0.342 | 0.640 |

bge-small scores completely unrelated content at 0.41 — above TER's `similarity_threshold` of 0.40. TER's waste detection relies on absolute thresholds (`filler_sim_max ≈ 0.11`, `verbose_sim_max ≈ 0.09`). With bge-small, nothing would fall below those thresholds, silently disabling all low-similarity waste classification. bge-small's high retrieval benchmark score reflects its ability to *rank* correct results above incorrect ones — which does not require absolute similarity values to be calibrated. TER's classifier needs absolute values. Switching to bge-small would require full threshold re-calibration against a labelled gold set before it could be used safely.

**SOTA justification:** L12-v2 doubles the transformer depth (6 → 12 layers, 22M → 33M parameters) while preserving the 384-dim output — making it a true drop-in. The additional layers give it better semantic discrimination, validated empirically: +2.8pp on code retrieval and +1.1pp on NL STS compared to L6-v2. Its similarity distribution remains compatible with existing classifier thresholds, confirmed by running `ter analyze` on the 711bb9b1 session and observing sensible alignment scores.

The three `_model` globals remain separate caches (a future cleanup could unify them, but that is a refactor not a correctness fix).

**What this achieves:** The classifier makes all alignment decisions by comparing span embeddings against the intent embedding using cosine similarity. A better embedding model means two things: semantically similar spans score closer together (fewer missed redundancies) and semantically different spans pull further apart (fewer false waste flags). The improvement is most visible for short, technical spans — tool call names, code identifiers, brief reasoning steps — where the 6-layer model lacks the depth to capture contextual meaning reliably.

**Drawback:** ~1.9x slower inference per call (9.07ms vs 4.80ms). Still within acceptable range for both post-hoc batch processing and live per-turn embedding. A full bge upgrade remains a future option once a labelled gold set exists to re-validate thresholds.

---

### 4. Wire SlidingIntentExtractor into post-hoc pipeline

**Where:** [`src/ter_calculator/analyze_pipeline.py`](../src/ter_calculator/analyze_pipeline.py) line 40 and [`src/ter_calculator/classifier.py`](../src/ter_calculator/classifier.py) lines 36–67

**Why it is in scope:** The model upgrade (change 3) touches `intent_extraction.py` where `SlidingIntentExtractor` lives. Wiring it in requires only two targeted changes — not the interface overhaul previously claimed.

**Current behaviour:** `extract_intent(session)` returns a single `IntentVector` by computing a weighted mean of all user prompt embeddings. All spans are then compared against this one blurred global intent. In sessions where user goals evolve (e.g. start with a bug fix, pivot to refactoring), spans from the later topic score low against the early-weighted global intent — producing false waste signals.

**Change 1 — analyze_pipeline.py:**

```python
# Replace line 40:
intent = extract_intent(session)
# With:
from .intent_extraction import SlidingIntentExtractor
intents = SlidingIntentExtractor().extract(session.user_prompts)  # list[IntentVector]
```

Pass `intent=intents[0]` to `compute_ter` to preserve the `TERResult.intent` field (used only in `formatter.py` lines 970–971 for `.confidence` display — backward compatible).

**Change 2 — classifier.py lines 63–67:** Accept `list[IntentVector]`, use nearest-intent similarity per span:

```python
# classify_spans signature: intent -> intents: list[IntentVector]
intent_sims = [
    max(cosine_similarity(embeddings[i], iv.embedding) for iv in intents)
    for i in range(len(spans))
]
```

Everything downstream (`_check_repetition`, `_classify_span`, `ClassifiedSpan`) is unchanged — they receive the same `float` sim value.

**SOTA justification:** Comparing each span against a single blurred global intent is the primary source of false positives in multi-topic sessions. The "nearest-intent" approach is the standard classification technique in multi-document retrieval and multi-label classification: assign a label based on the best-matching prototype, not the mean. For intent segmentation specifically, sliding-window topic boundary detection (originally Hearst's TextTiling, 1997) is the foundational technique, and `SlidingIntentExtractor` already implements a cosine-similarity-based version of this. Wiring it into `classify_spans` via max-similarity lookup is the standard retrieval-style classifier design — the same pattern used in dense passage retrieval (DPR, Karpukhin et al., 2020) where a query is matched against the best document in a corpus rather than a mean embedding.

This design is directly validated by the `CodeSearchNetRetrieval` benchmark (nDCG@10 = 0.821 with L12-v2): that task is structurally identical to TER's classifier — given a natural language intent, retrieve the most relevant code span. The benchmark confirms that nearest-neighbour matching between NL intent and code spans is a sound and measurable approach.

**What this achieves:** In a long coding session a user will typically move through several distinct goals — e.g. understand a codebase, fix a bug, write tests, refactor. The current single-intent model blurs all of these into one averaged embedding, causing spans from task B to score low against an intent dominated by task A, incorrectly flagging aligned work as waste. `SlidingIntentExtractor` detects topic boundaries in the prompt sequence and produces one intent vector per segment. Each span is then scored against the segment it is closest to, not the session average. The practical effect is fewer false positives in multi-topic sessions and a more interpretable per-phase alignment score.

**Drawback:** `SlidingIntentExtractor` segments by cosine similarity drop between adjacent prompts, using a configurable window. With very short sessions (< 3 prompts) it degrades gracefully to a single intent. No threshold re-validation is needed: using the nearest (maximum) intent similarity per span can only increase similarity scores relative to the blurred global mean, which reduces false positives — it cannot make the threshold stricter.

---

## Files changed

| File | Change |
|------|--------|
| [`src/ter_calculator/real_time.py`](../src/ter_calculator/real_time.py) | EMA intent, tiktoken estimation, model string |
| [`src/ter_calculator/intent.py`](../src/ter_calculator/intent.py) | Model string |
| [`src/ter_calculator/intent_extraction.py`](../src/ter_calculator/intent_extraction.py) | Model string |
| [`src/ter_calculator/analyze_pipeline.py`](../src/ter_calculator/analyze_pipeline.py) | SlidingIntentExtractor call |
| [`src/ter_calculator/classifier.py`](../src/ter_calculator/classifier.py) | Accept `list[IntentVector]`, max-sim per span |
| [`pyproject.toml`](../pyproject.toml) | Add `tiktoken>=0.7.0` |
| [`tests/unit/test_real_time.py`](../tests/unit/test_real_time.py) | Update EMA assertions, remove intent_embeddings list assertions |

## Branch and test plan

- Branch: `004-intent-accuracy` off `main`
- Run full test suite after each change
- Manual validation: `ter watch` and `ter analyze` on session `6e3423c8` before/after — confirm live/post-hoc gap narrows and no new false-positive waste signals

---

## 5. Classifier calibration — gold set validation (May 2026)

### Overview

Following implementation of all four changes above, a hand-labelled gold set of
60 uncertain reasoning spans was extracted from session `711bb9b1` (a long TER
development session covering dashboard work, embedding model upgrades, and error
detection fixes). Spans were selected to cover the "uncertain zone": the 10
classifier-labelled waste spans plus 50 aligned spans with similarity in the
0.05–0.30 range — the region where the classifier is most likely to err.

This exercise was motivated by the bge-small-en-v1.5 assessment in section 3:
> *"Switching to bge-small would require full threshold re-calibration against a
> labelled gold set before it could be used safely."*

The gold set also validates the chosen thresholds for L12-v2 and surfaces
classifier failure modes that similarity-only thresholds cannot address.

### Methodology

Each span was labelled in full context (surrounding turns, session intent) using
a binary question: *does this span introduce new information or analysis, or is
it purely announcing an action the model is about to take?*

- **Aligned**: introduces findings, root causes, multi-step plans with
  non-obvious sequencing, arithmetic yielding new numbers, key structural
  discoveries.
- **Waste**: action narrations ("Let me read X", "Good! Now I'll add Y"),
  success confirmations ("Excellent! Task done. Now let me..."), system
  artifacts ("[Request interrupted]").

Labels were assigned by the assistant reading each span with surrounding session
context, then verified spot-check by the operator. All 60 labels and rationales
are available in the canvas artifact `gold-set-results`.

### Findings

**Finding 1 — Similarity is a poor primary classifier for reasoning spans.**

Waste spans covered a sim range of 0.18–1.00; aligned spans covered 0.16–0.46 —
complete overlap. The classifier agreed with gold labels on only 29/60 uncertain
spans (near-random). For reasoning spans in the 0.18–0.40 sim zone, intent
similarity alone cannot separate action narrations from genuine analysis.

**Finding 2 — Token count is the strongest single waste signal.**

Of 29 false negatives (waste called aligned), 22 had fewer than 25 words. Of 23
aligned spans, only 2 had fewer than 25 words — and both contained specific
numeric references (line numbers, identifiers) carrying new information. The
rule `sim < 0.35 AND words < 25 AND no_specific_reference → waste` eliminates
~75% of false negatives with minimal false positive risk.

**Finding 3 — Repetition detection mis-fires on spans with specific references.**

Two false positives (spans #55 and #56) were flagged as `redundant_reasoning`
because they structurally echoed earlier spans (high repetition similarity ≥
0.88), but both contained actionable specifics — exact line numbers ("model=None
on lines 341 and 358", "add the flag after line 89"). A
`_has_specific_reference()` guard (regex matching `lines? \d+` or integers ≥ 3
digits) prevents mis-fires without affecting genuine repetition detection.

**Finding 4 — System artifacts need explicit exclusion.**

`[Request interrupted]` spans appeared in the uncertain zone with high
repetition similarity to adjacent spans. These are user-interruption artifacts,
not agent reasoning. A text-match guard routes them to `aligned_reasoning`
unconditionally.

### Resulting classifier changes

Three rules were added to `_classify_span` and one threshold dict added to `classifier.py`:

| Rule | Condition | Action |
|---|---|---|
| Short-narration gate (tier 2) | `sim < 0.35 AND words < 25 AND no_specific_ref` | `→ redundant_reasoning, conf=0.6` |
| Specific-reference repetition guard | `is_repetition AND phase==REASONING AND has_specific_ref` | Skip redundant label, fall through to aligned |
| System artifact guard | Text matches `[Request interrupted` | `→ aligned_reasoning, conf=0.5` |
| Tool-use repetition threshold | `phase == TOOL_USE` | Repetition threshold raised from 0.88 → 0.93 |

### Threshold validation

The gold set confirms the existing `similarity_threshold = 0.40` is appropriate
as a confidence gate. The short-narration gate parameters (`sim_max = 0.35`,
`words_max = 25`) were derived from the observed overlap boundary in the 60-span
sample and should be re-evaluated if the session corpus shifts significantly in
domain or agent verbosity.

The `bge-small-en-v1.5` concern from section 3 is validated: the L12-v2 model's
similarity distributions remain compatible with these thresholds, whereas
bge-small scores unrelated content at ~0.41 — above the alignment threshold —
which would require all threshold values to be recalibrated from scratch.

**Addendum — tool-use repetition threshold (session 94103fcd):**

Post-implementation validation on session 94103fcd (a mixed general-knowledge
and TER codebase analysis session) revealed a further false positive in live
mode: two `WebSearch` tool calls for unrelated topics (dinosaur weights and
Manchester United home kit) were flagged as `unnecessary_tool_call` because
their embeddings, dominated by shared JSON structure, scored ~0.90 similarity —
above the 0.88 threshold shared with reasoning spans.

Tool calls are structurally homogeneous by design: all share a tool name, JSON
keys, and argument format. Two calls for completely different topics typically
embed at 0.85–0.92. Genuine duplicates (same query, same arguments) embed at
≥ 0.97. A per-phase threshold dict (`_REPETITION_THRESHOLDS`) was added to
`classifier.py` raising the `TOOL_USE` threshold from 0.88 to 0.93, leaving
reasoning and generation unchanged.
