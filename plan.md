# TER: From Post-Hoc Analysis to Real-Time Capability Adaptation

**Date**: 2026-05-13 | **Status**: Active  
**Goal**: Evolve TER from a session-replay analyzer into a real-time efficiency signal that drives adaptive model behavior, token budgeting, and cost optimization.

---

## Vision

Today TER answers: *"How efficient was that session?"*  
Tomorrow TER answers: *"How should this session behave right now?"*

The core insight from recent research (SelfBudgeter, IARS, Route-To-Reason, Apple's "Illusion of Thinking") is that **fixed compute allocation is the primary source of token waste**. Models apply the same reasoning depth to trivial and complex tasks alike. TER's waste taxonomy — reasoning loops, verbose thinking, duplicate tool calls — are all symptoms of this mismatch. The cure is a closed-loop system where TER signals feed back into the session in real time.

---

## Completed Work

### Phase 1 — Foundation (v0.x)

**Core pipeline**: 9 modules (`models.py`, `loader.py`, `intent.py`, `classifier.py`, `waste.py`, `compute.py`, `formatter.py`, `compare.py`, `cli.py`) providing batch analysis of JSONL sessions with embedding-based intent alignment, 3 core + 5 extended waste patterns, and phase-weighted TER scoring.

**Bridge modules**: `real_time.py`, `adaptive_budget.py`, `cost_model.py`, `overthinking.py` — live monitoring, budget recommendations, cost-weighted TER, and reasoning value analysis.

**BDD test suite**: 538 tests across 37 feature files covering core pipeline, real-time, budgets, cost economics, waste detection, validation, and performance.

### Phase 1.5 — Watch Improvements (PR #18, 2026-05-13)

Fixed three critical issues discovered during real-world usage:

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Live watch showed 91% waste vs post-hoc 8% | Binary similarity gate (`sim >= 0.40 = aligned`) too aggressive | Adopted classifier's "aligned by default" philosophy with phase-specific heuristics |
| Watch not polling for new messages | Line-counting bug in `_read_new_lines` skipped appended content | Replaced with byte-offset `seek()` |
| No way to save watch output | Missing feature | Added `--log FILE` flag for JSONL output |
| Can't tell live from replay | All signals looked the same | Added `[HH:MM:SS] [LIVE/HISTORY]` tags using message timestamps |
| `--latest` replayed all sessions | Passed parent directory to LiveDashboard | Now uses SessionMonitor for single-file targeting |
| `python -m ter_calculator` didn't work | Missing `__main__.py` | Added entry point |

**Key finding — embedding quality gap**: The fast trigram-hash embeddings used for live monitoring lack semantic discriminative power. Cosine similarities between unrelated English texts cluster around 0.40-0.60, making intent-based waste detection unreliable. Repetition detection was added for reasoning/generation phases (threshold 0.88), but tool_use repetition produces false positives because structurally similar but semantically different tool calls (e.g., reading different files) hash to near-identical vectors. Tool calls remain always-aligned in live mode; post-hoc analysis with sentence-transformers catches the remaining ~8% waste.

### Phase 1.6 — Live Dashboard (PR #?, 2026-05-15)

**Objective**: Transform `ter watch` from cryptic line-by-line output into an interactive dashboard that makes live efficiency monitoring accessible and actionable.

**What changed**:
- **Dashboard mode** (new default): Rich-based interactive display with in-place updates showing TER, cost, cache hit rate, context growth, phase breakdown, warnings, and TER trend sparkline
- **Economics tracking**: Extended `RollingTERState` to accumulate token usage, cache metrics, and cost in real-time
- **Session metrics**: Added duration tracking, tokens/minute rate, context growth detection
- **Stream mode**: Renamed from simple/line-by-line; available via `--stream` flag
- **New module**: `dashboard.py` with Rich renderables for clean terminal display

**Implementation**:
| Component | What it does |
|-----------|-------------|
| `TERSignal` extended | Added 10 economics fields (input/output tokens, cache hit rate, cost, growth rate, duration) |
| `RollingTERState` extended | Added economics accumulators and helper methods for cost/cache/growth calculations |
| `compute_rolling_ter()` updated | Extracts usage data from JSONL, updates economics state, populates TERSignal |
| `rich_components.py` created | Shared Rich rendering components (panels, tables) used by both post-hoc and live |
| `dashboard.py` created | Live dashboard using shared components (158 lines) |
| `formatter.py` refactored | Post-hoc analysis now uses shared components from `rich_components.py` |
| CLI updated | Dashboard is default for `ter watch`, `--stream` for line-by-line mode |

**Architecture**: Shared rendering components eliminate duplication between post-hoc (`formatter.py`) and live (`dashboard.py`) displays. Both use the same visual components from `rich_components.py`, ensuring consistent look and feel while reducing code duplication by ~150 lines.

**Result**: Live monitoring now provides same visibility as post-hoc analysis, with real-time cost tracking and proactive warnings. Visual consistency achieved through shared components.

---

## Phase 2 — Embedding Quality & Live Accuracy

**Objective**: Close the gap between live watch and post-hoc analysis so that `ter watch` produces actionable, trustworthy signals in real time.

**Why this before routing/budgets**: The Phase 2 features in the original plan (task routing, budget control, prompt compression) all depend on accurate real-time TER signals. If watch shows 0% waste when post-hoc shows 8%, the routing decisions will be wrong. Fix the signal first.

### 2A. Lazy-Loaded Sentence-Transformer for Watch

The trigram hash was a premature optimization. Real sessions produce one assistant message every 2-10 seconds; embedding a few text spans per message with MiniLM-L6-v2 takes ~10ms on CPU. The polling interval (2s) provides ample budget.

- Load `sentence-transformers/all-MiniLM-L6-v2` on first signal (lazy init, not at startup)
- Cache the model instance on `SessionMonitor` / `LiveDashboard`
- Fall back to trigram hash if `sentence-transformers` is not installed (keep it optional)
- Keep `--model` flag for overriding with custom models
- **Success**: live TER within 3 percentage points of post-hoc TER on the same session

### 2B. Tool Call Deduplication

Post-hoc catches "unnecessary tool calls" via repetition detection with semantic embeddings. With proper embeddings from 2A, enable tool_use repetition detection in live mode:

- Compare tool call name + input against recent same-name calls (exact JSON match, not embedding)
- Flag exact duplicates as waste (e.g., running `mvn test` twice with no edits between)
- Different files with the same tool (e.g., `Read Gun.java` vs `Read Brain.java`) remain aligned
- **Success**: live watch catches the same ~5% tool waste that post-hoc identifies

### 2C. Bash Anti-Pattern Detection in Live Mode

The post-hoc `waste.py` already detects bash anti-patterns (30% of waste in test sessions). Port this to live:

- Check assistant bash commands against the anti-pattern catalog as they appear
- Requires access to the command text, not just embeddings
- Flag commands matching known anti-patterns (unnecessary `cat | grep`, re-running failed commands without changes, etc.)
- **Success**: live watch catches bash anti-patterns that currently show up only in post-hoc

### 2D. Success Criteria

- Live TER within 3 points of post-hoc TER on the same session
- Zero false-positive waste flags on legitimate different-file tool calls
- Watch startup remains under 3 seconds (model lazy-loads after first message)

---

## Phase 3 — Intervention Mechanisms

**Objective**: Move from observation to action. TER signals feed back into the active session.

**Decision resolved**: Use Claude Code hooks as the intervention mechanism. They execute shell commands in response to events (tool calls, messages), can modify the session context, and require no separate server process. This is simpler than an MCP server and already supported by the Claude Code CLI.

### 3A. TER Watch as a Claude Code Hook

Package `ter watch` as a hook that runs on each assistant message:

- Hook triggers on `assistant_message` events
- Runs TER classification on the new message content
- If waste signals exceed a threshold, outputs a warning that Claude Code surfaces to the user
- Configuration via `.claude/settings.json` with customizable thresholds

### 3B. Waste Pattern Prevention

Preventive interventions injected via hooks:

- **Reasoning loop breaker**: When 2 consecutive reasoning spans have >0.88 embedding similarity, inject: "You appear to be restating prior reasoning. Move to action."
- **Duplicate tool call preventer**: Before a tool call executes (pre-tool hook), check against session tool history. If exact duplicate: "You already ran this at step N with result X."
- **Permission loop circuit breaker**: After 2 denied tool calls of the same type: "This tool call has been denied. Try a different approach."

### 3C. Dynamic Token Budget Hints

Expose `ter budget` recommendations as a hook:

- On session start, analyze the initial prompt and inject a `max_thinking_tokens` recommendation into the system prompt
- Mid-session, if rolling TER shows reasoning plateauing (overthinking detector from `overthinking.py`), inject a budget reduction hint
- Track whether budget hints actually reduce waste (A/B comparison via `--log` output)

### 3D. Success Criteria

- Average session TER improves by 15%+ with hooks enabled vs disabled
- Reasoning loops reduced by 60%+ through preventive intervention
- Intervention precision > 95% (no false-positive interruptions)
- Hook latency < 500ms per message (must not slow down the session)

---

## Phase 4 — Cross-Session Intelligence

**Objective**: TER becomes a learning system that improves across sessions, users, and projects.

### 4A. Persistent TER Store

- Store session TER results in a local SQLite database (`~/.claude/ter/history.db`)
- Schema: session_id, project, timestamp, aggregate_ter, phase_ter, waste_breakdown, token_count, cost
- Queryable via `ter history` command with filters (project, date range, TER range)

### 4B. Project-Level TER Profiles

- Aggregate TER data per project directory
- Identify which projects have systemic waste patterns
- Surface recommendations: "Sessions in this project average 0.72 TER. The main waste source is redundant file reads — consider using `--latest` flag or narrowing your prompts."

### 4C. Predictive TER

- Given a prompt and project context, predict the TER before the session starts
- Use historical {prompt embedding, project, TER outcome} data
- Minimum viable: simple nearest-neighbor lookup on intent embeddings
- "This prompt is likely to produce a 0.55 TER session. Consider: adding explicit scope or specifying the target file."
- Requires 50+ sessions per project for reliable predictions

### 4D. Cost Optimization Dashboard

- `ter dashboard` command showing: TER trends, cost over time, model distribution, waste breakdown
- Rich terminal UI using `rich` (already a dependency)
- Weekly/monthly summaries: "This week: 12 sessions, avg TER 0.81, total cost $45.20, estimated waste $3.60"

### 4E. Success Criteria

- Predictive TER accuracy within 0.1 of actual for 80%+ of sessions
- Measurable project-wide TER improvement trend over 30 days
- Cost savings quantified and tracked per project

---

## Technical Decisions

### Resolved

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Intervention mechanism | Claude Code hooks | Already supported, no separate server, shell-level access |
| Embedding model for live | sentence-transformers/all-MiniLM-L6-v2 (lazy-loaded) | 10ms per span is fast enough for 2s polling; trigram hash too inaccurate |
| Tool call deduplication | Exact JSON match on name+input | Embedding similarity unreliable for structured tool call data |
| Live classification philosophy | Aligned by default, matching classifier.py | Prevents false waste inflation |

### Open

1. **Prompt compression**: LLMLingua-2 vs custom token filtering? LLMLingua-2 adds a PyTorch dependency. Defer until context rot is observed in real sessions.
2. **Privacy**: TER analysis touches session content. What anonymization is needed for cross-session features? Decide before Phase 4.
3. **Calibration data volume**: How many sessions per project before predictive TER becomes reliable? Empirical testing needed.
4. **Model routing**: Haiku/Sonnet/Opus routing requires API access to switch models mid-session. Is this feasible via hooks, or does it need API middleware?

---

## Dependency Map

```
Phase 1 (Foundation) ✅ COMPLETE
Phase 1.5 (Watch Fixes) ✅ COMPLETE
Phase 1.6 (Context Orchestrator) ✅ COMPLETE

Phase 2 (Embedding Quality)
├── 2A. Lazy sentence-transformer ← no deps, start here
├── 2B. Tool deduplication ← benefits from 2A but can use exact match
└── 2C. Bash anti-pattern live detection ← needs waste.py patterns

Phase 3 (Intervention) ← depends on accurate signals from Phase 2
├── 3A. Hook packaging ← needs 2A for accurate signals
├── 3B. Waste prevention ← needs 2B + 2C
├── 3C. Budget hints ← needs overthinking.py + adaptive_budget.py
└── All require Claude Code hooks API understanding

Phase 4 (Intelligence) ← depends on Phase 3 + sufficient data
├── 4A. Persistent store ← partially done via fragment_store.py (SQLite)
├── 4B. Project profiles ← needs 4A + context_graph.py
├── 4C. Predictive TER ← needs 4A + 50+ sessions + budget_optimizer.py
└── 4D. Dashboard ← needs 4A + cost_model.py
```

---

## Immediate Next Steps

1. ✅ ~~Build core pipeline (Phase 1A)~~ COMPLETE
2. ✅ ~~Bridge modules + tests + CLI (Phase 1B)~~ COMPLETE
3. ✅ ~~BDD specification suite~~ COMPLETE (PR #17, 538 tests)
4. ✅ ~~Fix watch alignment, polling, timestamps~~ COMPLETE (PR #18)
5. ✅ ~~Context Orchestrator (Phase 1.6)~~ COMPLETE — 5 modules, 93 unit tests, `ter context` CLI
6. **Next: Phase 2A** — Lazy-load sentence-transformers in live watch
7. **Then: Phase 2B** — Tool call deduplication via exact JSON match
8. **Then: Phase 2C** — Port bash anti-pattern detection to live mode
9. **Parallel: Phase 4A** — Extend fragment_store.py into full persistent TER store

---

## Research References

| Reference | Relevance |
|-----------|-----------|
| **SelfBudgeter** (May 2025) | Dual-phase token budget allocation — pre-estimate then budget-guided RL |
| **TALE** (Dec 2024) | Token-budget-aware reasoning — 81% accuracy at 32% cost |
| **Route-To-Reason** (May 2025) | Joint model + strategy routing — 60% token reduction |
| **IARS** | Intent-aware reasoning scheduler — adaptive directives during inference |
| **Apple "Illusion of Thinking"** (2025) | Overthinking phenomenon — models find answers early but keep exploring |
| **SDE** (April 2026) | Semantic density effect — information per token predicts quality |
| **Chroma Context Rot** | Performance degrades with input size independent of task difficulty |
| **LLMLingua-2** (EMNLP 2023) | 20x prompt compression with 1.5-point quality drop |
| **Anthropic Context Engineering** (2025) | "Find the smallest set of high-signal tokens" — official guidance |
| **NVIDIA Thinking Budget Control** | Production implementation of `max_thinking_tokens` with logits processor |
| **Nous Research Thinking Efficiency** | Reasoning token ratios vary 10x across models for identical tasks |
| **Mutual Information in Reasoning** | Specific tokens ("Wait", "Hmm") carry disproportionate reasoning value |

---

## Lessons Learned

Insights from real-world usage that should inform future work:

1. **Trigram hashing was a premature optimization**. Live monitoring has a 2-second polling interval — 10ms for a proper embedding is negligible. The accuracy loss from trigram hashing (0% waste vs 8% actual) made the live signal useless for decision-making. Always measure the actual latency budget before optimizing.

2. **"Aligned by default" is the right philosophy**. Agent tool calls are intentional actions, not idle chatter. The binary similarity gate classified 91% as waste on a session that was actually 96% efficient. Waste requires positive evidence (repetition, filler, verbosity), not just low similarity.

3. **Tool call similarity is structural, not semantic**. `Read Gun.java` and `Read Brain.java` share 90%+ trigram similarity because the path prefix dominates. Deduplication needs exact match on the discriminating parts (file path, command text), not embedding similarity.

4. **File polling needs byte offsets, not line counting**. Line counting across re-opens drifts when blank lines or parse errors occur. Byte-offset `seek()` is correct and simpler.

5. **Users need to know if output is live or historical**. Without timestamps and LIVE/HISTORY tags, watch output is ambiguous — you can't tell if the session is still active or if you're looking at stale replay.

6. **`--latest` should mean "the one file", not "the directory containing the latest file"**. Multi-session replay buries the live signal under hundreds of historical lines.
