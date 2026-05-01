# TER: From Post-Hoc Analysis to Real-Time Capability Adaptation

**Date**: 2026-05-01 | **Status**: Draft — for refinement  
**Goal**: Evolve TER from a session-replay analyzer into a real-time efficiency signal that drives adaptive model behavior, token budgeting, and cost optimization.

---

## Vision

Today TER answers: *"How efficient was that session?"*  
Tomorrow TER answers: *"How should this session behave right now?"*

The core insight from recent research (SelfBudgeter, IARS, Route-To-Reason, Apple's "Illusion of Thinking") is that **fixed compute allocation is the primary source of token waste**. Models apply the same reasoning depth to trivial and complex tasks alike. TER's waste taxonomy — reasoning loops, verbose thinking, duplicate tool calls — are all symptoms of this mismatch. The cure is a closed-loop system where TER signals feed back into the session in real time.

---

## Current State (v0.x — Post-Hoc Analyzer)

- Batch analysis of completed JSONL sessions
- Embedding-based intent alignment (all-MiniLM-L6-v2, 384-dim)
- 3 core + 5 extended waste patterns
- Phase-weighted TER scoring (R:0.3, T:0.4, G:0.3)
- Plugin system, validation, caching, acceleration modules designed
- **Gap**: No core pipeline implemented yet (models.py, loader.py, etc. pending)

---

## Phase 1 — Foundation (Current Sprint)

**Objective**: Get the core pipeline working end-to-end and add the four modules that bridge toward real-time.

### 1A. Core Pipeline (from existing spec)

Build the 9 core modules defined in `specs/001-ter-calculator/plan.md`:
- `models.py`, `loader.py`, `intent.py`, `classifier.py`, `waste.py`, `compute.py`, `formatter.py`, `compare.py`, `cli.py`

### 1B. Real-Time Bridge Modules (new — implementing now)

| Module | Purpose | Key Capability |
|--------|---------|----------------|
| `real_time.py` | Live session monitoring | Watch active JSONL files, compute rolling TER on each new message, emit efficiency signals as the session progresses |
| `adaptive_budget.py` | Token budget recommendations | Classify task complexity from intent, recommend thinking token budgets and model tier based on historical TER for similar tasks |
| `cost_model.py` | Cost-weighted TER + semantic density | Extend TER with dollar-cost weighting (input/output/cached pricing), compute semantic density (information per token) |
| `overthinking.py` | Reasoning value analysis | Detect when reasoning tokens plateau in marginal value using entropy/mutual-information proxies, recommend early termination points |

### 1C. Success Criteria

- `ter analyze <session>` produces correct TER with waste patterns
- `ter watch <project>` streams live TER updates for active sessions
- `ter budget <session>` recommends thinking token budget for similar future tasks
- Cost-weighted TER available via `--cost-weighted` flag
- Overthinking warnings appear in waste pattern output

---

## Phase 2 — Adaptive Routing (v1.x)

**Objective**: Use TER signals to route tasks to the right model at the right cost.

### 2A. Task Complexity Classifier

- Train a lightweight classifier on historical TER data: {intent embedding, session length, waste patterns} → complexity tier (simple/standard/complex)
- Use Anthropic's model tiers: Haiku for simple (TER historically > 0.85), Sonnet for standard, Opus for complex (TER historically < 0.5 due to task difficulty, not waste)
- Research reference: Route-To-Reason achieves 60% token reduction with joint model+strategy routing

### 2B. Dynamic Token Budget Controller

- Before a session starts, estimate complexity from the prompt and recommend `max_thinking_tokens`
- During a session, monitor rolling TER and signal when thinking is plateauing (overthinking detector)
- Research reference: TALE achieves 81% accuracy at 32% of vanilla CoT token cost; SelfBudgeter shows monotonic budget-to-complexity mapping
- Implementation: expose as an MCP server or Claude Code hook that injects budget hints

### 2C. Prompt Compression Integration

- When rolling TER degrades mid-session due to context size (context rot), trigger prompt compression
- Integrate LLMLingua-2 or similar for hard prompt compression (up to 20x compression, 1.5-point quality drop)
- Research reference: Chroma's context rot research shows performance degrades with input size even when task difficulty is constant
- Track compression-adjusted TER to measure net benefit

### 2D. Success Criteria

- Model routing reduces average cost per session by 30%+ without quality regression
- Dynamic budgets reduce thinking tokens by 40%+ on simple tasks
- Context compression prevents TER degradation in sessions > 50k input tokens

---

## Phase 3 — Closed-Loop Adaptation (v2.x)

**Objective**: Real-time TER feeds back into the active session, adjusting behavior mid-stream.

### 3A. Intent-Aware Reasoning Scheduler (IARS-inspired)

- Monitor reasoning tokens in real time, classify reasoning state: {exploring, confirming, ambiguous, near-answer}
- Issue adaptive directives: "you've been exploring for 2000 tokens with declining novelty — commit to an approach"
- Research reference: IARS operates purely at inference time with no retraining
- Implementation: Claude Code hook that injects system prompt amendments based on TER signals

### 3B. Waste Pattern Prevention (not just detection)

- **Preventive reasoning loop breaker**: When TER detects the start of a reasoning loop (2 consecutive similar reasoning spans), inject a prompt: "You appear to be restating prior reasoning. Move to action."
- **Duplicate tool call preventer**: Before a tool call executes, check against the session's tool call history. If duplicate, inject: "You already ran this command at step N with result X."
- **Permission loop circuit breaker**: After 2 denied tool calls of the same type, inject: "This tool call has been denied. Try a different approach."
- Research reference: Anthropic's context engineering guidance — "errors should steer agents toward more efficient behaviors"

### 3C. Semantic Density Optimization

- Measure semantic density of each generation span (information per token via embedding space density)
- When density drops below a threshold, signal the model to be more concise
- Research reference: SDE paper (April 2026) — higher semantic density per token correlates with better output quality and fewer hallucinations

### 3D. Success Criteria

- Average session TER improves by 20%+ compared to Phase 1 baseline
- Reasoning loops reduced by 80%+ through preventive intervention
- No false-positive interventions (intervention precision > 95%)

---

## Phase 4 — Intelligence Layer (v3.x)

**Objective**: TER becomes a learning system that improves across sessions, users, and organizations.

### 4A. Cross-Session Learning

- Build TER profiles per user, per project, per task type
- Identify which prompt patterns correlate with high TER and recommend them
- Track TER trends over time — are users getting more efficient?
- Surface: "Your refactoring sessions average 0.62 TER. Users who add explicit scope constraints average 0.81."

### 4B. Organizational Benchmarks

- Aggregate TER across team members (anonymized) to establish baselines
- Identify systemic waste patterns (e.g., "All sessions on the auth module have low tool-use TER because the test suite is slow")
- Feed into engineering process decisions: "Adding a pre-commit hook for auth tests would eliminate 40% of duplicate tool calls in that module"

### 4C. Predictive TER

- Given a prompt and project context, predict the TER before the session starts
- "This prompt is likely to produce a 0.55 TER session. Consider: adding explicit scope, specifying the target file, or using a simpler model."
- Train on historical {prompt, context, TER outcome} data

### 4D. Cost Optimization Dashboard

- Real-time dashboard showing: TER, cost, model distribution, token budget utilization
- Projected savings from optimization recommendations
- ROI tracking: "TER improvements this month saved $X in API costs"

### 4E. Success Criteria

- Predictive TER accuracy within 0.1 of actual for 80%+ of sessions
- Measurable team-wide TER improvement trend over 30 days
- Cost savings quantified and tracked

---

## Research References

These are the key papers and resources informing this roadmap:

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

## Technical Decisions to Make

These are open questions for Phase 2+ that need resolution:

1. **Intervention mechanism**: MCP server vs. Claude Code hooks vs. system prompt injection vs. API middleware? Each has different latency and capability tradeoffs.
2. **Model for complexity classification**: Fine-tuned classifier on TER data vs. LLM-as-judge vs. embedding-space clustering? Depends on data volume.
3. **Prompt compression library**: LLMLingua-2 vs. custom token filtering vs. API-side compression? LLMLingua-2 adds a PyTorch dependency.
4. **Real-time latency budget**: How fast must TER signals be to be useful for mid-session intervention? Sub-second? Sub-100ms?
5. **Privacy and data**: TER analysis touches session content. What anonymization is needed for cross-session and organizational features?
6. **Calibration data**: How much historical session data is needed before adaptive budgets and predictive TER become reliable? Minimum viable dataset.

---

## Dependency Map

```
Phase 1 (Foundation)
├── Core Pipeline (1A) ← must complete first
├── Real-Time Monitor (1B) ← depends on loader + models
├── Adaptive Budget (1B) ← depends on compute + intent
├── Cost Model (1B) ← depends on models + compute
└── Overthinking (1B) ← depends on models + classifier

Phase 2 (Routing) ← depends on Phase 1 complete + historical data
├── Task Classifier (2A) ← needs Phase 1 TER data
├── Budget Controller (2B) ← needs overthinking + adaptive_budget
├── Prompt Compression (2C) ← needs real_time + cost_model
└── All require intervention mechanism decision

Phase 3 (Closed-Loop) ← depends on Phase 2 + intervention mechanism
├── IARS (3A) ← needs budget controller + real_time
├── Waste Prevention (3B) ← needs waste detectors + real_time
└── Density Optimization (3C) ← needs cost_model semantic density

Phase 4 (Intelligence) ← depends on Phase 3 + sufficient data volume
├── Cross-Session Learning (4A) ← needs persistent TER store
├── Org Benchmarks (4B) ← needs cross-session + auth
├── Predictive TER (4C) ← needs large training set
└── Cost Dashboard (4D) ← needs cost_model + trending
```

---

## Immediate Next Steps

1. Build core pipeline (Phase 1A) — the 9 modules from the existing spec
2. Integrate the 4 new bridge modules (real_time, adaptive_budget, cost_model, overthinking)
3. Validate against sample sessions in `sample_sessions/`
4. Collect baseline TER data across 50+ sessions to inform Phase 2 decisions
5. Decide intervention mechanism (MCP server is likely best for Claude Code integration)
