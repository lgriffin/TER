# TER Calculator

Token Efficiency Ratio (TER) calculator for Claude Code sessions. Measures how efficiently an AI coding agent uses its token budget by classifying output token spans as **aligned** (contributing to the task) or **waste** (redundant reasoning, unnecessary tool calls, over-explanation), and surfaces the economics of each session -- cost, cache efficiency, context growth, and where waste concentrates.

**New in Phase 1B:** Real-time session monitoring, adaptive budget recommendations, cost-weighted analysis, and overthinking detection for proactive efficiency optimization.

## Why TER?

Every Claude Code session consumes tokens across two axes:

- **Output tokens** -- what the model generates (thinking, tool calls, responses). This is where waste happens: reasoning loops, duplicate tool calls, verbose explanations.
- **Input tokens** -- context sent to the model each turn (prompt, history, tool results, cached context). This is where cost accumulates and context bloat appears.

TER bridges both. The core score measures output quality (0-1, where 1 = every token was aligned with user intent). The economics layer reveals input cost structure, cache efficiency, and whether your context is growing faster than it should.

### What TER tells you

| Signal | What it means | What to do |
|--------|--------------|------------|
| Low TER (< 0.7) | Model is generating significant waste | Check which phase (reasoning/tool/generation) is dragging the score down |
| High waste % in tool use | Duplicate or unnecessary tool calls | Improve prompt specificity, reduce ambiguity |
| Low generation TER | Over-explanation or context restatement | The model is being verbose or repeating itself |
| Low cache hit rate | Prompt caching not effective | Restructure prompts for cacheability |
| Context bloat detected | Input tokens growing super-linearly | Break long sessions into smaller tasks |
| Late-session waste | Positional TER drops toward end | Session may be too long, model is losing focus |
| Early-session waste | Waste concentrated at start | Prompt was unclear, model needed iterations to understand intent |
| High prompt redundancy | User is repeating similar asks | Consolidate requests, be more specific upfront |
| Convergent drift | User keeps refining the same request | Model may not be addressing the core ask |
| Low prompt-response alignment | Responses don't match what was asked | Rephrase prompts or check if model is going off-track |
| Bash anti-patterns | Model using Bash where dedicated tools exist | Configure hooks or instructions to prefer Read/Grep/Glob |
| Failed tool retries | Tool calls failing and being retried | Check for incorrect paths, permissions, or assumptions |
| Edit fragmentation | Many sequential edits to the same file | Model should batch changes into fewer operations |

## Phase 1B: Real-Time & Adaptive Features

Beyond post-hoc analysis, TER now provides **real-time monitoring** and **proactive recommendations**:

### Live Session Monitoring (`ter watch`)

Monitor active Claude Code sessions as they run, with rolling TER computation:

- **Real-time TER updates** - See efficiency scores update as messages are added
- **Drift detection** - Automatic detection of improving/degrading/stable trends
- **Live warnings** - Get alerted when TER drops below thresholds
- **Multi-session tracking** - Monitor all active sessions in a project simultaneously

Use this to catch efficiency issues **while they're happening** rather than after the fact.

### Adaptive Budget Recommendations (`ter budget`)

Get model and token budget recommendations **before** starting a task:

- **Complexity classification** - Analyzes task description to classify as simple/standard/complex
- **Model tier routing** - Recommends haiku/sonnet/opus based on task complexity
- **Token budget estimates** - Suggests thinking token limits and total token estimates
- **Cost predictions** - Estimates session cost before you start
- **Historical learning** - Learns from past sessions to improve recommendations over time

Use this to **right-size** model selection and prevent over/under-provisioning compute.

### Cost-Weighted Analysis (`--cost-weighted`)

Extends TER with dollar-cost awareness:

- **Cost-weighted TER** - Weights waste by actual dollar cost, not just token count (thinking tokens cost less than output tokens, cached tokens cost less than fresh input)
- **Semantic density** - Measures information content per token (vocabulary richness, entropy, redundancy)
- **Phase cost breakdown** - Shows where your dollars are going (reasoning/tool/generation)
- **Alternative model savings** - Calculates how much you'd save by switching to a cheaper model with similar TER

Use this to understand **financial efficiency**, not just token efficiency.

### Overthinking Detection (`--check-overthinking`)

Identifies when reasoning tokens plateau in marginal value:

- **Reasoning efficiency** - Percentage of reasoning tokens that contributed value
- **Optimal cutoff detection** - Identifies the span where novelty drops and reasoning becomes repetitive
- **Waste quantification** - Calculates how many reasoning tokens were wasted on filler
- **Budget recommendations** - Suggests thinking token limits to prevent overthinking

Use this to detect the **"illusion of thinking"** phenomenon where models continue reasoning after finding the answer.

## Installation

The Python package lives in the **`TER/`** subdirectory of this repository (where `pyproject.toml` is). From the repo root:

```bash
cd TER
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### Analyze a session

```bash
ter analyze path/to/session.jsonl
```

### Live monitoring (NEW)

Monitor active sessions in real-time with rolling TER computation:

```bash
ter watch ~/.claude/projects/your-project
```

Shows live updates as sessions progress, including:
- Current TER and token counts
- Drift detection (improving/degrading/stable)
- Real-time warnings for efficiency issues

### Budget recommendations (NEW)

Get token budget and model recommendations for a task before starting:

```bash
ter budget "Fix the authentication bug in login.py"
ter budget "Implement full e-commerce checkout with Stripe" --use-history
```

Returns:
- Complexity classification (simple/standard/complex)
- Recommended model tier (haiku/sonnet/opus)
- Suggested thinking token budget
- Estimated total tokens and cost

### Cost-weighted analysis (NEW)

Include cost analysis with dollar-weighted TER:

```bash
ter analyze path/to/session.jsonl --cost-weighted
```

Adds:
- Cost-weighted TER (weights waste by dollar cost, not just token count)
- Semantic density scoring (information per token)
- Per-phase cost breakdown
- Alternative model savings recommendations

### Overthinking detection (NEW)

Analyze reasoning efficiency and detect when thinking plateaus:

```bash
ter analyze path/to/session.jsonl --check-overthinking
```

Shows:
- Reasoning efficiency percentage
- Optimal cutoff point where value drops
- Wasted reasoning tokens
- Recommended thinking budget

### Grouped analysis (parent + subagents)

When a session spawns subagents, use `--group` to analyze the entire run together:

```bash
ter analyze path/to/session.jsonl --group
```

This discovers subagent sessions automatically from the filesystem layout (`SESSION_ID/subagents/*.jsonl`), analyzes each one, and reports token-weighted aggregate TER, total cost, and per-session breakdown.

### JSON output

```bash
ter analyze path/to/session.jsonl --format json
```

### Compare multiple sessions

```bash
ter compare session1.jsonl session2.jsonl --sort ter
```

### Compare two sessions as before/after (baseline)

When you have exactly **two** session files (e.g. before and after a rules change), a Markdown **delta** table:

```bash
ter compare before.jsonl after.jsonl --baseline
```

Uses the same **default** thresholds as a plain `ter analyze` (see [docs/TER_GOAL_AND_CHANGES.md](docs/TER_GOAL_AND_CHANGES.md) for extending this).

### Discover sessions

```bash
ter list
ter list ~/.claude/projects/
```

Sessions with subagents show the count (e.g. `SESSION_ID (128.5 KB, 6 subagents)`). Subagent files are hidden from the listing.

### Markdown report (human summary)

```bash
ter report path/to/session.jsonl
ter report path/to/session.jsonl -o report.md
```

Prints a **Markdown** one-pager to **stdout**, or writes it to a file with **`-o` / `--output`**. Content includes TER, waste %, cost, **output calibration ratio**, cache, positional TER, top structural patterns, and suggested next steps. Same analysis pipeline and flags as `analyze` (except `--format` / `--group`).

### Options

```
ter analyze <path>
  --format text|json           Output format (default: text)
  --similarity-threshold       Cosine similarity threshold (default: 0.40)
  --confidence-threshold       Classifier confidence threshold (default: 0.75)
  --restatement-threshold      Context restatement threshold (default: 0.85)
  --phase-weights r,t,g        Phase weights (default: 0.3,0.4,0.3)
  --no-waste-patterns          Skip waste pattern detection
  --cost-model MODEL           Pricing: 'sonnet' (default) or 'input,output,cache_read,cache_write'
  --group                      Include subagent sessions in grouped analysis
  --no-input-analysis          Disable input analysis (token breakdown, drift, alignment)
  --prompt-similarity-threshold  Cosine similarity for flagging redundant prompts (default: 0.75)
  --cost-weighted              Include cost-weighted TER analysis (NEW)
  --check-overthinking         Analyze reasoning efficiency and detect overthinking (NEW)

ter watch <project-path>
  --poll-interval SECONDS      Seconds between polls (default: 2.0)
  --format text|json           Output format (default: text)
  --model PATH                 Path to custom sentence-transformers model (optional)

ter budget <intent-text>
  --use-history                Enable historical learning from past sessions
  --history-path PATH          Custom path to budget_history.json
  --format text|json           Output format (default: text)

ter compare <paths_or_dirs...>
  --format text|json
  --sort ter|tokens|waste
  --baseline                 Exactly two .jsonl files: before/after Markdown delta
  Accepts directories (expands to all *.jsonl files inside)

ter list [path]
  --format text|json
  --limit N

ter report <path>
  -o, --output FILE          Write Markdown to FILE instead of stdout
  (same threshold/cost/analysis flags as analyze)
```

## Try It

Sample sessions are included in `sample_sessions/`. Run TER against them to see what the output looks like:

```bash
# Analyze a single session
ter analyze sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl

# Analyze with cost and overthinking detection (NEW)
ter analyze sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl --cost-weighted --check-overthinking

# Get budget recommendation for a task (NEW)
ter budget "Fix the authentication bug in login.py"
ter budget "Implement a full user dashboard with charts" --use-history

# Monitor active sessions in real-time (NEW)
ter watch ~/.claude/projects/your-project

# Grouped analysis (parent + subagents)
ter analyze sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl --group

# Compare all sessions in a directory
ter compare sample_sessions/ --sort ter

# JSON output for programmatic use
ter analyze sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl --format json
```

## Example Output

### Single session

```
TER Report: b1a1450c...
════════════════════════════════════════

TER: 0.97  |  Waste: 7.5%  |  Cost: $2.45  |  Waste $: $0.06
Drift: stable  |  Alignment: 0.62  |  Redundancy: 0%  |  User: 12%

Phases:     Reasoning  Tool Use  Generation
            1.00       0.92      1.00

Output Tokens: 52,497  (aligned: 48,553  waste: 3,944)

Input: 7,700  Cache Read: 3,239,712  Cache Hit: 99.8%
Context Growth: 5.7x over 49 turns [BLOAT]
Positional TER: 1.00 (early) / 0.66 (mid) / 0.89 (late)

Waste Breakdown:
  Source                      Tokens     %       Cost      Count
  Duplicate Tool Calls           300   50%   $0.0045        2
  Redundant Reasoning            200   33%   $0.0030        3
  Over-Explanation               100   17%   $0.0015        1
  Total                          600  100%   $0.0090
```

### Grouped analysis (parent + subagents)

```
Group Analysis: b1a1450c...
══════════════════════════════════════════════════════

TER: 0.94  |  Waste: 8.2%  |  Cost: $15.30  |  Waste $: $0.42
Sessions: 1 parent + 6 subagent(s)  |  Tokens: 312,450

  Role       Session        TER    Waste%   Tokens     Cost       Waste $    Patterns
  parent     b1a1450c...    0.97   7.5      52,497     $2.45      $0.06      1
  agent      agent-001      0.92   10.1     48,210     $2.10      $0.08      2
  agent      agent-002      0.95   6.3      44,800     $1.95      $0.05      0
  ...

  Total                     0.94   8.2      312,450    $15.30     $0.42
```

### Comparison across sessions

```
TER Comparison
════════════════════════════════════════

  #   Session      TER    Waste%   Cache%   Cost       Waste $    Patterns
  1   64948793...  0.99   2.7      100%     $10.20     $0.11      0
  2   b1a1450c...  0.97   7.5      100%     $2.45      $0.06      1
  3   a3b73c37...  0.94   9.5      100%     $8.25      $0.42      3
  4   ff410fa9...  0.88   5.9      100%     $10.47     $0.14      2
  5   3331fd66...  0.83   15.2     100%     $7.46      $0.31      5

Average TER: 0.92  |  Total Cost: $38.84  |  Total Waste: $1.04
```

## Metrics Reference

### Core TER

TER is computed per-phase and combined with configurable weights:

| Phase | What it covers | Default Weight |
|-------|---------------|----------------|
| Reasoning | Thinking blocks, planning | 0.3 |
| Tool Use | Tool calls and results | 0.4 |
| Generation | Text responses to the user | 0.3 |

A span is **aligned by default**. It is only classified as waste when a specific signal fires:

- **Self-repetition** -- duplicates a recent same-phase span (cosine similarity >= 0.88)
- **Filler reasoning** -- very low relevance (< 0.10) and fewer than 15 words
- **Verbose generation** -- extremely low relevance (< 0.08) and more than 50 words

### Waste Patterns

Structural and behavioural patterns detected across the session:

- **Reasoning loops** -- 3+ consecutive redundant reasoning spans
- **Duplicate tool calls** -- identical tool invocations within a 5-step window
- **Context restatement** -- response text repeating prior responses (similarity > 0.85)
- **Repetitive reads** -- same file read 3+ times (first read necessary, rest redundant)
- **Edit fragmentation** -- 3+ consecutive edits to the same file (could be batched)
- **Bash anti-patterns** -- Bash commands that should use dedicated tools (`cat` → Read, `grep`/`rg` → Grep, `find` → Glob, `head`/`tail` → Read)
- **Failed tool retries** -- tool calls that error and are retried (wasted tokens on the failed attempt + error result)
- **Repeated commands** -- same Bash command run 3+ times (normalized to ignore trailing `| tail -N` / `| head -N`)

### Session Economics

Surfaces the actual API token usage data from each session:

- **Input/Output tokens** -- real token counts from API usage (not heuristic estimates)
- **Cache hit rate** -- `cache_read / (cache_read + input_tokens)` -- measures prompt caching effectiveness
- **Estimated cost** -- USD estimate using configurable per-MTok rates (Sonnet defaults: $3 input, $15 output, $0.30 cache read, $3.75 cache write)

### Positional Analysis

Splits classified spans into thirds (early/mid/late) and computes TER per segment. Reveals whether waste concentrates early (unclear prompts) or late (session fatigue / context overload).

### Input Analysis

Analyzes the user side of the conversation:

- **Token breakdown** -- classifies all tokens by origin (user prompt text, tool results, model reasoning, tool calls, generation) and computes the user/model ratio
- **Prompt redundancy** -- pairwise cosine similarity between user prompts; flags near-duplicate asks above a threshold (default 0.75) and reports a redundancy score
- **Intent drift** -- tracks how user intent evolves between consecutive prompts: convergent (refining same ask), divergent (new topic), or evolving (gradual shift). Reports overall trajectory (stable/convergent/divergent/mixed)
- **Prompt-response alignment** -- measures how well each model response matches its triggering prompt. Low alignment indicates the model went off-track

### Context Growth

Tracks total context size (input + cache read tokens) per turn. Detects:

- **Growth rate** -- ratio of final context size to first meaningful turn
- **Super-linear growth** -- context accelerating faster than linear (via second differences)
- **Context bloat** -- flagged when growth is both super-linear and > 2x

### Grouped Analysis

When `--group` is used, TER discovers subagent sessions from the filesystem and analyzes the full run:

- **Token-weighted TER** -- aggregate TER weighted by each session's token count, so large sessions dominate appropriately
- **Total cost/waste** -- summed across parent + all subagents
- **Per-session breakdown** -- each session shown with role (parent/agent), individual TER, waste%, cost, and pattern counts

## How It Works

1. **Load** -- parses Claude Code JSONL session files, deduplicates streaming entries by `requestId`
2. **Segment** -- splits content blocks into token spans, assigns phases by block type
3. **Intent extraction** -- embeds user prompts using `all-MiniLM-L6-v2` (384-dim) to create an intent vector
4. **Classification** -- embeds each span, checks for self-repetition, applies phase-specific heuristics
5. **TER computation** -- calculates aligned/total ratio per phase, combines with weights
6. **Waste detection** -- scans for structural patterns (loops, duplicates, restatement, bash anti-patterns, failed retries, repeated commands, repetitive reads, edit fragmentation)
7. **Economics** -- aggregates real API token usage, computes cache efficiency, cost, positional TER, and context growth
8. **Input analysis** -- token breakdown by origin, prompt redundancy, intent drift, prompt-response alignment
9. **Grouping** -- discovers and analyzes subagent sessions, computes token-weighted aggregates

## Architecture

```
src/ter_calculator/
  cli.py              CLI (analyze, report, compare, list, watch, budget)
  analyze_pipeline.py Shared full-session analysis (analyze + report)
  config_parse.py     Cost model & phase-weight parsing
  session_report.py   Markdown report + baseline delta formatting
  loader.py           JSONL parsing, deduplication, span segmentation, subagent discovery
  intent.py           Intent extraction and embedding
  classifier.py       Span classification (aligned vs waste)
  compute.py          TER score computation
  waste.py            Waste pattern detection (8 detectors) and summarization
  economics.py        Session economics, cost, positional analysis, growth
  input_analysis.py   Input-side analysis (token breakdown, redundancy, drift, alignment)
  formatter.py        Rich/text/JSON output formatting (single, comparison, grouped)
  models.py           Data models and enums
  
  # Phase 1B: Real-time & Adaptive Features
  real_time.py        Live session monitoring with rolling TER and drift detection
  adaptive_budget.py  Token budget recommendations based on task complexity
  cost_model.py       Cost-weighted TER and semantic density analysis
  overthinking.py     Reasoning efficiency analysis and waste detection
```

## Development

```bash
cd src

# Run tests
pytest

# Lint
ruff check .
```

## Limits of interpretation

TER is a **heuristic** tool, not a tokenizer clone of Anthropic’s API:

- **Spans** use `len(text) // 4` for rough token counts; they will not match billed tokens line-for-line.
- **Waste classification** uses embeddings and thresholds; it is **not** ground-truth labeling.
- **Waste \$** uses **assistant-origin** waste and **calibrates** to API `output_tokens` when usage data exists — see `waste_output_calibration_ratio` in JSON. Ratios far from **1.0** mean the heuristic mass diverges from billing; interpret dollars cautiously.
- **Input-priced vs output-priced** rows in the waste breakdown reflect whether waste behaved like **context re-injection** vs **generated** text (see [UPDATES.md](UPDATES.md)).

For **what changed** and **why** (measurement pass, decoupled tools, `ter report`), read [docs/TER_GOAL_AND_CHANGES.md](docs/TER_GOAL_AND_CHANGES.md).

## Changelog and design notes

Economics calibration, waste \$ pricing, classifier flags, and maintainer notes: [UPDATES.md](UPDATES.md).  
Evolution toward the project end goal and maintainer proposal: [docs/TER_GOAL_AND_CHANGES.md](docs/TER_GOAL_AND_CHANGES.md).

## Requirements

- Python 3.11+
- sentence-transformers (embeddings)
- numpy (similarity computation)
- rich (terminal formatting)
