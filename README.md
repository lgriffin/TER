# TER Calculator

[![CI](https://github.com/lgriffin/TER/actions/workflows/ci.yml/badge.svg)](https://github.com/lgriffin/TER/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Token Efficiency Ratio (TER) calculator for Claude Code sessions. Measures how efficiently an AI coding agent uses its token budget by classifying output token spans as **aligned** (contributing to intent) or **waste** (redundant reasoning, unnecessary tool calls, over-explanation), and surfaces session economics, context optimization, and cross-session consistency.

## Features

### Core Analysis
- **TER scoring** -- phase-weighted efficiency ratio (reasoning 0.3, tool use 0.4, generation 0.3)
- **8 waste pattern detectors** -- reasoning loops, duplicate tool calls, context restatement, repetitive reads, edit fragmentation, bash anti-patterns, failed retries, repeated commands
- **Session economics** -- real API token usage, cache hit rate, cost modeling, positional analysis, context growth detection
- **Input analysis** -- token breakdown by origin, prompt redundancy, intent drift, prompt-response alignment
- **Grouped analysis** -- parent + subagent sessions with token-weighted aggregates

### Real-Time & Adaptive
- **Live monitoring** (`ter watch`) -- rolling TER with drift detection and live warnings
- **Budget recommendations** (`ter budget`) -- complexity classification, model routing, thinking token budgets
- **Cost-weighted TER** (`--cost-weighted`) -- dollar-aware efficiency with semantic density scoring
- **Overthinking detection** (`--check-overthinking`) -- reasoning efficiency analysis with optimal cutoff detection

### Context Orchestrator
- **Fragment Store** (`ter context store`) -- content-addressable fragment storage with SHA-256 hashing, SQLite persistence, and automatic deduplication
- **Context Graph** (`ter context graph`) -- DAG of fragment relationships (dependency, derivation, co-occurrence) with topological sort and cycle detection
- **Budget Optimizer** (`ter context optimize`) -- knapsack optimization selecting maximum-relevance fragments within a token budget
- **Delta Composer** (`ter context delta`) -- reference-based prompt composition transmitting only uncached fragments
- **Consistency Coordinator** (`ter context check`) -- cross-session version skew detection with strict/relaxed enforcement modes

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.
## Usage

### Analyze a session

```bash
ter analyze path/to/session.jsonl
```

### Live monitoring (NEW)

Monitor active sessions in real-time with an **interactive dashboard** that updates as the session progresses:

```bash
# Dashboard mode (default) - rich interactive display
ter watch ~/.claude/projects/your-project

# Watch a specific session file
ter watch path/to/session.jsonl

# Stream mode - line-by-line output
ter watch --stream ~/.claude/projects/your-project
```

**Dashboard mode** displays (default):
```
╭───────────── TER Live Monitor — Session: 711bb9b1 — 🟢 LIVE ─────────────╮
│ TER: 0.97  │  Waste: 7.5%  │  Cost: $2.45  │  Waste $: $0.06           │
│ Drift: stable →  │  Messages: 49  │  Active: 15m 32s  │  Rate: 3,380 tok/min │
╰──────────────────────────────────────────────────────────────────────────╯
  Phases           Reasoning        Tool Use       Generation   
  Score               1.00            0.92            1.00      
                    ████████        ██████          ████████    
  Tokens     Output: 52,497  │  Aligned: 48,553  │  Waste: 3,944
             Input: 7,700  │  Cache: 3.2M  │  Hit: 99.8%
  Context    Growth: 5.7x over 49 turns  ⚠️  BLOAT DETECTED
Recent TER:  ▇▇▇▇▇▇▆▇▇▇  (0.97)
```

Features:
- **Real-time TER and cost tracking** with live updates
- **Phase breakdown** (reasoning/tool use/generation)
- **Cache hit rate** and input/output token metrics
- **Context growth monitoring** with bloat detection
- **Session duration** and tokens-per-minute rate
- **TER trend sparkline** showing recent history
- **Live warnings** when efficiency degrades
- **Updates in-place** (no scrolling) for clean monitoring

**Stream mode** provides traditional line-by-line output:
- Useful for logging or piping to other tools
- Each new message prints a status line
- Add `--log FILE` to save signals as JSONL for later analysis

### Budget recommendations 

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

### Cost-weighted analysis

Include cost analysis with dollar-weighted TER:

```bash
ter analyze path/to/session.jsonl --cost-weighted
```

Adds:
- Cost-weighted TER (weights waste by dollar cost, not just token count)
- Semantic density scoring (information per token)
- Per-phase cost breakdown
- Alternative model savings recommendations

### Overthinking detection

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

## Quick Start

```bash
# Analyze a session
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
  --stream                     Use streaming line-by-line output instead of dashboard
  --latest                     Watch the most recent session by modification time
  --log FILE                   Append signals as JSONL to FILE for later analysis
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

# Get budget recommendation before starting a task
ter budget "Fix the authentication bug in login.py"

# Monitor a live session
ter watch ~/.claude/projects/your-project --latest
# Monitor active sessions in real-time with live dashboard (NEW)
ter watch ~/.claude/projects/your-project
ter watch --stream ~/.claude/projects/your-project  # Stream mode

# Store session fragments for context optimization
ter context store sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl

# Optimize context within a token budget
ter context optimize sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl --budget 10000
```

## CLI Reference

### Analysis Commands

```
ter analyze <path>           Full TER analysis
  --latest                   Use most recent session
  --format text|json         Output format
  --cost-weighted            Include cost-weighted analysis
  --check-overthinking       Detect reasoning inefficiency
  --group                    Include subagent sessions
  --similarity-threshold     Alignment threshold (default: 0.40)
  --phase-weights r,t,g      Phase weights (default: 0.3,0.4,0.3)

ter report <path>            Markdown summary
  -o, --output FILE          Write to file instead of stdout

ter compare <paths...>       Multi-session comparison
  --sort ter|tokens|waste    Sort order
  --baseline                 Two-session before/after delta

ter list [path]              Discover sessions
  --limit N                  Max sessions to show
```

### Monitoring & Planning

```
ter watch <path>             Live session monitoring
  --latest                   Watch most recent session
  --poll-interval SECONDS    Poll frequency (default: 2.0)
  --log FILE                 Save signals as JSONL

ter budget <task-text>       Token budget recommendation
  --use-history              Learn from past sessions
```

### Context Orchestrator

```
ter context store <path>     Shard session into fragments
ter context graph <path>     Build and display context graph
ter context optimize <path>  Knapsack budget optimization
  --budget TOKENS            Token budget ceiling (required)
  --relevance-threshold      Min relevance score (default: 0.1)
ter context delta <path>     Show delta prompt composition
ter context check <path>     Cross-session consistency check
  --group                    Include subagent sessions
  --mode strict|relaxed      Consistency mode (default: relaxed)
```

## Architecture

```
src/ter_calculator/
  Core Pipeline:
    models.py               Data models and enums
    loader.py               JSONL parsing, span segmentation
    intent.py               Intent extraction and embedding
    classifier.py           Span classification (aligned vs waste)
    compute.py              TER score computation
    waste.py                Waste pattern detection (8 detectors)
    economics.py            Session economics and cost
    input_analysis.py       Input-side analysis
    formatter.py            Output formatting (Rich/JSON)
    compare.py              Multi-session comparison
    analyze_pipeline.py     Full analysis pipeline
    cli.py                  CLI entry point

  Real-Time & Adaptive:
    real_time.py            Live monitoring, rolling TER, drift detection
    adaptive_budget.py      Complexity estimation, budget recommendations
    cost_model.py           Cost-weighted TER, semantic density
    overthinking.py         Reasoning efficiency, optimal cutoff

  Context Orchestrator:
    fragment_store.py       Content-addressable fragment storage (SQLite)
    context_graph.py        Fragment relationship DAG
    budget_optimizer.py     Knapsack token budget optimization
    delta_composer.py       Reference-based prompt composition
    consistency.py          Cross-session version skew detection

  Infrastructure:
    embedding_cache.py      Span merging, disk cache, GPU detection
    token_counting.py       Calibrated token counting
    intent_extraction.py    Sliding window, hierarchical intent
    waste_detectors.py      Extended waste patterns
    feedback.py             Historical trending, CI thresholds
    plugins.py              Plugin system (protocols, registry)
    validation.py           JSONL validation, health reports
    acceleration.py         Incremental cache, quick mode
```

See [docs/architecture.md](docs/architecture.md) for detailed diagrams and data flow.

## How It Works

1. **Load** -- parse JSONL, deduplicate by requestId
2. **Segment** -- split content blocks into token spans by phase
3. **Intent** -- embed user prompts (all-MiniLM-L6-v2, 384-dim) to create intent vector
4. **Classify** -- embed spans, check self-repetition, apply phase-specific heuristics (aligned by default)
5. **Compute** -- per-phase aligned/total ratio, weighted aggregate
6. **Detect** -- structural waste patterns across the session
7. **Economics** -- real API token usage, cost, cache efficiency, context growth
8. **Context** (optional) -- fragment storage, graph construction, budget optimization

## Documentation

- [Architecture](docs/architecture.md) -- system design, module dependencies, data flow
- [Context Orchestrator](docs/context-orchestrator.md) -- patent implementation reference
- [User Guide](docs/user-guide.md) -- installation, workflows, troubleshooting

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up your development environment, running tests, and submitting pull requests.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/

# Type check
mypy src/

# Run specific test modules
pytest tests/unit/test_fragment_store.py -v
pytest tests/unit/test_budget_optimizer.py -v
```

## Limits of Interpretation

TER is a heuristic tool:
- Token counts use `len(text) // 4` approximation, not exact tokenization
- Waste classification uses embeddings and thresholds, not ground-truth labels
- Cost estimates use configurable per-MTok rates (Sonnet defaults)
- Context orchestrator fragment deduplication is content-based (identical text = same fragment)

## Requirements

- Python 3.11+
- sentence-transformers (embeddings)
- numpy (similarity computation)
- rich (terminal formatting)
- sqlite3 (stdlib, fragment storage)
