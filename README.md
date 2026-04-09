# TER Calculator

Token Efficiency Ratio (TER) calculator for Claude Code sessions. Measures how efficiently an AI coding agent uses its token budget by classifying output token spans as **aligned** (contributing to the task) or **waste** (redundant reasoning, unnecessary tool calls, over-explanation), and surfaces the economics of each session -- cost, cache efficiency, context growth, and where waste concentrates.

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

## Installation

```bash
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

### JSON output

```bash
ter analyze path/to/session.jsonl --format json
```

### Compare multiple sessions

```bash
ter compare session1.jsonl session2.jsonl --sort ter
```

### Discover sessions

```bash
ter list
ter list ~/.claude/projects/
```

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

ter compare <paths_or_dirs...>
  --format text|json
  --sort ter|tokens|waste
  Accepts directories (expands to all *.jsonl files inside)

ter list [path]
  --format text|json
  --limit N
```

## Try It

Sample sessions are included in `sample_sessions/`. Run TER against them to see what the output looks like:

```bash
# Analyze a single session
ter analyze sample_sessions/b1a1450c-b006-40fe-8f9c-f15622a94324.jsonl

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

Phases:     Reasoning  Tool Use  Generation
            1.00       0.92      1.00

Output Tokens: 52,497  (aligned: 48,553  waste: 3,944)

Input: 7,700  Cache Read: 3,239,712  Cache Hit: 99.8%
Context Growth: 5.7x over 49 turns [BLOAT]
Positional TER: 1.00 (early) / 0.66 (mid) / 0.89 (late)

Waste Patterns (1):
  Duplicate Tool Call: Duplicate tool call: Bash (17 tokens)
```

### Comparison across sessions

```
TER Comparison
════════════════════════════════════════

  #   Session      TER    Waste%   Cache%   Cost       Waste $    Patterns
  1   64948793...  0.99   2.7      100%     $10.20     $0.11      0
  2   b1a1450c...  0.97   7.5      100%     $2.45      $0.06      0
  3   a3b73c37...  0.94   9.5      100%     $8.25      $0.42      0
  4   ff410fa9...  0.88   5.9      100%     $10.47     $0.14      0
  5   3331fd66...  0.83   15.2     100%     $7.46      $0.31      0

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

Structural patterns detected across spans:

- **Reasoning loops** -- 3+ consecutive redundant reasoning spans
- **Duplicate tool calls** -- identical tool invocations within a 5-step window
- **Context restatement** -- response text repeating prior responses (similarity > 0.85)

### Session Economics

Surfaces the actual API token usage data from each session:

- **Input/Output tokens** -- real token counts from API usage (not heuristic estimates)
- **Cache hit rate** -- `cache_read / (cache_read + input_tokens)` -- measures prompt caching effectiveness
- **Estimated cost** -- USD estimate using configurable per-MTok rates (Sonnet defaults: $3 input, $15 output, $0.30 cache read, $3.75 cache write)

### Positional Analysis

Splits classified spans into thirds (early/mid/late) and computes TER per segment. Reveals whether waste concentrates early (unclear prompts) or late (session fatigue / context overload).

### Context Growth

Tracks total context size (input + cache read tokens) per turn. Detects:

- **Growth rate** -- ratio of final context size to first meaningful turn
- **Super-linear growth** -- context accelerating faster than linear (via second differences)
- **Context bloat** -- flagged when growth is both super-linear and > 2x

## How It Works

1. **Load** -- parses Claude Code JSONL session files, deduplicates streaming entries by `requestId`
2. **Segment** -- splits content blocks into token spans, assigns phases by block type
3. **Intent extraction** -- embeds user prompts using `all-MiniLM-L6-v2` (384-dim) to create an intent vector
4. **Classification** -- embeds each span, checks for self-repetition, applies phase-specific heuristics
5. **TER computation** -- calculates aligned/total ratio per phase, combines with weights
6. **Waste detection** -- scans for structural patterns (loops, duplicates, restatement)
7. **Economics** -- aggregates real API token usage, computes cache efficiency, cost, positional TER, and context growth

## Architecture

```
src/ter_calculator/
  cli.py          CLI entry point (analyze, compare, list)
  loader.py       JSONL parsing, deduplication, span segmentation
  intent.py       Intent extraction and embedding
  classifier.py   Span classification (aligned vs waste)
  compute.py      TER score computation
  waste.py        Waste pattern detection and summarization
  economics.py    Session economics, cost, positional analysis, growth
  formatter.py    Rich/text/JSON output formatting
  models.py       Data models and enums
```

## Development

```bash
cd src

# Run tests
pytest

# Lint
ruff check .
```

## Requirements

- Python 3.11+
- sentence-transformers (embeddings)
- numpy (similarity computation)
- rich (terminal formatting)
