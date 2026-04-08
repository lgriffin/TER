# TER Calculator

Token Efficiency Ratio (TER) calculator for Claude Code sessions. Measures how efficiently an AI coding agent uses its token budget by classifying every token span as **aligned** (contributing to the task) or **waste** (redundant reasoning, unnecessary tool calls, over-explanation).

## What is TER?

TER is a score between 0 and 1 that answers: *"What fraction of tokens actually moved the task forward?"*

The score is computed per-phase and then combined with configurable weights:

| Phase | What it covers | Default Weight |
|-------|---------------|----------------|
| Reasoning | Thinking blocks, planning | 0.3 |
| Tool Use | Tool calls and results | 0.4 |
| Generation | Text responses to the user | 0.3 |

A TER of **0.95** means 95% of tokens were aligned with the user's intent. The remaining 5% were waste.

## How Waste is Defined

A token span is **aligned by default**. It is only classified as waste when a specific signal fires:

- **Self-repetition** -- a span closely duplicates a recent span in the same phase (cosine similarity >= 0.88)
- **Filler reasoning** -- a reasoning span with very low relevance (< 0.10) and fewer than 15 words
- **Verbose generation** -- a response span with extremely low relevance (< 0.08) and more than 50 words

Waste patterns are detected on top of classification:

- **Reasoning loops** -- 3+ consecutive redundant reasoning spans
- **Duplicate tool calls** -- identical tool invocations repeated within a 5-step window
- **Context restatement** -- response text that closely repeats prior responses (similarity > 0.85)

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

Output includes aggregate TER, per-phase scores, token counts, waste patterns, and a human-readable waste summary.

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

ter compare <paths...>
  --format text|json
  --sort ter|tokens|waste

ter list [path]
  --format text|json
  --limit N
```

## Example Output

```
+-- TER Report: my-session -----------------------+
| Aggregate TER: 0.9822  |  Raw Ratio: 0.9563     |
+-------------------------------------------------+

Phase Scores
  Reasoning:    1.0000
  Tool Use:     0.9555
  Generation:   1.0000

Token Summary
  Total:        58,172
  Aligned:      55,631
  Waste:         2,541

Waste Summary:
  2,541 of 58,172 tokens (4.4%) were identified as waste.
  The largest waste category is Unnecessary Tool Calls (2,541 tokens).
```

## How It Works

1. **Load** -- parses Claude Code JSONL session files, deduplicates streaming entries by `requestId`
2. **Segment** -- splits content blocks into token spans, assigns phases by block type (`thinking` -> reasoning, `tool_use`/`tool_result` -> tool use, `text` -> generation)
3. **Intent extraction** -- embeds user prompts using `all-MiniLM-L6-v2` (384-dim, ~22MB) to create an intent vector
4. **Classification** -- embeds each span, checks for self-repetition against recent same-phase spans, applies phase-specific heuristics
5. **TER computation** -- calculates aligned/total ratio per phase, combines with weights
6. **Waste detection** -- scans for structural patterns (loops, duplicates, restatement)

## Architecture

```
src/ter_calculator/
  cli.py          CLI entry point (analyze, compare, list)
  loader.py       JSONL parsing, deduplication, span segmentation
  intent.py       Intent extraction and embedding
  classifier.py   Span classification (aligned vs waste)
  compute.py      TER score computation
  waste.py        Waste pattern detection and summarization
  formatter.py    Rich/text/JSON output formatting
  models.py       Data models and enums
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/

# Type check
mypy src/
```

## Requirements

- Python 3.11+
- sentence-transformers (embeddings)
- numpy (similarity computation)
- rich (terminal formatting)

## License

See [patent.md](patent.md) for the TER methodology disclosure.
