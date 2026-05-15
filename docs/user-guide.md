# TER Calculator User Guide

A practical guide to using the Token Efficiency Ratio (TER) Calculator for analyzing, monitoring, and optimizing AI coding sessions.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Commands](#cli-commands)
4. [Common Workflows](#common-workflows)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python 3.11 or higher
- pip (included with Python)

### Standard Installation

Clone the repository and install from the repo root:

```bash
pip install -e .
```

### Development Installation

To include development dependencies (pytest, ruff, etc.):

```bash
pip install -e ".[dev]"
```

### Verify Installation

After installing, confirm the CLI is available:

```bash
ter --help
```

On first run, TER will download the sentence-transformer model used for semantic similarity analysis. This is a one-time download and may take a minute depending on your connection.

---

## Quick Start

Try these three commands to get a feel for the tool.

### 1. Analyze a Session

Pick any JSONL session file from the samples directory and run a basic analysis:

```bash
ter analyze sample_sessions/session_example.jsonl
```

This produces a TER score and breakdown showing how efficiently tokens were used during the session.

### 2. Get a Budget Recommendation

Before starting a task, ask TER how many tokens you should budget:

```bash
ter budget "Fix the auth bug"
```

TER uses historical patterns to suggest a token budget for the described task.

### 3. Store Context Fragments

Break a session into reusable context fragments for later optimization:

```bash
ter context store sample_sessions/session_example.jsonl
```

This shards the session into fragments that can be analyzed, optimized, and composed into efficient prompts.

---

## CLI Commands

### ter analyze

Perform a single-session TER analysis. This is the primary command for evaluating token efficiency.

```bash
ter analyze <path>
```

**Examples:**

```bash
# Basic analysis of a session file
ter analyze session.jsonl

# Analyze the latest session in a project directory
ter analyze project_dir/ --latest

# Output as JSON for programmatic use
ter analyze session.jsonl --format json

# Adjust similarity threshold for waste detection
ter analyze session.jsonl --similarity-threshold 0.8

# Include cost-weighted analysis
ter analyze session.jsonl --cost-weighted

# Check for overthinking patterns
ter analyze session.jsonl --check-overthinking

# Skip waste pattern detection for faster results
ter analyze session.jsonl --no-waste-patterns

# Skip input analysis
ter analyze session.jsonl --no-input-analysis

# Group analysis across a directory
ter analyze project_dir/ --group
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Analyze only the most recent session file in the given directory |
| `--format` | Output format: `text` (default), `json` |
| `--similarity-threshold` | Minimum similarity score to flag content as repeated (default: model-dependent) |
| `--confidence-threshold` | Minimum confidence for pattern detection |
| `--restatement-threshold` | Threshold for detecting restated content |
| `--phase-weights` | Custom weights for phase scoring |
| `--no-waste-patterns` | Skip waste pattern analysis |
| `--cost-model` | Cost model to use for token pricing |
| `--no-input-analysis` | Skip analysis of input/prompt tokens |
| `--prompt-similarity-threshold` | Threshold for prompt similarity detection |
| `--group` | Group and aggregate results across multiple sessions |
| `--cost-weighted` | Weight TER scores by token cost |
| `--check-overthinking` | Detect overthinking patterns in the session |

---

### ter report

Generate a Markdown summary report from a session analysis. Useful for sharing results or archiving.

```bash
ter report <path>
```

**Examples:**

```bash
# Generate a report to stdout
ter report session.jsonl

# Save the report to a file
ter report session.jsonl -o report.md

# Report on the latest session
ter report project_dir/ --latest
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Report on the most recent session file |
| `-o`, `--output` | Write report to the specified file instead of stdout |
| All threshold flags | Same threshold flags as `ter analyze` are supported |

---

### ter compare

Compare TER metrics across multiple sessions. Useful for tracking improvement over time or comparing approaches.

```bash
ter compare <path1> <path2> [<path3> ...]
```

**Examples:**

```bash
# Compare two sessions side by side
ter compare session_v1.jsonl session_v2.jsonl

# Compare multiple sessions sorted by TER score
ter compare *.jsonl --sort ter

# Use the first session as a baseline for relative comparison
ter compare baseline.jsonl improved.jsonl --baseline

# Output comparison as JSON
ter compare session_v1.jsonl session_v2.jsonl --format json
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--format` | Output format: `text` (default), `json` |
| `--sort` | Sort results by a specific metric |
| `--baseline` | Treat the first path as a baseline and show deltas |

---

### ter list

Discover session files in a directory. Helpful when you have many session files and need to find the right one.

```bash
ter list [path]
```

**Examples:**

```bash
# List all sessions in the current directory
ter list

# List sessions in a specific directory
ter list sample_sessions/

# Limit output to 5 most recent sessions
ter list sample_sessions/ --limit 5

# List sessions as JSON
ter list sample_sessions/ --format json
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--format` | Output format: `text` (default), `json` |
| `--limit` | Maximum number of sessions to list |

---

### ter watch

Monitor a session in real time. Displays live TER metrics as the session progresses.

```bash
ter watch <path>
```

**Examples:**

```bash
# Watch a live session file
ter watch session.jsonl

# Watch the latest session in a project directory
ter watch project_dir/ --latest

# Set a custom polling interval (in seconds)
ter watch session.jsonl --poll-interval 5

# Watch with a specific cost model
ter watch session.jsonl --model gpt-4

# Log watch output to a file
ter watch session.jsonl --log watch_output.jsonl
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Watch the most recent session file in the directory |
| `--poll-interval` | How often to check for updates, in seconds (default: 2) |
| `--format` | Output format: `text` (default), `json` |
| `--model` | Cost model for pricing calculations |
| `--log` | Write watch events to a log file |

Press `Ctrl+C` to stop watching.

---

### ter budget

Get a token budget recommendation before starting a task. TER analyzes the task description and optionally uses historical session data to suggest an appropriate budget.

```bash
ter budget "<task description>"
```

**Examples:**

```bash
# Simple budget recommendation
ter budget "Fix the auth bug"

# Use historical sessions to improve the estimate
ter budget "Add pagination to the API" --use-history

# Point to a specific history directory
ter budget "Refactor the database layer" --use-history --history-path ./past_sessions/

# Output as JSON
ter budget "Write unit tests for the parser" --format json
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--use-history` | Use historical session data to refine the budget estimate |
| `--history-path` | Path to directory containing historical sessions |
| `--format` | Output format: `text` (default), `json` |

---

### ter context store

Shard a session into context fragments. This is the first step in the context optimization pipeline.

```bash
ter context store <path>
```

**Examples:**

```bash
# Store fragments from a session
ter context store session.jsonl

# Store fragments from the latest session
ter context store project_dir/ --latest
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Process the most recent session file |

---

### ter context graph

Build and display a context graph showing relationships between fragments, topics, and dependencies.

```bash
ter context graph <path>
```

**Examples:**

```bash
# Display the context graph
ter context graph session.jsonl

# Display graph for the latest session
ter context graph project_dir/ --latest

# Output graph as JSON
ter context graph session.jsonl --format json
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Process the most recent session file |
| `--format` | Output format: `text` (default), `json` |

---

### ter context optimize

Run knapsack optimization on stored context fragments. Given a token budget, selects the most relevant fragments to include in a prompt.

```bash
ter context optimize <path>
```

**Examples:**

```bash
# Optimize with default budget
ter context optimize session.jsonl

# Set a specific token budget
ter context optimize session.jsonl --budget 4000

# Adjust the relevance threshold for fragment inclusion
ter context optimize session.jsonl --relevance-threshold 0.6

# Optimize the latest session
ter context optimize project_dir/ --latest
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Process the most recent session file |
| `--budget` | Target token budget for the optimized context |
| `--relevance-threshold` | Minimum relevance score for a fragment to be considered |

---

### ter context delta

Compose a delta prompt from context fragments. Shows only what has changed since the last interaction, reducing redundant context.

```bash
ter context delta <path>
```

**Examples:**

```bash
# Generate a delta prompt
ter context delta session.jsonl

# Delta from the latest session
ter context delta project_dir/ --latest
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Process the most recent session file |

---

### ter context check

Run consistency checks on context fragments. Detects contradictions, stale references, and other issues.

```bash
ter context check <path>
```

**Examples:**

```bash
# Basic consistency check
ter context check session.jsonl

# Check the latest session
ter context check project_dir/ --latest

# Group check across multiple sessions
ter context check project_dir/ --group

# Set the check mode
ter context check session.jsonl --mode strict
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--latest` | Process the most recent session file |
| `--group` | Check consistency across grouped sessions |
| `--mode` | Check mode (e.g., `strict`, `lenient`) |

---

## Common Workflows

### Analyzing a Completed Session

After finishing a coding session, review how efficiently tokens were used.

```bash
# Step 1: Run the analysis
ter analyze session.jsonl

# Step 2: Interpret the results
#   - Look at the overall TER score (higher is better)
#   - Check the waste percentage
#   - Review phase-by-phase scores to find where efficiency dropped

# Step 3: Generate a report for your records
ter report session.jsonl -o session_report.md

# Step 4: Act on the findings
#   - If waste is high in early phases, improve your initial prompts
#   - If context growth is excessive, use context optimization next time
#   - If cache hit rate is low, restructure prompts for better caching
```

### Before/After Comparison

Measure whether changes to your prompting strategy improved efficiency.

```bash
# Run your baseline session, then your improved session

# Compare them with baseline mode
ter compare baseline_session.jsonl improved_session.jsonl --baseline

# The output shows deltas for each metric:
#   TER score: +0.15 (improvement)
#   Waste %:   -8.2  (less waste)
```

### Live Monitoring a Session

Watch token efficiency in real time as a session progresses.

```bash
# Start watching before or during a session
ter watch project_dir/ --latest --poll-interval 3

# The display updates every 3 seconds showing:
#   - Current TER score
#   - Running token counts
#   - Waste indicators

# Press Ctrl+C to stop watching
```

### Pre-Session Budget Planning

Estimate how many tokens a task should require before you start.

```bash
# Get a budget estimate
ter budget "Implement user authentication with OAuth2"

# For better estimates, point to historical sessions
ter budget "Implement user authentication with OAuth2" \
  --use-history --history-path ./past_sessions/

# Use the suggested budget to set context limits or
# monitor actual usage against the estimate
```

### Context Optimization

Reduce token waste by optimizing which context fragments to include in follow-up sessions.

```bash
# Step 1: Store fragments from a completed session
ter context store session.jsonl

# Step 2: Optimize fragment selection for a given budget
ter context optimize session.jsonl --budget 4000

# Step 3: Review the selected fragments and their relevance scores
# The output shows which fragments were selected and why

# Optional: Generate a delta prompt for a follow-up session
ter context delta session.jsonl
```

---

## Understanding the Output

### TER Score

The Token Efficiency Ratio is the primary metric. It measures how much useful work was accomplished per token spent.

- **0.8 - 1.0**: Excellent efficiency. Tokens were well-spent with minimal waste.
- **0.6 - 0.8**: Good efficiency. Some room for improvement but generally solid.
- **0.4 - 0.6**: Moderate efficiency. Significant waste present. Review the phase breakdown to identify problem areas.
- **Below 0.4**: Poor efficiency. A large portion of tokens was wasted on repetition, restated context, or unproductive exchanges.

### Waste Percentage

The percentage of total tokens that did not contribute to productive output. Common sources of waste include:

- **Repeated content**: The model restating information already provided.
- **Redundant context**: Sending the same context multiple times across turns.
- **Overthinking**: Excessive reasoning that does not lead to better output.

Lower waste is better. Aim to keep waste below 20%.

### Phase Scores

TER breaks the session into phases (e.g., understanding, planning, implementation, verification) and scores each one independently. This helps pinpoint where inefficiency occurs.

- A low score in the understanding phase suggests your initial prompt was unclear.
- A low score in the implementation phase may indicate the model was exploring too many approaches.
- A low score in verification suggests excessive back-and-forth on testing or validation.

### Cache Hit Rate

Measures how effectively prompt caching was used across turns. A higher cache hit rate means fewer tokens were re-processed.

- **Above 70%**: Good use of caching. Prompts are structured to take advantage of prefix caching.
- **30% - 70%**: Moderate. Some restructuring of prompts could improve caching.
- **Below 30%**: Poor. Prompts are likely being restructured significantly between turns, preventing cache reuse.

### Context Growth

Tracks how the total context size changes over the course of a session. Rapid context growth indicates that context is accumulating without being managed.

- **Linear growth**: Normal for sessions that build on previous context.
- **Exponential growth**: Problematic. Context is being duplicated or unnecessarily expanded. Consider using `ter context optimize` to manage it.

### Positional TER

Shows how efficiency varies by position in the context window. Content near the edges of the window (beginning and end) tends to be used more effectively than content in the middle, reflecting known attention patterns in language models.

---

## Troubleshooting

### Model Download on First Run

**Symptom**: The first time you run `ter analyze`, there is a long delay while a model downloads.

**Cause**: TER uses a sentence-transformer model for semantic similarity analysis. This model is downloaded once and cached locally.

**Solution**: Wait for the download to complete. It typically takes 1-2 minutes on a standard connection. Subsequent runs will be fast. If the download fails, check your internet connection and try again. The model is cached in the standard Hugging Face cache directory (usually `~/.cache/huggingface/`).

### Empty Sessions

**Symptom**: `ter analyze` reports an error or returns a TER score of 0 for a session file.

**Cause**: The session file is empty, contains no valid JSONL entries, or has a format TER does not recognize.

**Solution**:
- Verify the file is not empty: check its size.
- Ensure the file contains valid JSONL (one JSON object per line).
- Confirm the session format matches what TER expects. Use `ter list` to check whether TER recognizes the file.

### Low TER Score Interpretation

**Symptom**: Your TER score is consistently low and you are not sure why.

**Cause**: Low TER scores can result from several factors, not all of which indicate a problem with your workflow.

**Solution**:
- Check the **phase breakdown** to identify which phases are underperforming.
- Check the **waste percentage** to see if specific waste patterns dominate.
- Exploratory sessions (debugging, research) naturally have lower TER scores than implementation sessions. This is expected.
- Compare similar sessions using `ter compare` rather than judging a single session in isolation.
- Use `--check-overthinking` to detect if the model spent excessive tokens reasoning without progressing.

### Windows Encoding Issues

**Symptom**: Errors related to character encoding, such as `UnicodeDecodeError`, when reading session files on Windows.

**Cause**: Session files may be encoded in UTF-8 but Windows may default to a different encoding (e.g., cp1252).

**Solution**:
- Ensure your session files are saved as UTF-8.
- Set the `PYTHONUTF8` environment variable to force Python to use UTF-8:

```bash
set PYTHONUTF8=1
ter analyze session.jsonl
```

- Alternatively, set this permanently in your environment variables.

### Session File Not Found

**Symptom**: TER reports that it cannot find the session file, even though the file exists.

**Solution**:
- Use absolute paths or ensure your working directory is correct.
- On Windows, use forward slashes (`/`) or properly escaped backslashes (`\\`) in paths.
- Check that the file extension is `.jsonl` (not `.json` or `.txt`).

### High Memory Usage

**Symptom**: TER uses a large amount of memory when analyzing very long sessions.

**Cause**: The sentence-transformer model and embedding computations for large sessions can be memory-intensive.

**Solution**:
- For very large sessions, consider splitting them into smaller segments.
- Close other memory-intensive applications during analysis.
- Use `--no-waste-patterns` and `--no-input-analysis` to reduce the scope of analysis and lower memory usage.
