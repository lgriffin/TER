# The per-run report and the A3

TER produces two kinds of page for a session. The **per-run report** answers
"how efficient was this session?" with TER, token composition, phases and
waste patterns. The **A3** answers "what went wrong, why, and what do we
change?" with the Lean model. This guide shows how to produce each, how to
read it, and how to apply what it recommends.

| Want | Command | Output |
|---|---|---|
| A quick Markdown summary | `ter report session.jsonl` | Markdown on stdout, or `-o FILE` |
| A self-contained visual report | `ter report session.jsonl --html report.html` | One HTML file, no scripts, no network |
| An interactive report with a span inspector | `ter analyze session.jsonl --format html -o report.html` | HTML with embedded JavaScript and data |
| Standalone charts | `ter visualize session.jsonl -o charts/` | One SVG per chart |
| A slide deck | `ter present session.jsonl -o slides.md` | Marp Markdown |
| The Lean A3 | `ter a3 session.jsonl --html a3.html --json a3.json` | One HTML page and its JSON view-model |
| Lean findings in the terminal | `ter explain session.jsonl` | Text, or JSON with `--json` |

`ter a3` and `ter explain` are the same commands as `python -m ter a3` and
`python -m ter explain`; the `ter` entry point hands them to TER 4.

## The per-run report

```bash
ter report sample_sessions/example_session.jsonl --html report.html
ter report sample_sessions/example_session.jsonl --html report.html -o report.md
ter report --latest ~/.claude/projects/my-project --html latest.html
```

With `--html` only, no Markdown is written unless `-o` is also given. The
report reads a format-neutral view-model (`ter.domain.report.SessionReport`),
and its renderers never read TER 3 types. See
[docs/ter4/reports.md](../ter4/reports.md) for the design and example charts.

### Reading it, top to bottom

| Section | What it tells you | What to look at |
|---|---|---|
| Header KPIs | TER with its interval, aligned and waste tokens, cost and waste cost, number of waste patterns, reliability | A wide interval or low reliability means read the rest as a hint, not a verdict |
| Token composition | Scored tokens by label, aligned first | Which waste label dominates |
| Span timeline | One cell per span, in order, in its phase's lane; waste cells red and hatched | Clusters of waste: a stuck stretch of the session |
| TER by phase | Reasoning, tool use and generation on a fixed 0 to 1 scale | The phase to look at first |
| TER across the session | Early, middle and late thirds against the session TER | Did efficiency decay as context grew? |
| Waste Pareto | Pattern types by tokens with the cumulative share | The one or two patterns worth acting on |
| Provider token volume | Input, output and cache tokens | A low cache share means context was rebuilt often |
| Waste-pattern table, uncertainty note | Every pattern with its tokens; how far to trust the numbers | Patterns to cross-check in the A3 |

TER scores only what the agent generated. User prompts shape intent but are
never counted as work.

### Charts and slides

```bash
ter visualize sample_sessions/example_session.jsonl -o charts/
ter visualize sample_sessions/example_session.jsonl --charts composition,waste_patterns,positional_ter
ter present sample_sessions/example_session.jsonl -o slides.md
```

Available charts: `key_metrics`, `waste_breakdown`, `composition`,
`phase_scores`, `waste_patterns`, `positional_ter`, `economics`. Each SVG
stands alone (hex fills, a `<title>` and `<desc>`), so it can go straight into
a ticket or a slide.

## The A3

```bash
ter a3 tests/golden/sessions/lean_mix.jsonl --html a3.html --json a3.json --graph evidence.json
ter a3 session.jsonl --json                      # A3 JSON on stdout
ter a3 session.jsonl --html a3.html --ter off    # skip TER entirely
python -m ter a3 session.jsonl --html a3.html --ter model
```

| Option | Effect |
|---|---|
| `--html FILE` | Write the A3 page |
| `--json [FILE]` | Write the A3 view-model (`ter.a3/0.1`); stdout without FILE |
| `--graph FILE` | Write the evidence graph (`ter.evidence/0.1`) |
| `--ter offline` | TER 3 with the deterministic regex tokenizer and lexical embedder (default; no downloads) |
| `--ter model` | TER 3 with its sentence-transformers model (may download on first use) |
| `--ter off` | No TER in the scorecard |
| `--tokenizer regex\|tiktoken` | How event text is counted (`regex` is offline) |
| `--outcome FILE` | Judge the run's test results (JUnit XML, e.g. `pytest --junitxml`) and show the verdict in an Outcome box beside the scorecard; also on `explain` ([outcome.md](../ter4/outcome.md)) |
| `--repo DIR` | L3: ground the analysis on the repository the session worked in, checked out at its start commit: change surface per task, edits outside it, imports that break the repository's import-linter contracts; also on `explain` ([l3-grounded.md](../ter4/l3-grounded.md)) |
| `--repo-engine NAME` | Repository engine for `--repo` (default `syntax`: Python, TypeScript, JavaScript, Svelte and Vue imports) |

The page is self-contained (no scripts, no requests), follows light and dark
themes, reflows to one column on a phone, and prints on one A3 landscape
sheet. Every measure and finding on it is in the JSON, and every finding
cites event ids you can find in `ter explain --json` output or the evidence
graph. The countermeasure order, the *Action N* numbers and each action's
claimed tokens and share of waste are worked out on the page from the JSON's
findings and scorecard; the JSON keeps countermeasures in detector order.

A header row of chips gives the maturity the page was built at (L2 Explained,
or L3 Grounded with `--repo`), the session, events, generated tokens, agent
time, cost when priced, and the outcome verdict when `--outcome` was given. A
section bar under it stays at the top of the window while you scroll and
jumps to each section.

### Section by section

**At a glance.** The problem statement, four headline measures with a bar
each (value-adding share, avoidable waste, flow efficiency, TER) and *Do
these first*: the top three countermeasures in the order to act on them,
each with what its findings claim and where the change lands (`CLAUDE.md`,
`.claude/settings.json`, or your way of working). Each links to its full
countermeasure in section 5.

The numbered sections after it follow A3 problem-solving order. Use the
summary to decide what to change and sections 1 to 4 to check why.

**1 Background.** The developer's prompts and a one-line problem statement,
for example: *"39% of the 697 tokens the agent generated went to avoidable
work across 5 confident finding(s); flow efficiency is 61% of tokens and 61%
of agent time; 1 risk(s) to the outcome were flagged."*

**Scorecard.** Separate dimensions, never one opaque score:

| Field (in `analysis.scorecard`) | Meaning |
|---|---|
| `flow_efficiency_tokens`, `flow_efficiency_time` | Share of generated tokens and agent time that was *progressing* or *recovering* |
| `activity_tokens`, `activity_seconds` | Value-adding, necessary non-value-adding, avoidable, uncertain |
| `waste_tokens`, `waste_context_tokens`, `waste_seconds` | Avoidable generated tokens, context tokens that re-entered the window, time |
| `findings`, `uncertain_findings`, `risks`, `iterations`, `rework_cycles` | Counts |
| `ter` | The TER 3 ratio and the method that computed it |
| `composite` | Only with its components, weights and formula |

**Process control** (only with `--limits FILE`). The session placed
against your process's control limits, from `python -m ter control limits`
(see [Control charts](../ter4/control-charts.md)). Measures outside a limit
come first, firing ones on top. Each one shows its value, its limits and
centre, and links to the findings that move that measure. Read those findings
first. The measures inside their limits fold away. Only the one-session rule,
beyond limits, applies on an A3; the zone and run rules need the sessions
around this one, so they are on the control chart. Limits from another
detector set are marked stale.

```bash
ter a3 session.jsonl --limits limits.json --html a3.html
```

**2 Current state.** A value stream map: intent, explore, plan, implement,
validate, respond, with steps, tokens, context tokens and time per stage.
Stages where a confident finding landed are outlined in red with a badge, and
each shows its avoidable and uncertain share. This is where you see *where*
in the flow the waste sits.

**3 Analysis.** The waste Pareto (Lean wastes by tokens), a 100% bar of
activity classes, flow by tokens and by time, and any fail → fix cycles with
their verdict: *iteration* (the failure moved) or *rework* (it did not).

**4 Root causes.** Every finding as a card: detector, Lean waste, confidence
(with a bar), title, an explanation in plain words, its cost, the evidence
event ids, and a *Fix* link to the countermeasure that answers it. Waste
cards have a red edge, risks a violet one, and uncertain findings a dashed
yellow one; uncertain findings count in the headline waste until verified
and are reported separately ([ADR 0006](../decisions/0006-uncertain-waste-counts-until-verified.md)). Risk
findings (defects) claim no token cost.

**Repository evidence** (L3, only with `--repo`). The share of judged
repository reads that later work used, the context tokens carried by reads
nothing used, and files explored against files changed. In the explored list
a tick marks a file edited after it was read. In the changed list a tick marks
a file read before its first edit, `!` one edited before it was read, and `+`
one the session created; edits whose tool call failed do not count as
changes. Then the unused reads by context tokens with their read events (past
the first eight in a disclosure), and the outcome-value table: exploration,
reasoning and validation steps judged required, supporting or of no value.

**5 Countermeasures.** One numbered block per detector that fired, in the
order to act on them: confident before verify-first, outcome risks first,
then by the tokens their findings claim. Each has a title, the findings it
answers (linked back to their cards), a rationale, and concrete actions of
four kinds, each snippet labelled with the file it goes in:

| Action kind | What it is | Example |
|---|---|---|
| Add to CLAUDE.md | A line of standing instruction for the agent | "Change existing files with Edit; use Write only for new files…" |
| Install a hook | A Claude Code hook: the `.claude/settings.json` snippet and, where needed, a script | A `PreToolUse` hook on `Write` that blocks rewriting an existing file |
| Harness setting | A Claude Code setting | `"permissions": {"defaultMode": "plan"}` |
| Practice | Something for the developer to do | "Run the failing test alone with full output before the next edit" |

A countermeasure that answers only uncertain findings says so: verify the
finding before acting on it. Its cost still counts, because uncertain waste
is waste until verified (ADR 0006).

**6 Follow-up.** One row per fired detector: the metric, its current value,
the target for the next session, and where to read it in the JSON (for
example `findings[detector=repeated_tool_call]`).

**Evidence and method.** How classifications were made, and every detector's
confidence rule.

### Applying the countermeasures

Treat each countermeasure as an experiment: apply it, run the next comparable
session, compare the follow-up metric.

**CLAUDE.md lines.** Copy the line into the project's `CLAUDE.md` (or a
nested one for the directory concerned). Lines that name files or commands
are built from this session's findings, for example the test command the
agent actually ran. Adapt them; keep them short and imperative.

**Hooks.** A hook action has two parts: a settings snippet and, for most
detectors, a script. To install one:

1. Save the script as `.claude/hooks/<name>.sh` in the repository and make it
   executable (`chmod +x`). The scripts use `jq` and `sha1sum`.
2. Merge the snippet's `hooks` entry into `.claude/settings.json`. Merge, do
   not replace, if you already have hooks for that event.
3. Start a new Claude Code session and check the hook fires on a deliberate
   repeat.

The scripts rely on documented hook behaviour: exit code 2 from a `PreToolUse`
hook blocks the call and shows stderr to the agent; after a `PostToolUse`
call, exit code 2 feeds stderr back to the agent. The [hooks
guide](hooks.md#hooks-recommended-by-waste-findings) lists every recommended
hook by detector.

**Settings.** Merge the JSON into `.claude/settings.json` (project) or
`~/.claude/settings.json` (user). `permissions.defaultMode: plan` makes the
agent propose before it edits; `permissions.deny: ["Task"]` stops delegation
for small tasks. Both slow some work down: apply them where the findings say
they help.

**Practices.** These are for you: how you phrase prompts, when you ask for a
plan, when you `/compact`.

### Checking that it worked

```bash
ter a3 before.jsonl --json before.json
ter a3 after.jsonl --json after.json
jq '.analysis.scorecard | {flow_efficiency_tokens, waste_tokens, findings}' before.json after.json
jq '.follow_up' after.json
```

Keep the change if the follow-up metric moved toward its target on
comparable work. If a hook fires often and flow does not improve, remove it:
repeated warnings are waste too.

## Machine-readable output

| Schema | Produced by | Holds |
|---|---|---|
| `ter.a3/0.1` | `ter a3 --json` | Background, problem, current state, analysis (scorecard, Pareto, cycles), root causes, findings, countermeasures, follow-up, detectors |
| `ter.session-control/1`, under `process_control` in `ter.a3/0.1` | `ter a3 --limits FILE --json` | Every measure placed against its limits, signals, and the findings behind each measure outside a limit |
| `ter.evidence/0.1` | `--graph FILE` | Nodes per event and typed edges (`completes`, `motivated_by`, `validates`, `corrects`, `repeats`) |
| explain JSON | `ter explain --json` | Findings, cycles, value stream, scorecard, per-event classification with basis, detectors, evidence graph |

The golden snapshots under `tests/golden/snapshots/` (`<name>.a3.json`,
`<name>.lean.json`, `report/<name>.a3.html`) are worked examples of each.
