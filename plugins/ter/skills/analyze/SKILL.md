---
name: analyze
description: Lean analysis of a Claude Code session with TER. Use when the user asks how efficient or wasteful a session was, wants a Lean A3 for a session, or asks what to change in CLAUDE.md, hooks or settings to cut agent waste.
argument-hint: "[session .jsonl path | 'this' | 'last']"
allowed-tools: Bash(python -m ter *), Bash(ter list *)
---

# Analyse a session with TER

TER reads a Claude Code transcript, classifies every step as value adding,
necessary or avoidable, and finds waste with evidence. Its A3 report turns
the findings into countermeasures: lines for CLAUDE.md, hooks and harness
settings. Your job is to run it, explain the result honestly, and offer to
apply the countermeasures the user agrees with.

## 1. Check TER is installed

```bash
python -m ter --version
```

If that fails, TER is not installed in this Python. Ask the user before
installing anything; TER needs Python 3.11 to 3.13:

```bash
python -m pip install "ter-calculator @ git+https://github.com/lgriffin/TER"
```

Use the interpreter the user names if they have one (a virtual environment's
`python`), and use that same interpreter for every command below.

## 2. Pick the session

The argument is `$ARGUMENTS`.

- A path to a `.jsonl` file: use it.
- Empty or `this`: the current session is `${CLAUDE_SESSION_ID}.jsonl` under
  the Claude Code projects folder (`~/.claude/projects/<project>/`, on
  Windows `%USERPROFILE%\.claude\projects\<project>\`). Find it with the Glob
  tool, pattern `**/${CLAUDE_SESSION_ID}.jsonl`.
- `last` or a description ("yesterday's refactor"): list recent sessions,
  newest first, and pick with the user if more than one fits:

  ```bash
  ter list --limit 10 --format json
  ```

## 3. Run the A3

Write the output outside the user's repository, in a folder under the
system temporary directory such as `<tmp>/ter/<session id>/`:

```bash
python -m ter a3 SESSION.jsonl --json OUT/a3.json --html OUT/a3.html
```

Add these only when they are true for this session:

- `--outcome results.xml`: JUnit XML from the tests the session ran (for
  example `pytest --junitxml`). Without it value efficiency is *unknown*.
- `--limits limits.json`: a control-limits file the user built with
  `python -m ter control limits`. It places the session against their own
  history.
- `--repo DIR`: the repository the session worked in, checked out at the
  commit the session started from. Never pass a checkout that has moved on.

For a quick text summary instead of the A3, run
`python -m ter explain SESSION.jsonl`.

## 4. Explain the result

Read `a3.json`. Lead with the problem and the current state, then the
findings. Keep to what TER measured, and keep these rules:

- **Confident and uncertain findings are different.** A finding with
  `"uncertain": true` (confidence below 0.70) is labelled uncertain. TER
  counts it as waste until someone verifies it, but it is not a fact: say so,
  and never recommend acting on it alone.
- **Value is never inferred from tokens.** If no `--outcome` was given, say
  value efficiency is unknown rather than guessing whether the work was good.
- **Cite the evidence.** Each finding lists the event ids behind it. Quote the
  finding's `explanation` and `subject` (the file or command) so the user can
  check it.
- Point the user to `a3.html` for the full one-page report.

## 5. Offer the countermeasures

Each entry in `countermeasures` addresses one or more findings and carries
`actions`, each with a `kind` and usually a `snippet`:

| `kind` | What applying it means |
|---|---|
| `claude_md` | Add the snippet to the project's CLAUDE.md |
| `hook` | Save the script it names and register it in `.claude/settings.json` |
| `setting` | Change a Claude Code harness setting |
| `practice` | Advice for the user; nothing to apply |

Present them grouped by finding, confident ones first. Skip countermeasures
marked `"uncertain": true` unless the user has checked the finding and agrees
it is waste. Then ask which to apply. Apply only what the user picks, show
the change before writing it, and never edit `~/.claude/settings.json` or any
other settings file without an explicit yes. Hook scripts are examples to
adapt: read the snippet, and fit it to the user's tools and shell (a `.sh`
script needs Git Bash or WSL on Windows).

## What this skill does not do

It does not turn on live capture or live advice. For that, the user runs
`/ter:setup-hooks`.
