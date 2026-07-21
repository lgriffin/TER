# TER Waste Monitor — Claude Code Hooks Guide

The TER Waste Monitor runs as a Claude Code hook, detecting waste patterns in real-time during a session and injecting guidance back into Claude's context to course-correct before waste accumulates.

## How It Works

```
Claude calls a tool (Bash, Read, Edit, etc.)
        |
        v
Claude Code fires PostToolUse hook
        |
        v
ter hook monitor:
  1. Loads session state from temp file
  2. Runs 5 fast pattern checks against the tool call
  3. Saves updated state
  4. Returns JSON to Claude Code
        |
        v
  No alerts?  -> {} -> Claude continues normally
  Alerts?     -> additionalContext (Claude sees guidance)
               + systemMessage (you see a notification)
```

The monitor is stateful across tool calls within a session — it tracks read counts, edit sequences, command history, and tool call signatures in a JSON file at `{tempdir}/ter-hooks/{session_id}.json`. State is scoped per session and does not persist across sessions.

## Waste Patterns Detected

| Pattern | Triggers On | Default Threshold | What It Catches |
|---------|-------------|-------------------|-----------------|
| Bash antipattern | `Bash` tool | Immediate | `cat`, `grep`, `find`, `head`, `tail`, `rg` — commands that should use Read, Grep, or Glob |
| Repetitive read | `Read` tool | 3 reads | Same file read 3+ times when content is likely already in context |
| Edit fragmentation | `Edit`/`Write` | 3 consecutive | 3+ consecutive edits to the same file instead of batching |
| Duplicate tool call | Any tool | 2 identical | Exact same tool + parameters called twice |
| Repeated command | `Bash` tool | 3 runs | Same bash command run 3+ times (normalized to ignore `| tail -N` variants) |

## Setup

### Prerequisites

- Python 3.11+
- TER codebase cloned locally (pip install not required)

### Option A: TER installed via pip

If you've run `pip install -e .` in the TER repo:

Add to your project's `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -c \"import json; print(json.dumps({'systemMessage': 'TER Waste Monitor active'}))\"",
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Read|Edit|Write|Glob|Grep",
        "hooks": [
          {
            "type": "command",
            "command": "ter hook monitor",
            "timeout": 15,
            "statusMessage": "TER checking for waste..."
          }
        ]
      }
    ]
  }
}
```

### Option B: TER not on PATH (local clone)

Point `PYTHONPATH` to the TER source directory:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -c \"import json; print(json.dumps({'systemMessage': 'TER Waste Monitor active'}))\"",
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Read|Edit|Write|Glob|Grep",
        "hooks": [
          {
            "type": "command",
            "command": "python -m ter_calculator.cli hook monitor",
            "timeout": 15,
            "statusMessage": "TER checking for waste..."
          }
        ]
      }
    ]
  },
  "env": {
    "PYTHONPATH": "/path/to/TER/src"
  }
}
```

Replace `/path/to/TER/src` with the actual path to your TER clone's `src/` directory.

### Global setup

To enable across all projects, add the same configuration to `~/.claude/settings.json` instead of a project-level file.

## Verifying It Works

When you start a new Claude Code session, you should see:

```
TER Waste Monitor active
```

This confirms the `SessionStart` hook fired. The `PostToolUse` hook then runs silently on each tool call, only surfacing notifications when waste is detected.

To test manually:

```bash
echo '{"session_id":"test","tool_name":"Bash","tool_input":{"command":"cat foo.py"}}' | ter hook monitor
```

Expected output:

```json
{
  "additionalContext": "[TER Waste Monitor]\n  Bash anti-pattern: `cat foo.py`. Use the Read tool instead of `cat`...",
  "systemMessage": "TER: [~] bash antipattern"
}
```

## Customizing Thresholds

Add flags to the command string in your hook configuration:

```json
"command": "ter hook monitor --min-repetitive-reads 5 --min-edit-fragments 4 --min-duplicate-calls 3"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--min-repetitive-reads` | 3 | File read count before alerting |
| `--min-edit-fragments` | 3 | Consecutive same-file edits before alerting |
| `--min-repeated-commands` | 3 | Bash command repeat count before alerting |
| `--min-duplicate-calls` | 2 | Identical tool call count before alerting |
| `--no-bash-antipatterns` | off | Disable bash anti-pattern checking entirely |
| `--state-dir` | system temp | Override state file directory |

## What You See vs What Claude Sees

| Output | Recipient | Purpose |
|--------|-----------|---------|
| `statusMessage` | You (spinner) | Shows "TER checking for waste..." while the hook runs |
| `systemMessage` | You (warning) | Short notification like `TER: [!] repetitive read` |
| `additionalContext` | Claude only | Detailed guidance injected into Claude's context |

The `systemMessage` severity icons: `[~]` info, `[!]` warning, `[!!]` error.

## Troubleshooting

**No "TER Waste Monitor active" on session start:**
- Check that `.claude/settings.json` is valid JSON (use `python -m json.tool .claude/settings.json`)
- Verify Python is on PATH: `python --version`
- If using Option B, verify the PYTHONPATH: `PYTHONPATH=/path/to/TER/src python -c "import ter_calculator"`

**Hook fires but no alerts appear:**
- Alerts only trigger at thresholds — a single `cat` command triggers immediately (bash antipattern), but repetitive reads require 3 occurrences
- Check state dir for session files: `ls ${TMPDIR:-/tmp}/ter-hooks/`

**Hook is slow:**
- The monitor uses only stdlib (no numpy/ML). Typical execution is <100ms
- If slow, check disk I/O on the state dir. Use `--state-dir` to point to a faster location

## Phase 3 intervention configuration

TER v2.0.3 can use one monitor command across multiple Claude Code hook events:

```json
{
  "hooks": {
    "SessionStart": [{"hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 5}]}],
    "PreToolUse": [{
      "matcher": "Bash|Read|Edit|Write|Glob|Grep",
      "hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 5}]
    }],
    "PostToolUse": [{
      "matcher": "Bash|Read|Edit|Write|Glob|Grep",
      "hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 5}]
    }],
    "PostToolUseFailure": [{
      "matcher": "Bash|Read|Edit|Write|Glob|Grep",
      "hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 5}]
    }],
    "PermissionRequest": [{
      "matcher": "Bash|Edit|Write",
      "hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 5}]
    }],
    "Stop": [{"hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 5}]}]
  }
}
```

`PreToolUse` blocks only exact repeated calls. `SessionStart` injects an advisory
thinking-budget recommendation. Permission and reasoning-loop interventions
inject course-correction guidance after their configured thresholds.
