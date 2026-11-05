---
name: setup-hooks
description: Turn on TER's Claude Code hooks, to capture sessions live and, once the user has approved a policy file, to show live waste advice at the end of each turn.
argument-hint: "[capture | advice | check]"
disable-model-invocation: true
allowed-tools: Bash(python -m ter --version), Bash(python -m ter observe *), Bash(python -m ter policy check *), Bash(python -m ter policy effectiveness *)
---

# Set up TER's hooks

TER's hook is one command, `python -m ter hook`, that Claude Code calls with a
JSON payload on stdin. It always answers `{}` and never blocks a tool call.
It has two jobs, which the user turns on separately:

1. **Capture** (`capture`): record each prompt and tool call as events, for
   live analysis with `python -m ter observe`.
2. **Live advice** (`advice`): on `Stop`, show the user advice from a policy
   file they approved. The agent never sees it.

The argument is `$ARGUMENTS`; with none, ask which the user wants. `check`
means diagnose an existing setup (see Troubleshooting).

This plugin registers no hooks itself. Hooks record prompts and tool output
and run on every tool call, so the user chooses to turn them on, here, in a
settings file they can read.

## Before you change anything

- Check TER imports: `python -m ter --version`. If not, run `/ter:analyze`
  step 1 first.
- Find the absolute path of that interpreter
  (`python -c "import sys; print(sys.executable)"`). Hooks run without the
  user's shell setup, so the command must name it in full.
- Quote every path you put into a hook command (interpreter, event log,
  policy file, ledger, limits) in double quotes, escaped as `\"` inside the
  JSON string, so a path with spaces stays one argument. Refuse a path that
  contains a double quote, `$`, or a backtick rather than trying to escape it.
- Ask which settings file: the project's `.claude/settings.json` or the user's
  `~/.claude/settings.json` (every project). Show the exact JSON you will
  merge, keep every hook already there, and write only after a yes.

## Capture

Merge this into the chosen settings file, with `PYTHON` replaced by the
absolute interpreter path and `EVENTS` by the folder the user wants, each
quoted as above (default `~/.cache/ter/events`; the log holds prompts and
tool output, so keep it in their home directory):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "\"PYTHON\" -m ter hook --event-log \"EVENTS\""}]}
    ],
    "PostToolUse": [
      {"matcher": "*", "hooks": [{"type": "command", "command": "\"PYTHON\" -m ter hook --event-log \"EVENTS\""}]}
    ]
  }
}
```

PostToolUse alone records each tool call; add the same entry under
`PreToolUse` only if the user wants the request recorded before the tool runs.
After the user's next prompt, check events arrive:

```bash
python -m ter observe --event-log EVENTS
```

## Live advice

Live advice needs a policy file the **user** approved. Approval is a human
sign-off recorded in the file, and any edit voids it.

1. Check the file is valid and see its approval state:

   ```bash
   python -m ter policy check POLICIES.json
   ```

   A starting file is `docs/examples/policies/l4-advisory.json` in the TER
   repository. It ships unapproved.

2. Tell the user to read the policies and approve the file themselves, in
   their own terminal:
   `python -m ter policy approve POLICIES.json --by THEIR_NAME`.
   **Never run `policy approve` yourself, and never edit a policy file to make
   it pass.** If `policy check` still exits 1, stop and show why. Any later
   edit voids the approval, so the user approves again after every change.

3. With the file approved, recommend a shadow run over past sessions, so the
   user can judge the advice before any is shown (shadow mode refuses an
   unapproved file). It writes a ledger and the judge CSV, so ask first:

   ```bash
   python -m ter policy shadow POLICIES.json SESSIONS --ledger SHADOW_LEDGER --judge judge.csv
   ```

   Go through the judge CSV with the user. If they change a policy because
   of it, they approve the file again before step 4.

4. Once it is approved, add a `Stop` entry to the same settings file:

   ```json
   {
     "hooks": {
       "Stop": [
         {"hooks": [{"type": "command", "command": "\"PYTHON\" -m ter hook --event-log \"EVENTS\" --advise \"POLICIES.json\" --ledger \"LEDGER\""}]}
       ]
     }
   }
   ```

   Point `--advise` at a copy of the policy file kept outside any checkout
   (for example `~/ter-data/policies/`), so pulling a repository cannot void
   the approval. Add `--limits limits.json` if the policies name control
   signals.

5. After a week or so, show whether the advice helped:

   ```bash
   python -m ter policy effectiveness --ledger LEDGER
   ```

## Troubleshooting

- Nothing recorded: check the settings file is valid JSON and that the
  interpreter in the command imports TER. Claude Code's debug output shows
  `ter hook: event not recorded:` lines when the hook fails.
- Events in the wrong place: the hook uses `--event-log`, else
  `TER_EVENT_LOG_DIR`, else the default. Pass `--event-log` in the command.
- Full reference: `docs/guides/hooks.md` in the TER repository.
