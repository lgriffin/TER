# TER v2.0.10 — Closed-loop project intelligence

TER now connects repository memory to live Claude Code hooks. Session starts and submitted prompts can retrieve related code, fixes, and duplicate patterns before work proceeds. Waste alerts are persisted as deduplicated project lessons, intervention issuance is audited separately, and recurring patterns can be summarized with `ter memory trends`.

## Hook example

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{"type": "command", "command": "ter hook monitor", "timeout": 15}]
    }]
  }
}
```

Run `ter memory index .` first. Runtime files are stored under `.ter/` by default:

- `memory-index.json`
- `session-lessons.jsonl`
- `intervention-outcomes.jsonl`
