# Phase 3 Implementation — TER v2.0.3

TER v2.0.3 completes the intervention layer described in `plan.md`. The existing
PostToolUse waste monitor remains intact and is extended into a unified Claude
Code hook handler.

## Implemented

- **Session-start budget hints** using TER's existing complexity estimator and
  adaptive-budget recommender.
- **Pre-tool duplicate prevention** that denies an exact repeated tool call and
  reports the previous result summary.
- **Reasoning-loop breaker** using a fast, deterministic token-vector cosine
  similarity check with a configurable default threshold of `0.88`.
- **Permission-loop circuit breaker** after two denied requests for the same
  tool.
- **Post-tool result memory** for actionable duplicate-call feedback.
- **Persistent intervention telemetry** in the per-session hook state.

## Hook events

A single command handles all supported events:

```bash
ter hook monitor
```

Configure the command under `SessionStart`, `PreToolUse`, `PostToolUse`,
`PostToolUseFailure`, `PermissionRequest`, and `Stop` as needed. Existing
PostToolUse-only configurations continue to work.

## New options

```text
--min-denied-calls N
--min-reasoning-loops N
--reasoning-similarity-threshold FLOAT
```

## Safety model

Only exact duplicate tool calls are blocked. Semantic or structurally similar
calls are not denied. Budget recommendations are advisory, and reasoning-loop
interventions require substantial repeated content.
