# TER v2.0.11 — Metric-driven interventions and measured outcomes

TER v2.0.11 completes the first production-oriented closed-loop path:

```text
live signal → policy decision → hook intervention → outcome evaluation → trend analysis
```

## Sustained-degradation policy

TER evaluates rolling metric snapshots instead of reacting to a single dip. The policy combines TER decline, waste ratio, persistence across windows, and cooldown state.

Default thresholds:

| Setting | Default |
|---|---:|
| TER drop for refresh | `0.12` |
| TER drop for replan | `0.20` |
| Waste ratio for refresh | `0.25` |
| Waste ratio for replan | `0.40` |
| Consecutive degraded windows | `3` |
| Refresh cooldown | `120` seconds |
| Replan cooldown | `180` seconds |

Configure the policy through the unified Claude Code hook command:

```bash
ter hook monitor \
  --policy-mode warn \
  --ter-drop-warning 0.12 \
  --ter-drop-replan 0.20 \
  --waste-ratio-warning 0.25 \
  --waste-ratio-replan 0.40 \
  --degraded-windows-required 3 \
  --refresh-cooldown-seconds 120 \
  --replan-cooldown-seconds 180
```

Policy modes:

- `observe`: record decisions without changing Claude’s context;
- `suggest`: inject concise recovery guidance;
- `warn`: inject prominent refresh or replanning instructions;
- `block`: deny high-confidence redundant actions until corrective guidance is followed.

Metric degradation normally causes a refresh or replan, not a hard block. Blocking remains appropriate for deterministic conditions such as an exact duplicate tool call.

## Intervention delivery

The live `SessionMonitor` can emit pending policy decisions directly. Pending interventions are persisted per session, consumed once by the next eligible hook, and ignored after expiration.

A context refresh asks Claude to restate the objective, summarize verified progress, identify the blocker, discard obsolete hypotheses, retrieve relevant project memory, and continue with one concrete action.

A replan asks Claude to stop the current approach and provide the objective, known facts, failed approaches, remaining work, smallest verifiable next step, and proof criterion.

## Outcome evaluation

Each issued intervention receives an identifier and baseline metric snapshot. TER then evaluates subsequent events for two independent questions:

1. Did Claude acknowledge and follow the intervention?
2. Did the measured session state improve?

Stored outcome evidence can include:

- acknowledgement and compliance flags;
- before/after TER;
- before/after waste ratio;
- repeated-tool and reasoning-loop changes;
- effect classification such as `improved`, `regressed`, `followed_no_measurable_gain`, `acknowledged_not_followed`, or `ignored`;
- confidence and supporting evidence.

Raw measurements are retained alongside classifications so evaluation thresholds can be recalibrated later.

## Trend analysis

Use project-local lessons and outcomes to inspect recurring risks and intervention effectiveness:

```bash
ter memory trends --minimum-occurrences 2
ter memory trends --format json
```

Aggregates can include issuance count, acknowledgement rate, compliance rate, improvement rate, median TER delta, and median waste-ratio delta by intervention type.

## Runtime data

The default project-local files are:

```text
.ter/memory-index.json
.ter/session-lessons.jsonl
.ter/intervention-outcomes.jsonl
```

Explicit paths can be supplied with `--memory-index`, `--lesson-store`, and `--outcome-store`.

## Validation

The v2.0.11 release was validated with:

```text
ruff format --check src tests
ruff check src tests
mypy src/
python -m pytest
```

Verified result: 1,115 tests passed; mypy succeeded across 83 source files.
