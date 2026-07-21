# Closed-loop interventions (v2.0.12)

TER v2.0.12 adds metric-driven context refreshes and replanning, plus outcome measurement.

## Data flow

`live metrics → policy evaluation → pending intervention → Claude Code hook → outcome evaluation → trends`

A metric payload may be supplied to `ter hook monitor` as `ter_metrics`, `metrics`, or `ter_signal`. It must include `ter` (or `aggregate_ter`) and should include `waste_ratio`, `context_tokens`, `context_growth_rate`, `drift_score`, `repeated_tool_calls`, and `reasoning_loop_streak`.

The policy requires sustained degradation rather than a single dip. Defaults:

- context refresh: TER drop ≥ 0.12 and waste ratio ≥ 0.25 for three windows;
- replan: TER drop ≥ 0.20 and waste ratio ≥ 0.40 for two windows;
- cooldowns: 120 seconds for refresh and 180 seconds for replan.

Configure these using `--ter-drop-warning`, `--ter-drop-replan`, `--waste-ratio-warning`, `--waste-ratio-replan`, `--degraded-windows-required`, `--refresh-cooldown-seconds`, and `--replan-cooldown-seconds`.

Pending interventions are written under `.ter/runtime/<session-id>/pending-intervention.json` and consumed once by the next applicable hook. `suggest` and `warn` inject recovery context. `block` may deny a tool call when a high-confidence replan is pending. `observe` records policy state without injecting guidance.

## Outcome evaluation

Each delivered intervention has a stable identifier and a baseline metric snapshot. After five metric-bearing events, TER records whether the recovery was acknowledged and followed, then classifies the result as:

- `improved`;
- `regressed`;
- `neutral`;
- `acknowledged_not_followed`;
- `ignored`.

Raw before/after metrics and deltas are retained in `.ter/intervention-outcomes.jsonl`. Trend analysis can aggregate compliance rate, improvement rate, override rate, mean TER delta, and mean waste delta by intervention type.

## Example hook payload

```json
{
  "hook_event_name": "UserPromptSubmit",
  "session_id": "abc123",
  "cwd": "/repo",
  "prompt": "Continue implementing the API",
  "ter_metrics": {
    "timestamp": 1784650000,
    "ter": 0.48,
    "waste_ratio": 0.43,
    "context_tokens": 98000,
    "context_growth_rate": 2.3,
    "drift_score": 0.21,
    "repeated_tool_calls": 4,
    "reasoning_loop_streak": 2
  }
}
```
