# TER 3.0 release notes

TER 3.0 turns TER from a session-scoring utility into a closed-loop quality and efficiency control layer for Claude Code.

## What ships in 3.0

### Static intelligence

TER indexes repository code, documentation, Git history, prior fixes, and durable session lessons. It retrieves relevant context before work proceeds and flags exact or semantic duplicate implementations, repeated defects, and project-specific risk patterns.

### Real-time intervention

TER monitors rolling TER, waste ratio, context growth, drift, repeated tool calls, and reasoning-loop signals. Sustained degradation can trigger an alert, context refresh, replanning request, or a high-confidence block according to the configured policy mode.

### Feed-forward learning

Every intervention can be tracked from issuance through acknowledgement, compliance, and measured outcome. Cross-session trends identify recurring scenarios, estimate cost saved or wasted, and support bounded per-repository threshold recommendations.

### Stakeholder visibility

The static effectiveness dashboard shows intervention volume, compliance, improvement rates, recurring scenarios, cost estimates, improvement-rate bars, weekly saved-versus-wasted charts, and applied or pending threshold tuning.

## Validation baseline

- Ruff formatting and linting: passed
- Mypy: passed across 84 source files
- Full test suite with extras: 1,162 passed
- Branch coverage gate: at least 90%
