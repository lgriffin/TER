# TER 2.0.2.1 — Phase 2 real-time intervention

This patch completes the lightweight Claude Code hook implementation of the
Phase 2 vision: detect degradation during a live session and request a refresh
before waste compounds.

## Added

- Rolling live-efficiency proxy over recent tool events.
- Degradation detection based on rolling score, negative drift, and waste
  acceleration.
- Mandatory replan/refresh guidance with an intervention cooldown.
- Repeated failed-action detection using PostToolUse result metadata.
- Persistent live metrics in per-session hook state.
- Backward-compatible state loading across TER versions.
- CLI controls for thresholds, windows, cooldowns, and disabling the policy.

## Scope

The live score is an explainable, stdlib-only proxy derived from deterministic
hook alerts. It intentionally does not claim to be the full offline TER model.
Historical-session RAG and cross-session feed-forward learning remain Phase 3.
