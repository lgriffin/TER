# Phase 12 — Pre-send duplicate and pattern checks

- Added the opt-in `pre_send_check` intervention for `UserPromptSubmit`.
- Added repository-memory and session-lesson similarity retrieval before prompt dispatch.
- Added explicit acknowledgement and override markers for held prompts.
- Added CLI flags for enablement, similarity threshold, and cooldown.
- Added outcome logging and generic trend aggregation for fired, acknowledged, overridden, and no-match checks.
- Added unit and integration coverage and updated hook/closed-loop documentation.
