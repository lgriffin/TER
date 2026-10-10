# TER v08 — Weighted Intent Construction

TER v08 replaces artificial prompt-text repetition with independently embedded prompts and an explicit weighted centroid.

## Signals

Prompt weights combine:

- information content,
- recency,
- correction strength,
- goal and constraint language,
- low-information operational filtering.

Messages such as `continue`, `retry`, and `go ahead` remain in provenance but receive very little embedding weight.

## Topic shifts

Adjacent informative prompts are compared semantically. When similarity falls below the topic threshold, a new intent topic begins. The compatibility `extract_intent` API represents the latest active topic while preserving every original prompt in `source_prompts`. `extract_intent_topics` exposes all detected topics.

## Compatibility

The historical `_combine_prompts_weighted` helper remains available but is no longer used by intent extraction. Existing `IntentVector` and classifier interfaces are unchanged. Production repetition thresholds are unchanged.
