# TER v2.0.4 — Phase 4 Cross-Session Intelligence

TER v2.0.4 adds an opt-in, local intelligence layer over historical TER results.

## Persistent history

`ter history record SESSION --project NAME` stores aggregate TER, phase scores,
waste categories, tokens, and estimated cost in `~/.claude/ter/history.db`.
Raw session content is not stored. Prompt text is converted to a deterministic
hashed bag-of-words fingerprint when prediction data is available.

## Project profiles

`ter history profile --project NAME` identifies systemic waste sources and
summarizes token and cost efficiency across sessions.

## Predictive TER

`ter history predict "PROMPT" --project NAME` performs a nearest-neighbor
estimate from privacy-preserving prompt fingerprints. Predictions are marked
experimental below 50 project sessions.

## Cost dashboard

`ter dashboard --project NAME` displays session totals, TER trend, total cost,
avoidable waste cost, and the dominant waste source.

## Privacy

Recording is explicit rather than automatic. The history database contains
aggregate metrics and optional hashes, never raw prompts or full session text.
