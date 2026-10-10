# TER v07 — Blended Repetition Scoring

TER v07 replaces one-dimensional repetition decisions with an explainable score composed of semantic similarity, lexical overlap, entity overlap, action similarity, and parameter novelty.

The implementation is conservative. Exact structured tool duplicates score 1.0. Parameter changes sharply reduce tool repetition evidence. For reasoning and generation, semantic similarity remains dominant while new paths, identifiers, and numeric specifics reduce the score.

Existing per-phase production thresholds are unchanged. The scoring components are deterministic and independently testable so future benchmark data can calibrate their weights without changing the public data model.
