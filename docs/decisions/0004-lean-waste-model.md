# ADR 0004: A formal Lean waste model, with conservative plugin detectors

- Status: accepted
- Date: 2026-09-30

## Context

TER 3 labels spans as aligned or waste against the stated intent. That
answers "how many tokens were off-topic" but not "what kind of waste was this,
what caused it, and what should change". Maturity level L2 (Explained) must
classify agent activity in Lean terms (value-adding, necessary
non-value-adding, avoidable), map classic Lean wastes onto agent behaviour,
and trace every classification to observable events (points 7 to 40, 71 to
86 of the TER 4 brief). A heuristic that silently asserts waste would disrupt
productive agent behaviour (point 91).

## Decision

1. **The Lean model is pure domain code** in `ter.domain.lean`: activity
   classes, eight Lean wastes (rework, motion, over-processing, waiting,
   inventory, defects, overproduction, handoffs), six value-stream stages
   (intent → explore → plan → implement → validate → respond), and flow
   states. It imports only `ter.domain.events` and the standard library.
2. **Detectors are plugins** behind a `WasteDetector` protocol and a
   `DetectorRegistry`. Each publishes a `confidence_rule` in plain language.
   A finding cites every event it rests on (`evidence`) and the subset whose
   cost it claims (`waste_events`, scaled by `share`).
3. **Precision over recall.** Thresholds are structural (the same call with
   the same output, the same failure signature after a fix), never token
   counts. Findings below confidence 0.70 are *uncertain*: reported, counted
   in their own bucket, and never folded into avoidable waste or flow
   efficiency. (Superseded by ADR 0006: uncertain waste now counts until
   verified, still in its own bucket.) Risk findings (unvalidated or uninformed changes) claim no
   cost.
4. **Iteration is not rework.** A failed check followed by edits and a run
   that passes or fails differently is productive iteration (flow state
   *recovering*, counted toward flow efficiency). Only an unchanged failure
   signature after a fix is rework.
5. **No opaque score.** The scorecard keeps flow efficiency (tokens and
   time), activity-class proportions, waste cost and TER apart. A composite
   exists only with its components and weights (an unweighted mean).
6. **Live equals batch.** Steps are folded inside the L1 `AnalysisEngine`
   in O(1) amortised time per event; detectors run when an explanation is
   asked for. The equivalence suite asserts live == batch on the corpus.
7. **Countermeasures derive from findings**, per detector, with concrete
   CLAUDE.md lines, Claude Code hook configurations and settings. None is
   triggered by a fixed token threshold (point 139).

## Consequences

- Session-only evidence limits some detectors: "unused context" was
  uncertain until a judged sample (ADR 0006) promoted it.
- Validation outcomes are read from tool output text, because the
  `ter.event/0.1` contract carries no error flag. Unknown outcomes form no
  cycle rather than a guessed one. Adding an `is_error` field to the
  contract is a candidate for `ter.event/0.2`.
- New wastes and detectors are additive: register a detector, add its
  countermeasure and follow-up entries, and a golden snapshot diff shows the
  behaviour change for review.
