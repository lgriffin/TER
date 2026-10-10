# ADR 0006: Uncertain waste counts until verified

- Status: accepted
- Date: 2026-10-10
- Supersedes: point 3 of [ADR 0004](0004-lean-waste-model.md), for counting only

## Context

ADR 0004 kept findings below confidence 0.70 out of avoidable waste and flow
efficiency. On the first judged sample of real sessions (10 Oct 2026, 122
findings, see [l2-explained.md](../ter4/l2-explained.md#first-judged-sample-10-oct-2026))
most uncertain findings were waste: four uncertain detectors were 10 of 10,
and only two detectors produced mostly false findings, which were then
tightened or turned into a pointer. Leaving uncertain waste out of the
headline numbers therefore understated waste far more often than it
overstated it.

The owner's rule: **uncertain = waste until verified.**

## Decision

1. An uncertain waste finding counts as waste. Its share is in the
   scorecard's waste tokens, waste context tokens, waste seconds and finding
   count, in flow efficiency (as its waste's flow state), in the A3 Pareto
   and in the countermeasure cost.
2. It stays labelled. `Finding.uncertain`, the scorecard's
   `uncertain_findings` and `uncertain_waste_tokens`, the activity-class
   `uncertain` bucket and the value stream's `uncertain_tokens` say how much
   of the waste is unverified, so it is never read as a confirmed fact
   (points 85, 86).
3. It never triggers an intervention. Routing escalation and any other
   action that changes what the agent does still ignores uncertain findings
   (point 91: precision first where a false intervention disrupts work).
4. Verification goes both ways. A judged sample that finds a detector's
   findings are waste promotes it at the sample's 95% lower bound
   (`JUDGED_CONFIDENCE`, 0.72 for 10 of 10). A sample that finds they are
   not waste tightens the rule or makes the detector a risk, which claims no
   cost (`unrelated_modification`, 0 of 29 judged waste).

## Consequences

- Waste totals rise and flow efficiency falls on sessions with uncertain
  findings; the golden snapshots show the change.
