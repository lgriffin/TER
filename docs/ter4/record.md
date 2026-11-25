# The TER Record

A TER Record is one JSON object (`ter.record/0.1`) per session: what was
consumed, what it did, what was verified and how sure TER is. It carries no
prompt, path, file content, tool input or tool output, so it can leave the
machine the session ran on and be joined to billing and telemetry data
without anyone reading the session.
[ADR 0007](../decisions/0007-the-ter-record.md) records why; the JSON Schema
is `src/ter/data/schemas/ter.record/0.1.json`.

```bash
python -m ter record export session.jsonl --out session.record.json
python -m ter record verify session.record.json session.jsonl
```

`export` reads the session the way `python -m ter a3` does (`--tokenizer`,
`--ter`, `--outcome`) and prices it with the shipped price book. `verify`
rebuilds the record from the session it names, with the tokenizer and
TER 3 mode the record says it was built with, and lists every figure that
differs: exit 0 when the session is the one the record's events digest
names and every figure matches, 1 when not, 2 when either cannot be read.

## What it holds

| Block | Holds | Comes from |
|---|---|---|
| `produced_by` | event schema, tokenizer, detector fingerprint, TER 3 method | the run |
| `session` | hashed id, source format, event count, record coverage, usage limits | `SessionTrace` |
| `join` | models, charge period (the session date); principal, credential and conversation stay null | the source |
| `usage` | model turns and provider tokens: input, cache read, cache write, output | `TokenUsage` on each turn |
| `work` | generated and context tokens, tokens by activity class, waste tokens, flow efficiency by tokens and by time, TER | the L2 scorecard |
| `findings` | per detector: waste, kind, counts, confidence range, avoidable and uncertain tokens, evidence events, judged samples | findings and `ter/data/calibration.json` |
| `outcome` | verdict and check counts, generated tokens per verified outcome; `unknown` with no evidence | `ter.domain.outcome` |
| `shape` | intents, model turns per intent, agent depth, observed Big-T class | [observed shape](l2-explained.md#observed-shape) |
| `cost` | session cost by token type and by activity class, whole turn and output only, with every estimate reason; null without prices | [cost by activity class](l2-explained.md#context-inventory-and-dated-cost) |
| `integrity` | digests of the event stream and the evidence graph | the events and the graph |

## Rules

- **No content** (TER-REC-002). A finding is its detector and its numbers;
  its title, subject, explanation and evidence ids stay in the A3. The
  schema closes every object, and the only free strings are identifiers TER
  defines, model names the provider reported, dates and digests. A price
  book named by a path is reported as `custom`, and an outcome keeps its
  counts, never a check's name. A test scans each corpus session's text and
  tool arguments for any string that reaches its record.
- **Uncertainty travels with the number** (TER-REC-003, TER-REC-004).
  Tokens and cost a finding below 0.70 claims are `uncertain`, never
  `avoidable`. Each detector carries `calibration`: how many of its findings
  were judged on real sessions, over how many samples, and how many were
  waste, not waste or unsure. Counts are as judged under the rule in force
  that day; they say how much judgement a figure rests on, not its
  precision. The samples are data in `src/ter/data/calibration.json`
  (`ter.calibration/1`), read through the `CalibrationSource` port.
- **No outcome, no per-outcome figure** (TER-REC-006).
- **Recomputable** (TER-REC-005). The events digest ties a record to the
  session it was built from; `verify` rebuilds it from there.

## What it does not claim

The Record measures what the tokens did and reports a verdict beside it; it
does not measure value. Most detectors are not calibrated, so `avoidable`
is a hypothesis wherever `calibration.judged` is small. Records are not for
ranking models, vendors, harnesses or people.

The golden Records in `tests/golden/snapshots/record/` price the synthetic
corpus at sonnet rates. On `lean_mix` the avoidable work is 4.2% of the
session's cost priced by its output tokens and 19.3% priced by the whole
turns it consumed. The synthetic usage was written by hand, so only the gap
matters, and whether it holds is a question for real sessions.

## The first real run (25 Nov 2026)

The owner exported Records for 61 of his own Claude Code sessions (top-level
sessions only; the records stay on his PC). Every export succeeded. Only 7
sessions had a priced turn, because the price book then held three models
and the sessions mostly ran newer ones: 6,042 turns were unpriced, counted
and never guessed. Over the $62.78 those 7 sessions cost:

| Class | Whole turn | Output only | Ratio |
|---|---:|---:|---:|
| avoidable | $0.43 (0.7%) | $0.25 (0.4%) | 1.7× |
| uncertain | $4.40 (7.0%) | $0.78 (1.2%) | 5.7× |
| both | $4.82 (7.7%) | $1.03 (1.6%) | 4.7× |

So the gap the synthetic corpus showed holds on real sessions: pricing waste
by its output alone understates it about fivefold. Most of it is still
uncertain (ADR 0006), so it is a hypothesis until those detectors are
judged.

The run changed three things:

- The price book gained the current models (Claude Opus 5.5 and 5, Sonnet
  5.5 and 5, Fable 5.2 and 5.1) at their published rates, and the Haiku 4.5
  and Opus 4.6 rates TER 3 carried, which were an earlier generation's, were
  corrected (Opus 4.6 now costs a third of the old figure). A rerun prices
  far more of the corpus.
- Messages Claude Code writes itself, which name the model `<synthetic>`,
  carry no usage (TER-ANL-044): no model ran, so they are not unpriced turns.
  There were 30.
- A Record counts unpriced turns but not which model each was; the per-model
  counts above came from the session files.
