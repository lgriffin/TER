# ADR 0007: TER as the efficiency ledger, and the TER Record

- Status: accepted
- Date: 2026-11-25
- Source: "Upstream of the Bill: TER as the Open Efficiency Layer for
  Tokenomics" (direction document, 25 Nov 2026) and its read-only prototype
  `ter-record-prototype.zip`

## Context

Cost tooling for AI work starts at the bill and works backwards. A billing
row says which model was used, by whom, with which token direction and cache
action, and at what price. It cannot say what share of those tokens did
useful work, or whether the outcome was verified. Session identifiers are
set aside as observability, so the bill has no session either.

TER already measures that step at the session. The Lean model classifies
every generated event (value-adding, necessary non-value-adding, avoidable,
uncertain), the price book prices each model turn at the rates in force when
the session ran (TER-ANL-040), and the outcome port judges a verdict against
an acceptance contract. Nothing portable leaves TER: the A3 is written for a
developer, names files and quotes prompts, and is not shaped for a billing or
telemetry pipeline to join.

The prototype priced the same waste two ways on the seven synthetic
sessions: by the output tokens of the avoidable events (3.7% to 11.8% of
session cost where there was waste) and by the whole cost of the model turns
those events consumed (6.5% to 21.9%). An avoidable step re-reads the
context before it writes a word, so most of its cost is input.

## Decision

1. **TER names a third ledger.** Billing is the settlement ledger and
   attribution joins it to invocations. TER is the **efficiency ledger**:
   written at the session, it records what the tokens did. TER is read three
   ways: the Ratio (the published definition), the Record (a portable
   receipt for one session) and the Reference (this repository).
2. **The Record is one JSON object per session, schema `ter.record/0.1`.**
   It holds counts, kinds, classes, rule names, confidence, calibration
   status and digests: session, usage, work, findings, outcome, cost, join,
   shape and integrity. It is built by a pure function in `ter.domain` from
   the analysis, the cost and the verdict, so it adds no new measure of its
   own beyond the two below.
3. **The Record carries no content, by construction.** No prompt text, file
   path, file content, tool input or tool output. A finding appears as its
   detector and its numbers; its title, subject and explanation stay in the
   A3. The schema closes every object (`additionalProperties: false`) and its
   only free strings are identifiers TER defines, model names the provider
   reported and digests.
4. **A turn's whole cost follows the activity class of what it generated.**
   Provider usage is reported once per model turn. The turn's cost (input,
   cache read, cache write, output) is shared among the generated events of
   that turn in proportion to their generated tokens, and each event's part
   is split by its classification: the avoidable share to `avoidable`, the
   uncertain share to `uncertain` (never to avoidable, ADR 0006), the rest
   to its base class. Class costs add up to the session cost. The output-only
   price is kept beside it, so the two can be compared on real sessions.
5. **Shape is observed, not read from the architecture.** The Record counts
   intents (`n`), model turns per intent (`k`) and agent depth (`a`) from the
   events of what ran, and names the consumption class they imply. Where the
   source does not record subagent nesting, depth is a lower bound and says
   so.
6. **Uncertainty travels with the number.** Uncertain tokens and cost stay
   in their own bucket, and each detector the Record reports carries the
   size and result of the judged samples behind it.
7. **A Record is recomputable.** It carries a digest of the event stream
   and of the evidence graph, and a verifier rebuilds every figure from the
   session and reports each one that differs.

   The digests are plain SHA-256, not keyed. Each one hashes the whole
   session in order, never a single field: the events digest covers every
   event's id, text, tool call, usage and timestamp, and the evidence digest
   covers a graph that carries every event id and the session id. Event and
   session ids are random UUIDs, so a guessed command or path cannot be
   confirmed without reproducing the entire log, ids included. Anyone who
   already holds the log learns nothing new, and that party is exactly the
   verifier, who needs no shared key to recompute the digests. A keyed digest
   would stop a party outside the producing company from checking a Record,
   which is what the Record is for. (Decided by the owner, 25 Nov 2026.)

## Consequences

- One new versioned format, `ter.record/0.1`, ships with a JSON Schema
  (TER-TST-001) and golden Records for the synthetic sessions.
- `SessionCost` gains the split by activity class and by token type, so the
  A3's `cost` block changes on purpose in the golden snapshots.
- The requirement areas `REC` (record) and `SHP` (observed shape) open.
- The cost split is a TER rule now, not the prototype's. Whether it adds
  anything is an open question for real sessions: if waste costs about the
  same priced either way, the split is dropped and the Record stays.
- What not to claim: that TER measures value (it measures what the tokens
  did and reports a verdict beside it), that waste figures are calibrated,
  or that Records can rank models, vendors, harnesses or people.
- Out of scope here, entered when their stages start: an OpenTelemetry GenAI
  session source, a FOCUS SKU price book and exporter, an MCP server for
  ended sessions and a conformance kit. Their requirements stay out of the
  catalogue until then, so levels L0 to L4 stay met.
