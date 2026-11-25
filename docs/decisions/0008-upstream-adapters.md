# ADR 0008: The upstream adapters (stages U2 and U3)

- Status: accepted
- Date: 2026-11-25
- Source: "Upstream of the Bill: TER as the Open Efficiency Layer for
  Tokenomics" (direction document, 25 Nov 2026), its capability surface and
  staged path; Stage U1 is [ADR 0007](0007-the-ter-record.md)

## Context

Stage U1 gave TER one portable output, the TER Record. The document's later
stages add two roads in and three ways out, all adapters around the same
core:

| Stage | Surface | Shape in the hexagon |
|---|---|---|
| U2 | OpenTelemetry GenAI source | `SessionSource.otel-genai`, reading exported span files |
| U2 | FOCUS SKU price book | `PriceBook.focus-sku-price`, reading a FOCUS SKU Price file |
| U2 | FOCUS exporter | a driving adapter joining Records to billing rows |
| U3 | MCP server | a driving adapter over the existing use cases |
| U3 | Conformance kit | the contract suites, packaged to run against an entry point |

Both standards are moving. The OpenTelemetry GenAI conventions are at
Development status; in 1.42.0 they left the core semantic conventions for
their own repository, which publishes no schema URL yet. FOCUS 1.5 ratifies
on 3 Dec 2026, and its SKU Price dataset ships as a preview whose columns may
change.

## Decision

1. **Every new surface is an adapter.** None changes the domain beyond the
   reader-facing text of new usage limits. Each reads or writes files by a
   pinned version and never imports an OpenTelemetry, FOCUS or MCP SDK
   (ADR 0005).
2. **OpenTelemetry spans are read by convention version** (TER-SRC-027 to
   TER-SRC-031). A span names its version through the `schemaUrl` of its
   instrumentation scope (a resource's `schemaUrl` versions the resource,
   not the spans). The adapter reads the versions it lists and refuses
   any other, naming it. A file that names none is read under the latest
   known version and says so (`convention-unstated`). Cached tokens, which
   the conventions count inside `gen_ai.usage.input_tokens`, are taken out
   of it, and a turn with no cache attribute is read as unreported rather
   than as zero, because the conventions record cache counts only when
   non-zero. Without message content (opt-in, off by default) each trace
   is one intent and the trace says so (`intents-from-traces`). One file is
   one session, keyed by `gen_ai.conversation.id`.
3. **The harness-neutral claim is a test, not a hope.** Every golden
   session is committed as the span export an instrumented agent would
   have written. Read that way, it gives the same events and the same
   Record as its transcript, apart from the source name, session id and
   digests (TER-SRC-031). This is the document's second falsification test,
   run on synthetic sessions; a real agent's export is the next check.
4. **The FOCUS price book reads the SKU Price preview by pinned version**
   (TER-ANL-045 to TER-ANL-047). Only public USD list prices are read. A
   row's rate comes from its SKU Price Details (`TokenDirection`,
   `TokenCacheAction`, `ModelId`). The 1.5 dataset defers its own SKU Price
   Details column, so the details come from the price row when it carries
   one, else from the provider's Cost and Usage rows matched by
   `SkuPriceId`. A file missing a 1.5 column is refused naming it, and a
   model missing one of its four rates on a date is not priced on that
   date. The book's name carries no file name, because it travels into
   Records.
5. **The FOCUS exporter joins many to one and writes only `x_` columns of
   its own** (TER-EXP-011 to TER-EXP-013). FOCUS keeps the session out of
   billing, so Records roll up to a billing row by model and charge period,
   and each Record's usage of that model (`usage.by_model`, added to the
   Record for this) gives its share of the row's tokens of the row's kind.
   Records carry no principal, so rows that differ only by principal share a
   Record's tokens by their consumed tokens rather than each counting them.
   Efficiency figures go only into columns with the `x_` prefix FOCUS
   reserves for custom columns, and no billing value changes.
6. **The MCP server answers only for sessions that have ended.** An agent
   that asks TER about the session it is running in would receive TER's
   analysis as input, which changes what the agent reads. At the L4 ceiling
   advice may change no agent input (TER-INT-002), so the document
   recommends holding that pull for L5, and the server's tools refuse a
   session that is still running.
7. **The conformance kit is the contract suites, run against an entry
   point.** An adapter author names an entry point; the kit loads it and
   runs that port's contract suite unchanged, naming each obligation that
   fails. `SessionSource` comes first, then `PriceBook` and `OutcomeSource`.

## Consequences

- The vision points P207 to P211 and their requirements enter the catalogue
  with each stage's change, verified by the tests in that change, so levels
  L0 to L4 stay met.
- `docs/ter4/otel-genai.md` documents the span mapping and
  `docs/ter4/focus.md` the FOCUS price book; the MCP and conformance pages
  follow their stages.
- What not to claim: that TER reads every OpenTelemetry export (it reads the
  versions it lists), that the FOCUS columns are final (the SKU Price
  dataset is a preview), or that the span reader has met a real agent's
  export (it has met the golden corpus written as spans).
- Out of scope still: the efficiency commons, which waits for L5
  calibration and the L6 datasets; a hosted service; dashboards; a pricing
  engine; a router.
