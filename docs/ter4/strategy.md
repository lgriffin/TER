# TER 4 strategy and maturity roadmap

9 October 2026. Where TER 4 stands after L3 Grounded, and how it gets from
here to L6 Learning.

## Vision

TER started as the Token Efficiency Ratio: the share of an agent's output
that served what the developer asked. TER 4 turns it into **Lean analysis
for agentic software engineering**. A coding session is a value stream from
intent to verified change, and TER works up that stream one level at a
time: it **measures** the session (L0), **observes** it as an event stream,
live or recorded (L1), **explains** where effort was wasted and why, with
evidence (L2), **grounds** that explanation in the repository the agent
worked in (L3), then **advises** the developer while the session runs (L4),
**corrects** the session where calibration shows it is safe (L5), and
**learns** from the outcome of every intervention (L6). Each level is useful
on its own, and no level acts on the session before the levels below it have
been shown to be right on real data.

## The maturity model

A level is two things at once ([architecture](architecture.md#maturity-levels)):

- a **build gate**: TER claims a level only when every requirement at that
  level is verified by a passing test, and `ter-req trace --gate LN` checks
  every verified requirement at or below it ([requirements.md](requirements.md));
- a **runtime ceiling**: `ter.domain.Maturity.permits` decides what an
  installation may do. Below L4, TER delivers no intervention at all
  (TER-INT-001, `tests/architecture/test_no_intervention.py`).

| Level | Name | What it means | Gate |
|---|---|---|---|
| L0 | Measured | TER 3 scores reproduced inside the hexagon: event contract, scoring, dated prices | Golden TER 3 scores unchanged; user tokens never scored; dependencies point inward; no intervention below L4 |
| L1 | Observed | The `ter.event` stream is the core boundary; Claude Code hooks feed it live; live analysis equals batch | Hook append under 50 ms p95; redelivery changes nothing; incremental = batch on every corpus session; hook ids equal transcript ids |
| L2 | Explained | Lean model: activity classes, value stream, waste detectors with evidence and published confidence rules, scorecard, A3 | Every finding cites events; uncertain findings never counted; iteration is not rework; A3 countermeasures come from findings only |
| L3 | Grounded | Repository evidence behind the analysis: symbols, imports, tests, call edges, Git, change surface, architecture contracts, context bundles, advisory routing | Evidence only through the `RepositoryEvidence` port; deterministic engines; change surface and boundary findings; bundle precision and recall; routing by role only |
| L4 | Advisory | An intervention engine applies declarative policies to detector signals and advises the developer; every intervention is written to a ledger | Advice changes no agent input; no advice without evidence; cooldowns; every intervention ledgered with its outcome |
| L5 | Corrective | Calibration on labelled data and benchmarks decides which waste categories may be corrected, and corrective actions are opt-in | Annotations, agreement and per-category precision and recall with confidence intervals; benchmark comparisons with confidence intervals |
| L6 | Learning | The loop closes: policies are scored and suspended from ledger outcomes, controlled experiments run, research datasets are published; more than one harness | Experiments assigned by recorded seed; every figure reproducible from recorded data; policies without effect suspended |

## Where TER is now: L0 to L3 met

Every requirement at L0, L1, L2 and L3 is verified by a passing, tagged
test. Counts from `requirements/l*.yaml` on this commit:

| Level | Requirements verified | Planned | Vision points done / partial / not started |
|---|---:|---:|---:|
| L0 Measured | 25 of 25 | 0 | 10 / 1 / 0 |
| L1 Observed | 22 of 22 | 0 | 15 / 1 / 0 |
| L2 Explained | 54 of 54 | 0 | 49 / 5 / 0 |
| L3 Grounded | 34 of 34 | 0 | 25 / 23 / 0 |
| L4 Advisory | 0 of 16 | 16 | 0 / 5 / 34 |
| L5 Corrective | 0 of 10 | 10 | 1 / 4 / 8 |
| L6 Learning | 11 of 19 | 8 | 1 / 4 / 14 |
| **All** | **146 of 180** | **34** | **101 / 43 / 56** of 200 |

The L6 requirements already verified are the ones that did not have to wait:
a second harness (the GARE session source, TER-SRC-010 to TER-SRC-017) and
the stack comparison (TER-STK-010 to TER-STK-013).

**CI gates.** CI runs the L0 gate (golden, contract and architecture
suites), the full suite with 90% branch coverage, `lint-imports` (eight
import contracts), `mypy` on strict `ter` code, `ter-req lint --strict` and
`ter-req points --check`, and `ter-req trace` at `--gate L0`, `L1`, `L2` and `L3`, so main cannot slip
back below L3.

```bash
python -m pytest --req-trace=req-trace.json
ter-req trace --results req-trace.json --gate L3
ter-req report --results req-trace.json
```

### What was checked on real sessions (9 October 2026)

Synthetic tests prove the rules; real sessions show whether the rules are
the right ones. On 9 October the owner ran TER over his own data:

| Check | Data | Result |
|---|---|---|
| Record coverage (TER-SRC-005) | 286 real Claude Code sessions imported through the redacting corpus importer | Every session at 100% of records mapped or classified as documented metadata after the fixes (no session reached 99% before them) |
| Hook ids equal transcript ids (TER-OBS-005, TER-OBS-007) | Three `hooks check` runs on real recordings (Windows) ([summary](../../tests/fixtures/hooks/real-check-2026-10-09.md)) | Every prompt, tool request, tool result and Stop matched (10 of 10, 11 of 11, 15 of 15 Stops); Claude Code's internal helper agents are counted apart |
| Confident L2 detectors | One project's cloud transcripts | 23 confident findings across `repeated_tool_call`, `unnecessary_handoff`, `premature_implementation` and `fragmented_edits`; none true. Each rule was fixed structurally ([l2-explained.md](l2-explained.md#calibration-on-real-sessions)) |
| `fragmented_edits` recount | 286-session private corpus | Shared event ids made parallel edits visible (16 to 72 confident); counting round trips instead of calls brought it back to 17 in 13 sessions. Judged 10 Oct: 15 of 15 sampled were true |
| Grounded detectors | 52 sessions with their repository at the start commit | Session-root rules for worktrees, Windows spellings and harness state (TER-EVD-017 to TER-EVD-020) |
| `unrelated_modification` | 9 confident and 20 sampled uncertain findings | **0 of 29 truly unrelated**: tests, CI, docs and modules the prompt implied but did not name. Capped at 0.60, so every finding is now uncertain and never counted ([l3-grounded.md](l3-grounded.md#calibration-unrelated_modification-9-oct-2026)) |
| Outcomes | 61 labelled sessions | Derived from each session's commits and pull request: 29 merged, 3 submitted, 4 closed unmerged, 1 committed, 24 with no commit (a `git commit -q` prints nothing, so a few may be wrong) |
| Stack comparison | Same 61 sessions | One comparable cell (merged features, svelte+sveltekit against vue), and it compares two repositories as much as two stacks ([l6-stack-comparison.md](l6-stack-comparison.md#first-real-run-9-oct-2026)) |
| Second harness | One real GARE run (local model, one repair) | Translated with token totals equal to GARE's export (TER-SRC-010); no failover yet |

### Calibration lessons

1. **Confident is a claim, and real data disproved most of the first
   claims.** Every confident finding judged so far on real data was false
   before its fix: 23 of 23 in L2, 9 of 9 for `unrelated_modification`. The
   rules were right about the events and wrong about what the events meant.
   A detector earns confidence on judged real sessions, not by construction.
2. **An import graph is not the change surface.** Developers' tasks reach
   tests, CI workflows, docs and modules an issue names. Until the surface
   reads that evidence too, `unrelated_modification` stays a review pointer.
3. **Event identity changes counts.** Giving each parallel tool call its own
   id made 13% of tool requests visible and moved `fragmented_edits` by a
   factor of four. Any change to identity or kinds is followed by a recount
   on the corpus.
4. **Outcomes can be derived, carefully.** Pull requests and commits give a
   usable outcome label without asking the owner for each session, with
   known blind spots.
5. **Observational comparisons are confounded.** With one dominant
   repository per stack, a stack comparison measures repositories. It needs
   more stacks and more repositories per stack (issue #64), and causal
   claims need the controlled experiments of L6.

### What remains unproven on real data

L3 is claimed by the rule, and its engines are deterministic and tested.
What it has not shown on real data:

- **Precision of any detector as a number.** Fixes so far are precision
  fixes from small samples; no per-category precision and recall exist
  (issues #36, #41; TER-ANL-025 now sits at L5).
- A second judge. The first judged sample (10 Oct 2026, 122 findings,
  [l2-explained.md](l2-explained.md#first-judged-sample-10-oct-2026)) found
  39 of 40 confident findings true and tightened four rules, but had one
  judge and at most 15 findings per detector.
- **Change surface and boundary findings** (P063 to P066) and **evidence
  usage** (P067 to P070) on judged real sessions.
- **Context recall** against critical-evidence lists, and bundles against
  full context (P153, P155, P156; issue #42). No lists exist yet.
- **Escalation value**: whether escalating helps and what it costs (P147,
  P149, P150; issue #39). The router is advisory and offline.
- **Priced context inventory** against a billing export (issue #40).
- **Live diff** during a running session (P062) and symbol references other
  than calls (P057).
- **GARE failover** on a real run (issue #55).

Points that need real data are never marked `done` from synthetic tests: 47
points need real data and 3 are verified on it so far.

## L4 Advisory

**Goal.** TER tells the developer, during the session, what it sees and why,
with the evidence, and records every piece of advice and what followed. It
changes nothing the agent reads.

### Capabilities

| Group | In plain words | Requirements |
|---|---|---|
| Intervention engine | A separate engine reads only structured detector signals from the analysis, never raw events; Observe, Analyse, Decide, Intervene, Measure, Learn is the central loop | TER-INT-006, TER-INT-013 |
| Declarative policies | Policies are files stating evidence required, confidence thresholds, cooldowns and permitted actions; loaded as capabilities; a changed policy applies only once a human approved it | TER-INT-008, TER-ARC-008, TER-INT-015 |
| Guards | No advice for a signal that cites only token counts; none for a waste category the calibration policy does not mark eligible; none during a policy's cooldown | TER-INT-007, TER-INT-010, TER-INT-016 |
| Advice | At the L4 ceiling, interventions are advice only; a signal meeting a policy issues the warning it names; every intervention shows its signal and evidence | TER-INT-002, TER-INT-003, TER-INT-014 |
| Ledger | One record per intervention (signal, evidence, action, outcome); after 10 further tool events, the response and before/after flow measures; each policy scored from its outcomes and flagged for recalibration when it does not help | TER-INT-009, TER-INT-011, TER-INT-012 |
| Corrective (opt-in) | Inject evidence or guidance, or propose a catalogued corrective action, only where enabled and thresholds are met | TER-INT-004, TER-INT-005 |

TER-INT-004 and TER-INT-005 sit in the L4 file but change agent input,
which TER-INT-002 forbids at the L4 ceiling. **Recommendation: move them to
L5**, where calibration decides which categories may be corrected. Until
that decision, build them last and only behind a ceiling of L5.

### Entry criteria

- L3 gate in CI (`--gate L3`): done.
- An **eligibility list** for TER-INT-010: only waste categories whose
  confident findings were judged on real sessions with no known false
  positives. Today that means judging the open `fragmented_edits`,
  `regeneration` and `repeated_exploration` findings first. Every
  `unrelated_modification` finding is uncertain and so ineligible by rule.
- A choice of **delivery channel** that is visible to the developer and not
  to the model (a terminal or status line message, `ter watch`, a hook's
  user-facing message), checked against the Claude Code hook fields that
  `test_no_intervention.py` lists.

### Exit gate

All 16 L4 requirements verified (or 14, if TER-INT-004 and TER-INT-005 move
to L5), `--gate L4` in CI, and `test_no_intervention.py` made ceiling-aware:
below L4 every hook still answers `{}`; at L4 only user-facing fields appear,
never `additionalContext`, a decision or `continue`. For points to go
`done`: a ledger from the owner's own sessions showing advice delivered,
suppressed by cooldown and suppressed for ineligible categories.

### Data needed from the owner

- Verdicts on the open confident findings above (a review CSV per detector).
- Continued hook recordings (`python -m ter hook --record`) while advice is
  on, so the ledger's "10 further tool events" window is checked on real
  sessions (issue #35 tooling).
- One or two weeks of the owner's own sessions with advice on, and whether
  each piece of advice was useful (the start of issue #43).
- Approval of the first policy files.

### Risks

- **Steering on wrong findings.** Real data has disproved every confident
  rule judged so far. Advice on an uncertain finding (below 0.70) is never
  given; eligibility is opt-in per category; advice always shows its
  evidence so a developer can dismiss it.
- **Fail-open hooks.** The hook must still never raise, never block and stay
  inside its latency budget. A policy that cannot load, or a ledger that
  cannot write, means no advice, not a failed tool call.
- **Leaking into agent input.** One misplaced hook field turns advice into
  correction. The ceiling check is enforced by an architecture test, not by
  convention.
- **Noise.** Repeated advice is ignored. Cooldowns are part of every policy,
  and the ledger scores each policy so noisy ones are flagged.
- **Privacy.** Ledger records cite event ids and rule names, not prompt or
  file content, so they can be shared like the corpus reports.

### Build order (small PRs)

1. Domain: signal, policy and ledger-record values; pure policy matching
   with the evidence, eligibility and cooldown guards (TER-INT-006,
   TER-INT-007, TER-INT-010, TER-INT-016). Incremental equals batch.
2. Policy files: a JSON policy reader as a capability, with an approval
   record (TER-INT-008, TER-ARC-008, TER-INT-015).
3. `InterventionLedger` port, JSONL adapter, fake and contract suite
   (TER-INT-009).
4. Shadow mode: replay recorded sessions and write what TER *would* have
   advised, with no channel. Run it on the corpus and judge it before any
   live advice.
5. `InterventionChannel` port and the advisory adapter; ceiling-aware
   `test_no_intervention.py` (TER-INT-002, TER-INT-003, TER-INT-014).
6. Ledger follow-up and policy effectiveness (TER-INT-011, TER-INT-012).
7. The control loop wired in bootstrap (TER-INT-013); raise CI to
   `--gate L4`.

## L5 Corrective

**Goal.** Know, with confidence intervals, how often each detector is right,
and allow corrective actions only for the categories where it is.

### Capabilities

| Group | In plain words | Requirements |
|---|---|---|
| Annotation | Store expert labels against event ids; reject a label file that names events the session does not have | TER-CAL-001, TER-EVD-010 |
| Agreement | Inter-rater agreement per waste category: Krippendorff's alpha with a 95% confidence interval | TER-CAL-002, TER-EVD-011 |
| Calibration | Precision and recall with 95% confidence intervals for every waste category | TER-ANL-025 |
| Benchmark | A versioned benchmark corpus (synthetic sessions, controlled tasks with expected outcomes, cleared real sessions); a reporter that gives the same table every run | TER-BEN-001, TER-ANL-030 |
| Comparison | Same task across models, prompting, context and intervention settings; paired differences with confidence intervals | TER-BEN-002, TER-ANL-031 |
| Claims | Every efficiency claim in a report links to the benchmark result behind it | TER-BEN-003 |
| Corrective actions | (if moved from L4) evidence injection and catalogued corrective actions, enabled per category by the calibration policy | TER-INT-004, TER-INT-005 |

### Entry criteria

- L4 claimed and a ledger of real advisory use, so corrections start from
  advice that developers found useful.
- Annotation tooling and an annotation guide agreed (issue #36).

### Exit gate

All L5 requirements verified and `--gate L5` in CI. A category becomes
correctable only when its precision's lower confidence bound clears a
published threshold on annotated real sessions; the calibration policy
records that evidence.

### Data needed from the owner

- **Annotators** (issue #36): two or more people besides the owner, about
  20 sessions, two annotators each and five by all.
- Calibration labels per category (issue #41).
- **Controlled benchmark tasks** with expected outcomes (issue #37), and
  runs across models and prompting strategies (issue #38).
- **Escalation runs** with a cloud provider (issue #39) and a real GARE
  failover (issue #55).
- **Critical-evidence lists** for about 10 sessions (issue #42) and a
  billing export for the same period (issue #40).

### Risks

- Small samples give wide intervals; a category may stay uncorrectable for
  a long time. That is the correct outcome.
- Annotator disagreement can show a category is ill-defined; the fix is to
  the definition (ADR 0004), not the threshold.
- Corrective actions change the session being measured: a corrected session
  is labelled as such so it never mixes with uncorrected baselines.

### Build order (small PRs)

1. Annotation importer and store (TER-CAL-001, TER-EVD-010).
2. Agreement report (TER-CAL-002, TER-EVD-011).
3. Calibration reporter (TER-ANL-025), feeding the eligibility list of
   TER-INT-010.
4. Benchmark corpus format and deterministic reporter (TER-BEN-001,
   TER-ANL-030).
5. Benchmark comparer (TER-BEN-002, TER-ANL-031).
6. Claim links in reports (TER-BEN-003).
7. Corrective actions behind a ceiling of L5 (TER-INT-004, TER-INT-005, if
   moved); raise CI to `--gate L5`.

## L6 Learning

**Goal.** TER learns which interventions work and publishes the evidence:
policies that do not help are suspended, effects are measured in controlled
experiments, and the datasets behind every figure are released.

Already verified at L6: a second harness (GARE, TER-SRC-010 to TER-SRC-017)
and the stack comparison (TER-STK-010 to TER-STK-013).

### Capabilities

| Group | In plain words | Requirements |
|---|---|---|
| Self-correcting policies | A policy with 30 or more interventions and no flow improvement after acceptance is suspended, with the evidence | TER-INT-020 |
| Experiments | Runs assigned to arms from a recorded random seed, recorded before the run starts | TER-ANL-032 |
| Datasets | Value stream dataset; Lean waste study dataset; controlled experiment dataset; expert-labelled sessions with static analysis | TER-RSH-001 to TER-RSH-004 |
| Publication | Each paper ships the Lean model definition and the TER version; one script reproduces every figure from recorded data | TER-RSH-005, TER-EXP-010 |
| More harnesses and stacks | Further session sources through the same contract suite; comparisons across stacks with enough sessions per stratum | TER-SRC-010 (verified), TER-STK-010 to 013 (verified) |

### Entry criteria

- L5 claimed: calibrated categories and a benchmark.
- Enough ledger history for TER-INT-020 to have something to judge (30
  delivered interventions per policy).

### Exit gate

All 19 L6 requirements verified and `--gate L6` in CI. The research points
(issues #44, #45, #46) need the experiments themselves, not only the tools.

### Data needed from the owner

- Controlled experiments with and without TER interventions (issue #45).
- Sessions from more stacks and more repositories per stack, labelled with
  task category and outcome (issue #64).
- Decisions on the research questions and papers (issues #44, #46), and on
  what may be published under which licence.

### Risks

- Before/after measures in the ledger are not causal; only randomised arms
  are (TER-ANL-032).
- Optimising a measure can change what it measures; the outcome verdict
  stays beside the Lean measures and never inside them.
- Publishing data needs consent and redaction beyond the corpus importer's.

### Build order (small PRs)

1. Value stream dataset export (TER-RSH-001), mostly existing analysis.
2. Lean waste study dataset (TER-RSH-002).
3. Policy suspension from the ledger (TER-INT-020).
4. Experiment runner with recorded seeds (TER-ANL-032), then the experiment
   dataset (TER-RSH-003).
5. Expert-labelled dataset with static analysis (TER-RSH-004).
6. Reproduction script and publication bundle (TER-EXP-010, TER-RSH-005);
   raise CI to `--gate L6`.

## Open real-data issues by level

| Issue | What it needs | Blocks |
|---|---|---|
| #34 | Real session corpus and dataset | Done for L1 coverage; research use (L6) |
| #35 | Recorded hook payloads | Done for L1; continues for the L4 ledger |
| #36 | Expert annotation and agreement | L5 (TER-CAL-001, TER-CAL-002) |
| #37 | Controlled benchmark tasks | L5 (TER-BEN-001) |
| #38 | Models and prompting on equivalent tasks | L5 (TER-BEN-002) |
| #39 | Escalation value and cost | L3 points P147, P149, P150; L5 comparisons |
| #40 | Billing export | Priced inventory (P035, P036) |
| #41 | Per-category precision and recall | L4 eligibility; L5 (TER-ANL-025) |
| #42 | Critical-evidence lists, context strategies | L3 points P153, P155, P156; context bundles as advice at L4 |
| #43 | Intervention acceptance from the ledger | L4 points, L6 |
| #44 | Research questions | L6 |
| #45 | Experiments with and without TER | L6 (TER-RSH-003) |
| #46 | Papers | L6 (TER-RSH-005) |
| #52, #55 | GARE access and a real failover run | L6 points P103, P105 |
| #64 | More stacks and repositories | L6 points P182, P185 |

Tracker: #47.

## Principles that hold at every level

- **Hexagon.** Dependencies point inward; the domain is pure; vendors and IO
  live in adapters loaded as capabilities
  ([ADR 0001](../decisions/0001-hexagonal-strangler-rebuild.md),
  [ADR 0005](../decisions/0005-admitting-external-capabilities.md)).
  Interventions get ports (`InterventionChannel`, `InterventionLedger`) like
  everything else.
- **EARS, traced to tests.** Every behaviour is a requirement; a level is
  claimed only when every requirement at it has a passing tagged test, and
  CI gates it ([requirements.md](requirements.md)).
- **Batch equals incremental.** Analysis is a fold over `ter.event`, O(1)
  amortised and idempotent by event id; the intervention engine is a fold
  too ([l1-observed.md](l1-observed.md)).
- **Never count, or act on, uncertain findings.** Below 0.70 a finding is a
  pointer for review: not counted as avoidable, not eligible for advice,
  never corrected ([ADR 0004](../decisions/0004-lean-waste-model.md)).
- **Real data before `done`.** Synthetic tests prove rules; a point that
  needs real data stays `partial` until real sessions confirm it
  ([points.md](points.md)).
- **Fail open.** Hooks never raise and never block; a broken policy or
  ledger means no intervention.
- **Behaviour apart from outcome.** The outcome verdict is shown beside the
  measures and never read by them ([outcome.md](outcome.md)).
- **Prices are data; scoring changes are visible.** Rates come through the
  `PriceBook` port ([ADR 0003](../decisions/0003-price-book-as-data.md));
  any scoring change shows as a golden snapshot diff.
