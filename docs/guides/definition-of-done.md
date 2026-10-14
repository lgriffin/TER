# Vision points and the definition of done

TER 4 is steered by Leigh's 200-point vision. Every point has a definition of
done, the EARS requirements (rules) that enforce it, and how it is verified.
This guide explains how to find your way around the points, what each status
means, when a point is done, and what every change must record.

| File | Role |
|---|---|
| [requirements/points.yaml](../../requirements/points.yaml) | The source: all 200 points (and any contributed ones past P200), hand-edited |
| [docs/ter4/points.md](../ter4/points.md) | The generated index; never edit it by hand |
| `requirements/l*.yaml` | The EARS rules that points cite (see the [EARS guide](ears.md)) |

## The shape of a point

```yaml
  - id: P119
    text: >-
      Ensure duplicate events do not distort analysis.   # the vision, verbatim
    level: L1                  # the maturity level it belongs to
    kind: capability           # capability | principle | research
    status: done               # done | partial | not-started
    definition_of_done:        # 1 to 3 checkable statements
      - Re-delivered events are discarded without changing analysis state.
    rules: [TER-OBS-004]       # EARS requirement ids that enforce it
    verification:
      - 'test: tests/contract/test_event_ingest.py::test_redelivery_changes_nothing'
```

Points that need real session data also carry:

```yaml
    issue: 41                  # the GitHub issue collecting the data (#34 to #46)
    real_data: true            # blocks status done ...
    # real_data_verified: true # ... until this is set, when the issue closes
```

### Points from other sources

P001 to P200 are Leigh's list and keep its text verbatim. When an external
capability (GARE is the first, see
[ADR 0005](../decisions/0005-admitting-external-capabilities.md)) brings a
goal those points do not already state, it becomes a **contributed point**,
numbered past the last one (P201, P202, …), with an `origin`:

```yaml
  - id: P201
    text: Read the usage records GARE writes, by schema name.
    origin: {source: GARE, ref: "docs/ter-integration.md#usage", author: "Leigh Griffin"}
    # level, kind, status, definition_of_done, rules, verification as above
```

`source` names the project, `ref` the document, path or commit the point is
taken from, and `author` who wrote it; all three are required. Prefer citing
an existing point over adding one: a contributed point is for a goal Leigh's
list does not cover. A contributed point follows every other rule here (done
criteria, two-way links, statuses, the real-data rule), and its rules follow
the [capability pack](contributing.md#adding-a-capability-pack) workflow.

**Kinds.** 136 *capabilities* (something TER does), 37 *principles* (a rule the
design keeps, usually enforced by an architectural check) and 27 *research*
points (a question answered with data, enforced by a protocol requirement such
as "TER shall produce dataset X with fields Y").

## Navigating

[points.md](../ter4/points.md) opens with totals and a progress bar, then:

- **By level**: done, partial and not started per maturity level.
- **Origins**: how many points are Leigh's and how many each source
  contributed.
- **Map**: a 10 × 20 grid (longer once points are contributed), one symbol per point (`●` done, `◐` partial,
  `○` not started), rows starting at P001, P021, … so you can see which parts
  of the vision are moving.
- **Points**: one row per point with its origin (`vision` for P001 to P200,
  otherwise the source, reference and author), level, status, issue link,
  definition of done, rules (`✓` verified, `·` planned) and verification.

To find a point, search points.md for a word from the vision (`WIP`,
`handoff`, `precision`) or its id. To go from code to points, look up the
requirement a test cites in `requirements/*.yaml` and read its
`source_points`. To list points programmatically:

```bash
python -c "import yaml; [print(p['id'], p['status'], p['text'][:70]) for p in yaml.safe_load(open('requirements/points.yaml'))['points'] if p['level'] == 'L2' and p['status'] != 'done']"
```

## Statuses

| Status | Means | Typical verification |
|---|---|---|
| `not-started` | Nothing built yet; the rules exist as `planned` requirements | `planned: <how it will be verified>` |
| `partial` | Some of the definition of done holds, proved by tests; the rest is named | a `test:` entry for what exists and a `planned:` entry for what does not |
| `done` | Every definition-of-done statement holds and is proved by a check | `test:` or `ci:` entries |

### Verification entries

| Form | Meaning | What the lint checks |
|---|---|---|
| `test: <pytest node id>` | a test on this branch proves it | the file and every `::` name exist |
| `ci: <step name>` | a CI step proves it | a step with that exact name exists in `.github/workflows` |
| `branch: <branch> <what>` | proof exists on an unmerged branch | warning until merged (`--strict` fails) |
| `planned: <what>` | how it will be verified | nothing |

## When is a point done?

A point is `done` when **every statement in its definition of done is true
and a check proves it**. In practice:

1. Read each definition-of-done statement and ask: is this true on this
   branch, and which test or CI step would fail if it stopped being true?
2. Every rule the point lists is `verified`, or the point names an existing
   `test:` or `ci:` check that covers the whole definition of done.
3. If the point needs real session data, the data exists and its issue is
   closed (see below).

The lint enforces a floor: a `done` point must have all its rules verified or
name a `test:`/`ci:` check that exists. Passing the lint is necessary, not
sufficient. If one statement of the definition of done is not yet true, the
point is `partial`, and a `planned:` entry says what remains.

Examples from the catalogue:

- **P119** (duplicate events do not distort analysis) is done: its one rule,
  TER-OBS-004, is verified by contract and property tests.
- **P063** (activity stays within the expected change surface) is partial
  although its rule, TER-EVD-006, is verified: the detector was run on the
  owner's real sessions, and the judged findings showed that the import
  graph misses tests, CI, docs and modules a prompt implies, so the point
  waits on a judged corpus with evidence beyond imports.
- **P155** (context recall when critical evidence was needed) is partial:
  TER-EVD-005 is verified on synthetic sessions, and no critical-evidence
  lists for real sessions exist yet (issue #42).

### The real-data rule

46 points make claims that synthetic sessions cannot prove: precision on real
agent behaviour, inter-rater agreement, benchmarks, experiments, papers.
Synthetic sessions are written to trigger each pattern, so passing tests on
them say nothing about real sessions. These points:

- carry `issue:` (the GitHub issue that collects the data) and
  `real_data: true`;
- **are never marked done on synthetic tests.** The lint rejects `done`
  (`POINT-REAL-DATA`) until `real_data_verified: true` is set, and that flag
  is set only in the change that closes the issue with real data (the lint
  cannot query GitHub, so the flag records the closure).

| Issue | Data needed | Points |
|---|---|---|
| [#34](https://github.com/lgriffin/TER/issues/34) | Real Claude Code session corpus and research dataset | P087, P092, P095, P181 |
| [#35](https://github.com/lgriffin/TER/issues/35) | Recorded hook payloads from live sessions (Stop, SubagentStop, correlation) | P115, P116, P117 |
| [#36](https://github.com/lgriffin/TER/issues/36) | Expert annotation of waste and inter-rater agreement | P088, P089 |
| [#37](https://github.com/lgriffin/TER/issues/37) | Controlled task benchmark with known expected outcomes | P094 |
| [#38](https://github.com/lgriffin/TER/issues/38) | Models and prompting strategies on equivalent tasks | P096, P097 |
| [#39](https://github.com/lgriffin/TER/issues/39) | Whether escalation to a stronger model helps, and its cost | P147, P149, P150 |
| [#40](https://github.com/lgriffin/TER/issues/40) | Real pricing data for context inventory cost | P035, P036 |
| [#41](https://github.com/lgriffin/TER/issues/41) | Per-category precision and recall calibration of waste detectors | P090, P091, P100 |
| [#42](https://github.com/lgriffin/TER/issues/42) | Context strategy comparison and context recall ground truth | P098, P153, P155, P156, P186 |
| [#43](https://github.com/lgriffin/TER/issues/43) | Intervention acceptance and effectiveness from the ledger | P167, P168, P169, P170, P179 |
| [#44](https://github.com/lgriffin/TER/issues/44) | Research questions on Lean wastes, WIP, exploration and outcomes | P033, P182, P183, P184, P185, P187, P188 |
| [#45](https://github.com/lgriffin/TER/issues/45) | Controlled experiments with and without TER intervention | P099, P189, P190, P200 |
| [#46](https://github.com/lgriffin/TER/issues/46) | The three papers and the operationalisation claim | P191, P192, P193, P194, P199 |

[#47](https://github.com/lgriffin/TER/issues/47) tracks them all. The table
reflects `points.yaml` today; the `Issue` column of points.md is always
current.

## What the lint enforces

`ter-req lint` checks the points alongside the requirements:

| Code | Rule |
|---|---|
| `POINT-MISSING`, `POINT-DUPLICATE` | P001 to P200 each appear exactly once; no point id repeats |
| `POINT-ORIGIN` | P001 to P200 record no origin; every point past P200 records one with `source`, `ref` and `author` (TER-REQ-009, TER-REQ-010) |
| `POINT-UNKNOWN` | every point a requirement cites in `source_points` is catalogued (TER-REQ-012) |
| `POINT-DOD` | 1 to 3 non-empty definition-of-done statements |
| `POINT-RULES`, `POINT-VERIFY` | at least one rule and one verification entry |
| `POINT-RULE-UNKNOWN` | every rule id exists in the catalogue |
| `POINT-LINK` | links go both ways: a point lists R if and only if R's `source_points` contains it |
| `REQ-ORPHAN` | every requirement serves at least one point |
| `POINT-CHECK-MISSING` | every `test:` and `ci:` entry names something that exists |
| `POINT-DONE` | a done point has all rules verified or an existing check |
| `POINT-PENDING` | (warning) done only on another branch |
| `POINT-REAL-DATA`, `POINT-REAL-DATA-FLAG` | real-data points follow the rule above; `issue` implies `real_data: true` |

## The working rule for every change

1. **Name the point ids** the change advances (`P044`, `P102`, …) in the
   commit message and the PR body.
2. **Update those points** in `requirements/points.yaml` in the same PR:
   `status` and `verification`, honestly. Advancing a point often means adding
   a `test:` entry and leaving it `partial`.
3. **Regenerate the index** and check it:

   ```bash
   ter-req points
   ter-req points --check
   ter-req lint --strict --tests tests
   ```

   CI runs `ter-req points --check` and fails when points.md is stale;
   `tests/docs/test_docs.py` checks the same thing in the test suite.

4. **Write the test before the code**: `python dev.py red TEST` before the
   change, `python dev.py green TEST` after it (the
   [testing guide](testing.md#test-first-workflow) has the workflow). A
   `test:` entry is evidence only if that test could fail.
5. **Never mark a real-data point done** from synthetic tests.
6. **Number a new goal past P200 with its origin**; never edit the text of
   P001 to P200 or give them an origin.

A commit message that follows the rule:

```text
Add unused-context cost to the scorecard

- Cost unused reads as context tokens (TER-DET-004 stays planned: the cost
  is not yet asserted on real sessions).

Points: P034 (verification adds the scorecard test), P035 stays partial
(real pricing data, issue #40).
```
