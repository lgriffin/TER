# Control charts

Point P201, requirements TER-SPC-001 to TER-SPC-013 (L2) and TER-SPC-020
(L4, planned).

Lean separates **routine variation**, the noise every session of a stable
way of working shows, from **special causes** worth a look. TER charts each
session measure on an individuals and moving range (XmR) chart, the Lean
chart for one value per unit of work. The limits come from your own
sessions, not from a target. You can tune them, and each tuned limit keeps
its reason. At L4 the same signals become firing rules for advice.

## Quick start

```bash
# 1. Measure: explain every session and keep only its measures (slow; run it
#    where the sessions are). The output has no prompt, path or code.
python -m ter control measure ~/ter-data/corpus --out measures.json --pseudonymise

# 2. Limits: natural process limits from those sessions.
python -m ter control limits measures.json --out limits.json

# 3. Chart: XmR charts and every signal.
python -m ter control chart measures.json --limits limits.json --html control.html
```

`measure` reads session files, folders of them, or a corpus written by
`python -m ter corpus import` (its `sessions/` folder; subagent transcripts
belong to their parent session and are skipped). It exits 1 when a session
could not be read, after measuring the rest.

## The arithmetic

For each measure, over the baseline sessions in process order (by start
time):

| Line | Average moving range (default) | Median moving range (`--method median`) |
|---|---|---|
| Centre line (CL) | mean | mean |
| Upper and lower natural limits | CL ± 2.66 × mR̄ | CL ± 3.145 × median mR |
| Sigma, for the zone rules | mR̄ / 1.128 | median mR / 0.954 |
| Moving range upper limit (URL) | 3.268 × mR̄ | 3.865 × median mR |

The median method suits skewed measures, because one wild session cannot
inflate its limits. When most successive sessions are equal, as with a count
that is usually 0, the median moving range is 0 and would collapse the
limits onto the centre. That measure then uses the average moving range,
and its `natural.method` in the limits file says so (TER-SPC-014). A limit past a measure's natural boundary is reported
as no limit. Examples are a lower limit below 0 for a count, or an upper
limit above 100% for a ratio. TER computes limits only from 8 or more
sessions with a known value (TER-SPC-002). Below 20 sessions they are
marked **provisional** (TER-SPC-012).

## Detection rules

Each rule has a published name, and each can be switched on or off per
measure (TER-SPC-003):

| Rule | Signal |
|---|---|
| `beyond_limits` | One session outside a control limit |
| `two_of_three` | Two of three successive sessions more than 2 sigma from the centre line, on the same side |
| `four_of_five` | Four of five successive sessions more than 1 sigma from the centre line, on the same side |
| `run_of_eight` | Eight successive sessions on the same side of the centre line |

Every signal names the sessions that form the pattern as its evidence.
Where a boundary removed a natural limit, the zone and run rules are not
applied on that side. A count that is mostly 0 says nothing about special
causes below its mean.

## Measures

| Measure | Basis | Worse when |
|---|---|---|
| `avoidable_share` | ratio | higher |
| `unverified_waste_share` (avoidable plus uncertain) | ratio | higher |
| `value_adding_share` | ratio | lower |
| `flow_efficiency_tokens`, `flow_efficiency_time` | ratio | lower |
| `edits_validated_share` | ratio | lower |
| `ter` (when TER was scored) | ratio | lower |
| `confident_findings`, `uncertain_findings` | count | higher |
| `rework_cycles`, `risk_findings`, `wip_peak` | count | higher |
| `unvalidated_edits_at_end`, `unresolved_failures_at_end` | count | higher |
| `generated_tokens` | tokens | higher (never fires) |
| `agent_seconds` | seconds | higher (never fires) |

A count that is almost always 0, such as `confident_findings` on a
healthy corpus, is a rare event. Its XmR limits are wide and a single
session above them is the signal worth reading.

Uncertain findings count as waste until they are verified, so the process is
also charted with them (`unverified_waste_share`, `uncertain_findings`).
Measures come from the Lean analysis only, never from the outcome verdict
(point 5).

## Tuning limits

`limits.json` (`ter.control-limits/1`) is meant to be edited. Each measure
looks like this:

```json
"avoidable_share": {
  "enabled": true,
  "rules": ["beyond_limits", "two_of_three", "four_of_five", "run_of_eight"],
  "natural": {"centre": 0.0973, "ucl": 0.3487, "lcl": null, "sigma": 0.0945, "...": "..."},
  "tuned": {"ucl": 0.2, "lcl": null, "reason": "team target after the retro"}
}
```

- **`tuned`** sets action limits. They replace the natural limits for
  `beyond_limits` only. The zone and run rules keep the natural centre and
  sigma, because those describe the process, not a target. A tuned limit
  needs a `reason`. A tuned `lcl` must be below `ucl`, and a ratio must stay
  between 0 and 1. Otherwise the whole file is rejected (TER-SPC-005).
- **`rules`** lists the rules that apply to the measure.
- **`enabled: false`** keeps the measure charted but raises no signals.

When more sessions arrive, recompute the natural limits and keep your edits
(TER-SPC-008):

```bash
python -m ter control limits measures.json --out limits.json --keep limits.json
```

## Firing

A signal **fires** only when both of these hold (TER-SPC-006):

- it is unfavourable: it moves the way the measure gets worse;
- its measure is a ratio or a count.

Token and time totals are charted but never fire, because no intervention
may rest on a token count alone (TER-INT-007). Each signal has a stable id,
`control.<measure>.<rule>` (for example `control.rework_cycles.beyond_limits`).
At L4 an approved policy will name that id to advise on it (TER-SPC-020,
planned).

## Stale limits

Limits record the detector set they were computed with, a short hash of
every detector and its confidence rule. Recalibrating a detector changes
what a finding count means. Charting sessions from another detector set
warns that the limits are stale (TER-SPC-007), and TER refuses to mix
detector sets in one chart (TER-SPC-013). Re-measure and recompute after a
detector change.

## On the A3

Give `ter a3` the limits file and the A3 places that session against them
(TER-SPC-011):

```bash
python -m ter a3 session.jsonl --limits limits.json --html a3.html
```

The **Process control** section lists the measures outside a limit first,
each linked to the findings that move it. For example, `rework_cycles` links
the rework findings, and `unvalidated_edits_at_end` links the
`unvalidated_implementation` findings. Shares, flow and totals link every
waste finding. Only `beyond_limits` applies to one session. The JSON carries
the same placements under `process_control`.

## Next

- Limits from the owner's real corpus, with the signals reviewed, before
  P201 is done.
