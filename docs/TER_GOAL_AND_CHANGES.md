# TER: end goal, what changed, and proposal to maintainers

## End goal (unchanged)

**TER Calculator** exists to make **Claude Code sessions** *measurable*: output aligned vs waste, session economics (cost, cache, context growth), input-side signals (redundancy, drift, alignment), and grouped parent/subagent runs — so teams can **change behavior** (prompts, hooks, session length) with numbers that are **as honest as we can make them** given heuristics and API usage fields.

Prompt drafting tools are **out of scope** for this repository; they live separately (see `../prompt-glow/`).

---

## Already documented in `UPDATES.md` (measurement pass)

The table in [UPDATES.md](../UPDATES.md) describes a focused pass on:

- **`source_role` on spans** — align “waste \$” with **assistant** generation vs **user/context** (e.g. tool results).
- **Calibration** of waste dollars to **`output_tokens`** when present — `waste_output_calibration_ratio` exposes mismatch.
- **Output- vs input-priced** waste rows — stop pricing re-injected context at output \$/MTok.
- **Overlap rules** — avoid double-counting classifier vs pattern buckets.
- **CLI thresholds** — `--confidence-threshold` / `--similarity-threshold` actually drive the classifier.

**Why it matters:** Without this, headline metrics could be **directionally useful** but **wrong on money and attribution** — which undermines trust and any “we saved \$X” narrative.

---

## Changes in this evolution (repository layout + product)

### 1. `ter report` — human-facing summary

- **What:** `ter report <session.jsonl>` runs the **same pipeline** as `analyze` and prints **Markdown** (or writes it with **`-o FILE`**, e.g. `report.md`): headline TER, waste %, cost, **calibration ratio**, cache, positional TER, top structural patterns, **action bullets**.
- **Why:** Closes the loop from “JSON for scripts” to **action** (UPDATES “future idea #1” — implemented in minimal form).
- **End goal:** Faster **human-in-the-loop** improvement without reading full Rich/JSON output.

### 2. `ter compare --baseline`

- **What:** Exactly **two** session files → Markdown **delta table** (TER, waste ratio, cost) + short interpretation note.
- **Why:** Supports **before/after** narratives (rules change, prompt template change). Uses **default analyze thresholds** unless we later extend the CLI (documented).
- **End goal:** Comparable **A/B** storytelling for process changes.

### 3. Shared pipeline — `analyze_pipeline.py` + `config_parse.py`

- **What:** Single `analyze_session(args)` used by `analyze` and `report`; cost/phase parsing centralized.
- **Why:** Less drift between commands; easier testing.
- **End goal:** **Consistency** of measurement across entry points.

### 4. Trust layer — README “Limits of interpretation”

- **What:** Explicit limits: heuristic spans, `len(text)//4`, calibration meaning, TER ≠ ground truth labels.
- **Why:** Stops over-claiming; aligns user expectations with epistemics (per prior review).
- **End goal:** **Credible** use in serious retros, not only demos.

---

## Not done yet (recommended next)

| Item | Rationale |
|------|-----------|
| **Gold set + precision/recall** | Thresholds are still intuition-led. |
| **Lite mode (no embeddings)** | CI / hooks; cold-start (UPDATES #2). |
| **Wire analyze flags into `--baseline`** | Today baseline uses `default_analyze_args()` only. |
| **`ter report` from grouped runs** | Parent + subagents as one Markdown report. |

---

## Maintainer decisions (recorded)

| Topic | Decision |
|--------|----------|
| **Versioning** | **No semver bump required** for this work alone; ship on a **branch** as needed (no obligation to jump to `0.2.0`). |
| **`ter report` output** | **`-o` / `--output FILE`** supported (e.g. `report.md`); default remains stdout. |
| **`ter compare --baseline` flags** | **Undecided** — compare currently uses **default analyze thresholds** only; wiring full `analyze` flags is optional follow-up. |

---

*This file is the “proposal + changelog” requested to align maintainers on scope and direction.*
