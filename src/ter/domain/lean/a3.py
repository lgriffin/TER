"""The A3 view-model: one page, in the order of a Toyota A3 report.

Background (the developer's intent) → Current state (value stream map) →
Analysis (waste Pareto, activity classes, flow) → Root causes (findings with
their evidence) → Countermeasures → Follow-up. Renderers read this model and
nothing else, so the HTML and the JSON of one report always agree.

When outcome evidence was supplied, the A3 also carries the outcome verdict
(``ter.domain.outcome``) beside the scorecard. The verdict is judged
separately and no behaviour measure reads it (point 5); the ``outcome`` key
appears in the JSON only when there is a verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..costing import Prices, SessionCost, price_session
from ..outcome import OutcomeVerdict, per_verified_outcome
from .analysis import LeanAnalysis, apportion
from .countermeasures import Countermeasure, FollowUp, build_countermeasures, follow_ups
from .inventory import ContextInventory, context_inventory
from .concepts import LEAN_MEASURES
from .model import ActivityClass, Finding, FindingKind, LeanWaste
from .scorecard import (
    ScorecardDimension,
    SoftwareValueEfficiency,
    scorecard_dimensions,
    software_value_efficiency,
)

__all__ = ["A3_SCHEMA", "A3Report", "ParetoBar", "build_a3"]

A3_SCHEMA = "ter.a3/0.1"

#: Root causes shown on the page; the JSON keeps every finding.
ROOT_CAUSES_SHOWN = 6


@dataclass(frozen=True)
class ParetoBar:
    waste: LeanWaste
    tokens: int
    findings: int

    def as_dict(self) -> dict[str, object]:
        return {
            "waste": self.waste.value,
            "tokens": self.tokens,
            "findings": self.findings,
        }


@dataclass(frozen=True)
class A3Report:
    title: str
    session_id: str | None
    intents: tuple[str, ...]
    problem: str
    analysis: LeanAnalysis
    pareto: tuple[ParetoBar, ...]
    root_causes: tuple[Finding, ...]
    countermeasures: tuple[Countermeasure, ...]
    follow_up: tuple[FollowUp, ...]
    outcome: OutcomeVerdict | None = None
    #: What the session's source cannot report (``SessionTrace.usage_limits``).
    usage_limits: tuple[str, ...] = ()
    #: Unused and re-read context, in tokens (TER-DET-004).
    inventory: ContextInventory | None = None
    #: The session and its inventory priced from a dated price book, when one
    #: was supplied (TER-ANL-040, TER-ANL-041).
    cost: SessionCost | None = None

    @property
    def tokens_per_verified_outcome(self) -> float | None:
        """Generated tokens per accepted outcome; None without an accepted one."""
        if self.outcome is None:
            return None
        generated = self.analysis.scorecard.generated_tokens
        return per_verified_outcome(generated, (self.outcome,))

    @property
    def value_efficiency(self) -> SoftwareValueEfficiency:
        """Software Value Efficiency, reported next to TER (TER-SCR-003)."""
        return software_value_efficiency(self.analysis.scorecard, self.outcome)

    @property
    def dimensions(self) -> tuple[ScorecardDimension, ...]:
        """The six scorecard dimensions (TER-SCR-001)."""
        return scorecard_dimensions(self.analysis, self.value_efficiency, self.outcome)

    def scorecard_dict(self) -> dict[str, object]:
        """The behaviour scorecard with Software Value Efficiency right after TER."""
        out: dict[str, object] = {}
        for key, value in self.analysis.scorecard.as_dict().items():
            out[key] = value
            if key == "ter":
                out["software_value_efficiency"] = self.value_efficiency.as_dict()
        return out

    def as_dict(self) -> dict[str, object]:
        a = self.analysis
        out: dict[str, object] = {
            "schema": A3_SCHEMA,
            "title": self.title,
            "session_id": self.session_id,
            "background": {
                "intents": list(self.intents),
                "events": a.events,
                "intent": a.intent.as_dict([f.id for f in a.drift_findings]),
                # Languages and stack (TER-STK-001, TER-STK-002).
                "profile": a.profile.as_dict(),
                # Only a grounded (L3) A3: the repository engine and every
                # directory the session named it by (TER-EVD-017).
                **(
                    {
                        "repository": {
                            "engine": a.repository.engine,
                            "roots": list(a.repository.roots),
                        }
                    }
                    if a.repository is not None
                    else {}
                ),
            },
            "problem": self.problem,
            "current_state": {"value_stream": [s.as_dict() for s in a.value_stream]},
            "analysis": {
                "scorecard": self.scorecard_dict(),
                "dimensions": [d.as_dict() for d in self.dimensions],
                "wip": a.wip.as_dict(),
                "pareto": [p.as_dict() for p in self.pareto],
                "cycles": [c.as_dict() for c in a.cycles],
                # Only a grounded (L3) A3: evidence usage, outcome value and
                # the role-typed evidence graph (TER-EVD-008, TER-LEN-009,
                # TER-GRF-001).
                **(
                    {"evidence_usage": a.usage.as_dict()} if a.usage is not None else {}
                ),
                **({"outcome_value": a.value.as_dict()} if a.value is not None else {}),
                **(
                    {"evidence_graph": a.grounded_graph.as_dict()}
                    if a.grounded_graph is not None
                    else {}
                ),
            },
            "root_causes": [f.as_dict() for f in self.root_causes],
            "findings": [f.as_dict() for f in a.findings],
            "countermeasures": [c.as_dict() for c in self.countermeasures],
            "follow_up": [f.as_dict() for f in self.follow_up],
            "detectors": [
                {"id": i, "waste": w, "kind": k, "confidence_rule": r}
                for i, w, k, r in a.detectors
            ],
            "lean_concepts": [
                {"concept": c.value, "measures": [m.as_dict() for m in ms]}
                for c, ms in LEAN_MEASURES.items()
            ],
        }
        if self.usage_limits:
            out["usage_limits"] = list(self.usage_limits)
        if self.inventory is not None:
            out["context_inventory"] = self.inventory.as_dict()
        if self.cost is not None:
            out["cost"] = self.cost.as_dict()
        if self.outcome is not None:
            per = self.tokens_per_verified_outcome
            out["outcome"] = {
                **self.outcome.as_dict(),
                "generated_tokens_per_verified_outcome": None
                if per is None
                else round(per, 1),
            }
        return out


def _title(intents: Sequence[str]) -> str:
    if not intents:
        return "Agent session"
    first = " ".join(intents[0].split())
    return first if len(first) <= 120 else first[:119].rstrip() + "…"


def _problem(analysis: LeanAnalysis) -> str:
    sc = analysis.scorecard
    if sc.generated_tokens == 0:
        return "The session generated no agent activity to analyse."
    avoidable = sc.waste_tokens / sc.generated_tokens
    parts = [
        f"{avoidable:.0%} of the {sc.generated_tokens:,} tokens the agent generated went to "
        f"avoidable work across {sc.findings} finding(s)"
    ]
    if sc.uncertain_findings:
        parts.append(
            f"{sc.uncertain_findings} of those finding(s), covering "
            f"{sc.uncertain_waste_tokens:,} tokens, are uncertain and count as "
            "waste until verified"
        )
    if sc.flow_efficiency_tokens is not None:
        flow = f"flow efficiency is {sc.flow_efficiency_tokens:.0%} of tokens"
        if sc.flow_efficiency_time is not None:
            flow += f" and {sc.flow_efficiency_time:.0%} of agent time"
        parts.append(flow)
    if sc.risks:
        parts.append(f"{sc.risks} risk(s) to the outcome were flagged")
    return "; ".join(parts) + "."


def _pareto(analysis: LeanAnalysis) -> tuple[ParetoBar, ...]:
    """Generated waste tokens by waste type, reconciling with the scorecard.

    Each event's avoidable tokens count once, under the finding the scorecard
    charged them to, so the bars add up to ``scorecard.waste_tokens``.
    """
    allocated = analysis.allocated_waste_tokens()
    parts: dict[str, float] = {}
    counts: dict[LeanWaste, int] = {}
    for f in analysis.findings:
        if f.kind is not FindingKind.WASTE:
            continue
        parts[f.waste.value] = parts.get(f.waste.value, 0.0) + allocated.get(f.id, 0.0)
        counts[f.waste] = counts.get(f.waste, 0) + 1
    rounded = apportion(parts, analysis.scorecard.waste_tokens)
    tokens = {w: rounded[w.value] for w in counts}
    return tuple(
        ParetoBar(w, tokens[w], counts[w])
        for w in sorted(tokens, key=lambda w: (-tokens[w], w.value))
    )


def build_a3(
    analysis: LeanAnalysis,
    intents: Sequence[str] = (),
    outcome: OutcomeVerdict | None = None,
    usage_limits: Sequence[str] = (),
    prices: Prices | None = None,
) -> A3Report:
    """Assemble the A3 from an analysis, the developer's prompts and, when
    known, the outcome verdict (shown beside the analysis, never read by it).
    ``usage_limits`` are the source's, stated beside the figures they qualify.
    With ``prices`` the session and its context inventory are priced at the
    prices in force on the session date."""
    findings = analysis.findings
    ranked = sorted(
        findings,
        key=lambda f: (
            f.uncertain,
            -(f.tokens + f.context_tokens),
            -f.confidence,
            f.id,
        ),
    )
    sc = analysis.scorecard
    inventory = context_inventory(analysis.steps, findings)
    return A3Report(
        title=_title(intents),
        session_id=analysis.session_id,
        intents=tuple(intents),
        problem=_problem(analysis),
        analysis=analysis,
        pareto=_pareto(analysis),
        root_causes=tuple(ranked[:ROOT_CAUSES_SHOWN]),
        countermeasures=build_countermeasures(
            findings, analysis.steps, analysis.allocated_waste_tokens()
        ),
        follow_up=follow_ups(
            findings,
            flow_efficiency=sc.flow_efficiency_tokens,
            avoidable_share=sc.activity_share(ActivityClass.AVOIDABLE),
        ),
        outcome=outcome,
        usage_limits=tuple(usage_limits),
        inventory=inventory,
        cost=None
        if prices is None
        else price_session(analysis.steps, inventory, prices),
    )
