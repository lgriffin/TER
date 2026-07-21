"""Transparent, opt-in per-repository policy threshold tuning."""

from __future__ import annotations
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from .intervention_policy import PolicyConfig


def recommend_policy_config(
    effectiveness: dict, current: PolicyConfig, *, min_sample_size: int = 8
) -> PolicyConfig:
    cfg = current
    replan = effectiveness.get("replan", {})
    if int(replan.get("issued", 0)) >= min_sample_size:
        if (
            float(replan.get("improvement_rate", 0)) >= 0.7
            and float(replan.get("override_rate", 0)) <= 0.2
        ):
            cfg = replace(
                cfg,
                ter_drop_replan=max(0.10, current.ter_drop_replan * 0.9),
                waste_ratio_replan=max(0.20, current.waste_ratio_replan * 0.9),
            )
    refresh = effectiveness.get("refresh_context", {})
    if int(refresh.get("issued", 0)) >= min_sample_size:
        if (
            float(refresh.get("improvement_rate", 0)) <= 0.3
            or float(refresh.get("override_rate", 0)) >= 0.5
        ):
            cfg = replace(
                cfg,
                ter_drop_warning=min(0.50, cfg.ter_drop_warning * 1.1),
                waste_ratio_warning=min(0.70, cfg.waste_ratio_warning * 1.1),
            )
    return cfg


def describe_config_changes(
    current: PolicyConfig, recommended: PolicyConfig, effectiveness: dict
) -> list[dict[str, object]]:
    """Describe threshold changes with the evidence that motivated them."""
    changes: list[dict[str, object]] = []
    field_to_kind = {
        "ter_drop_replan": "replan",
        "waste_ratio_replan": "replan",
        "ter_drop_warning": "refresh_context",
        "waste_ratio_warning": "refresh_context",
    }
    for field, kind in field_to_kind.items():
        old = getattr(current, field)
        new = getattr(recommended, field)
        if old == new:
            continue
        metrics = effectiveness.get(kind, {})
        issued = int(metrics.get("issued", 0))
        improvement = float(metrics.get("improvement_rate", 0.0))
        override = float(metrics.get("override_rate", 0.0))
        direction = "more sensitive" if new < old else "less sensitive"
        changes.append(
            {
                "field": field,
                "old_value": old,
                "new_value": new,
                "intervention_type": kind,
                "sample_size": issued,
                "improvement_rate": improvement,
                "override_rate": override,
                "reason": (
                    f"{direction}; {issued} outcomes, "
                    f"{improvement:.0%} improved, {override:.0%} overridden"
                ),
            }
        )
    return changes


def save_tuned_policy_config(root: Path, config: PolicyConfig) -> None:
    path = root / ".ter" / "tuned-policy-config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if path.exists():
        try:
            previous = json.loads(path.read_text()).get("config")
        except Exception:
            previous = None
    payload = {"config": asdict(config), "previous": previous}
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    with os.fdopen(fd, "w") as h:
        json.dump(payload, h, indent=2, sort_keys=True)
        h.write("\n")
    os.replace(tmp, path)


def load_tuned_policy_config(root: Path) -> PolicyConfig | None:
    path = root / ".ter" / "tuned-policy-config.json"
    if not path.exists():
        return None
    try:
        return PolicyConfig(**json.loads(path.read_text())["config"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
