from __future__ import annotations

import json
from pathlib import Path

import pytest

from ter_calculator.adaptive_optimizer import (
    learn_policy,
    personalize_policy,
    save_policy,
)
from ter_calculator.cli import main
from ter_calculator.history_store import HistoryRecord, TERHistoryStore


def _record(index: int, ter: float, tokens: int, waste: int) -> HistoryRecord:
    return HistoryRecord(
        session_id=f"s{index}",
        project="demo",
        timestamp=float(index),
        aggregate_ter=ter,
        phase_ter={
            "reasoning": ter - 0.05,
            "tool_use": ter,
            "generation": min(1.0, ter + 0.05),
        },
        waste_breakdown={"repeated_tool_call": waste},
        token_count=tokens,
        waste_tokens=waste,
        cost_usd=1.0,
        waste_cost_usd=0.1,
    )


def test_policy_is_bounded_and_history_driven():
    policy = learn_policy(
        [_record(i, 0.65 + i * 0.03, 100 + i * 20, 30 - i) for i in range(6)],
        "demo",
    )
    assert policy.sample_size == 6
    assert policy.confidence == "experimental"
    assert 0.30 <= policy.thresholds["similarity"] <= 0.55
    assert sum(policy.phase_weights.values()) == pytest.approx(1.0, abs=0.001)
    assert policy.token_budget["soft_limit"] <= policy.token_budget["hard_limit"]
    assert policy.intervention["min_duplicate_calls"] >= 2


def test_personalization_and_atomic_save(tmp_path: Path):
    policy = learn_policy([_record(i, 0.8, 100, 20) for i in range(5)], "demo")
    personalized = personalize_policy(
        policy,
        {"available": True, "predicted_ter": 0.4, "neighbors": 3},
    )
    assert personalized.token_budget["recommended"] < policy.token_budget["recommended"]
    target = save_policy(personalized, tmp_path / "policy.json")
    assert json.loads(target.read_text())["evidence"]["prompt_neighbors"] == 3
    assert not (tmp_path / "policy.json.tmp").exists()


def test_empty_history_is_rejected():
    with pytest.raises(ValueError, match="No history records"):
        learn_policy([], "missing")


def test_optimize_cli_writes_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    db = tmp_path / "history.db"
    store = TERHistoryStore(db)
    try:
        for index in range(5):
            store.put(_record(index, 0.8, 100 + index * 10, 20))
    finally:
        store.close()
    output = tmp_path / "adaptive-policy.json"
    assert (
        main(
            [
                "--quiet",
                "optimize",
                "--project",
                "demo",
                "--db",
                str(db),
                "--output",
                str(output),
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"] == "demo"
    assert output.exists()
