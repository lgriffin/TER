from pathlib import Path

from ter_calculator.history_store import (
    HistoryRecord,
    TERHistoryStore,
    prompt_fingerprint,
)


def record(session_id: str, project: str, ter: float, prompt: str) -> HistoryRecord:
    return HistoryRecord(
        session_id=session_id,
        project=project,
        timestamp=1.0,
        aggregate_ter=ter,
        phase_ter={"reasoning": ter},
        waste_breakdown={"duplicate_tool": 10},
        token_count=100,
        waste_tokens=10,
        cost_usd=1.0,
        waste_cost_usd=0.1,
        prompt_fingerprint=prompt_fingerprint(prompt),
    )


def test_store_query_profile_and_replace(tmp_path: Path):
    store = TERHistoryStore(tmp_path / "history.db")
    store.put(record("a", "alpha", 0.8, "fix parser tests"))
    store.put(record("b", "alpha", 1.0, "fix parser tests"))
    store.put(record("a", "alpha", 0.9, "fix parser tests"))

    rows = store.query(project="alpha")
    assert len(rows) == 2
    assert {row.aggregate_ter for row in rows} == {0.9, 1.0}
    profile = store.profile("alpha")
    assert profile["sessions"] == 2
    assert profile["average_ter"] == 0.95
    assert profile["main_waste_source"] == "duplicate_tool"
    store.close()


def test_prediction_uses_similar_private_fingerprints(tmp_path: Path):
    store = TERHistoryStore(tmp_path / "history.db")
    store.put(record("a", "alpha", 0.9, "add parser tests"))
    store.put(record("b", "alpha", 0.3, "redesign dashboard colors"))

    prediction = store.predict("add more parser tests", "alpha")
    assert prediction["available"] is True
    assert prediction["predicted_ter"] > 0.6
    assert prediction["confidence"] == "experimental"
    store.close()


def test_empty_profile_and_prediction(tmp_path: Path):
    store = TERHistoryStore(tmp_path / "history.db")
    assert store.profile("missing") == {"sessions": 0, "project": "missing"}
    assert store.predict("anything", "missing")["available"] is False
    store.close()
