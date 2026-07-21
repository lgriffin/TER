from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from ter_calculator import closed_loop as cl
from ter_calculator import input_analysis as ia
from ter_calculator.models import ContentBlock, Message, Session


def test_closed_loop_guidance_lessons_trends_and_atomic(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    index = root / ".ter" / "memory-index.json"
    index.parent.mkdir()
    index.write_text("{}")
    event = {
        "cwd": str(root),
        "prompt": "fix duplicate",
        "tool_input": {"file_path": "a.py", "command": "cat a.py"},
    }
    monkeypatch.setattr(
        cl,
        "search_index",
        lambda *a, **k: {
            "matches": [
                {
                    "path": "a.py",
                    "start_line": 2,
                    "end_line": 5,
                    "score": 0.9,
                    "excerpt": "same   code",
                }
            ],
            "risk_flags": [
                {"type": "duplicate_pattern", "path": "a.py"},
                {"type": "prior_defect_or_fix", "path": "b.py"},
            ],
        },
    )
    text, matches = cl.build_memory_guidance(event)
    assert "TER Project Memory" in text and "reuse" in text and len(matches) == 1
    assert cl.resolve_project_root({"project_dir": str(root)}) == root.resolve()
    assert cl.build_memory_guidance({"cwd": str(root)}) == ("", [])
    monkeypatch.setattr(
        cl, "search_index", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert cl.build_memory_guidance(event) == ("", [])

    alerts = [
        SimpleNamespace(
            pattern_type="reasoning_loop",
            message="loop",
            severity="high",
            details={"n": 2},
        )
    ]
    lessons = tmp_path / "lessons.jsonl"
    assert (
        cl.append_lessons(lessons, session_id="s", repository="r", alerts=alerts) == 1
    )
    assert (
        cl.append_lessons(lessons, session_id="s", repository="r", alerts=alerts) == 0
    )
    lessons.write_text(lessons.read_text() + "bad\n{}\n", encoding="utf-8")
    assert ("s", "reasoning_loop", "loop") in cl._recent_keys(lessons)

    outcomes = tmp_path / "outcomes.jsonl"
    cl.record_outcome(
        outcomes,
        session_id="s",
        intervention_type="refresh",
        outcome="issued",
        details={"x": 1},
    )
    rows = [
        {
            "effect": "improved",
            "intervention_type": "refresh",
            "followed": True,
            "deltas": {"ter": 0.2, "waste_ratio": -0.1},
        },
        {
            "effect": "ignored",
            "intervention_type": "refresh",
            "followed": False,
            "deltas": {"ter": 0, "waste_ratio": 0},
        },
    ]
    outcomes.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\ninvalid\n", encoding="utf-8"
    )
    trends = cl.analyze_trends(lessons, minimum_occurrences=1, outcome_path=outcomes)
    metrics = trends["intervention_effectiveness"]["refresh"]
    assert (
        metrics["issued"] == 2
        and metrics["compliance_rate"] == 0.5
        and metrics["improvement_rate"] == 0.5
    )
    target = tmp_path / "nested" / "state.json"
    cl.atomic_write_json(target, {"b": 1})
    assert json.loads(target.read_text()) == {"b": 1}


def _session() -> Session:
    return Session(
        session_id="s",
        file_path="x",
        user_prompts=["alpha", "alpha again", "beta"],
        messages=[
            Message(
                uuid="u1",
                role="user",
                content_blocks=[ContentBlock(block_type="text", text="alpha")],
            ),
            Message(
                uuid="a1",
                role="assistant",
                content_blocks=[
                    ContentBlock(block_type="thinking", text="think"),
                    ContentBlock(
                        block_type="tool_use",
                        tool_name="Read",
                        tool_input={"path": "x"},
                    ),
                    ContentBlock(block_type="text", text="alpha answer"),
                ],
            ),
            Message(
                uuid="tr",
                role="user",
                content_blocks=[ContentBlock(block_type="tool_result", text="result")],
            ),
            Message(
                uuid="u2",
                role="user",
                content_blocks=[ContentBlock(block_type="text", text="beta")],
            ),
            Message(
                uuid="a2",
                role="assistant",
                content_blocks=[
                    ContentBlock(block_type="text", text="different response")
                ],
            ),
        ],
    )


def test_input_analysis_without_optional_model(monkeypatch):
    def emb(texts):
        rows = []
        for t in texts:
            if "alpha" in t:
                rows.append([1.0, 0.0, 0.0])
            elif "beta" in t:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows)

    monkeypatch.setattr(ia, "embed_texts", emb)
    session = _session()
    bd = ia.compute_token_breakdown(session)
    assert (
        bd.model_reasoning_tokens > 0
        and bd.model_tool_tokens > 0
        and bd.user_result_tokens > 0
    )
    sim = ia.compute_prompt_similarity(["alpha", "alpha again", "beta"], 0.7)
    assert sim.prompt_redundancy_score == pytest.approx(2 / 3, abs=1e-4)
    assert ia.compute_prompt_similarity([]).prompt_count == 0
    assert ia.compute_prompt_similarity(["x"]).similarity_matrix == [[1.0]]
    matrix = ia._pairwise_cosine_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert matrix[1, 1] == 0
    drift = ia.compute_intent_drift(["alpha", "alpha again", "beta"])
    assert [s.drift_type for s in drift.steps] == ["convergent", "divergent"]
    assert drift.overall_trajectory == "mixed"
    assert ia.compute_intent_drift([]).overall_trajectory == "stable"
    assert ia._classify_trajectory([]) == "stable"
    assert ia._classify_trajectory([SimpleNamespace(drift_type="evolving")]) == "stable"
    align = ia.compute_prompt_response_alignment(session, 0.5)
    assert len(align.pairs) == 2 and align.low_alignment_count >= 1
    empty = Session(session_id="e", file_path="e", messages=[], user_prompts=[])
    assert ia.compute_prompt_response_alignment(empty).pairs == []
    assert len(ia._extract_prompt_response_pairs(session)) == 2
