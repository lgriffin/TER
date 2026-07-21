import json
from pathlib import Path

from ter_calculator.closed_loop import (
    analyze_trends,
    append_lessons,
    build_memory_guidance,
)
from ter_calculator.hook_monitor import WasteAlert
from ter_calculator.repository_memory import build_index, inspect_index


def test_memory_guidance_and_semantic_duplicates(tmp_path: Path):
    (tmp_path / "a.py").write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "def increment(value):\n    return value + 2\n", encoding="utf-8"
    )
    build_index(tmp_path)
    inspected = inspect_index(tmp_path / ".ter" / "memory-index.json")
    assert inspected["semantic_duplicate_group_count"] >= 1
    guidance, matches = build_memory_guidance(
        {"cwd": str(tmp_path), "prompt": "implement increment helper"},
        minimum_score=0.0,
    )
    assert "TER Project Memory" in guidance
    assert matches


def test_lessons_are_deduplicated_and_trended(tmp_path: Path):
    path = tmp_path / "lessons.jsonl"
    alert = WasteAlert("duplicate_tool_call", "warning", "duplicate", {"x": 1})
    assert append_lessons(path, session_id="s1", repository="repo", alerts=[alert]) == 1
    assert append_lessons(path, session_id="s1", repository="repo", alerts=[alert]) == 0
    assert append_lessons(path, session_id="s2", repository="repo", alerts=[alert]) == 1
    result = analyze_trends(path)
    assert result["lesson_count"] == 2
    assert result["scenarios"][0]["occurrences"] == 2
