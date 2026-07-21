from types import SimpleNamespace

from ter_calculator.commands import memory


def test_trends_accepts_legacy_namespace_without_outcomes(
    monkeypatch, tmp_path, capsys
):
    captured = {}

    def fake_analyze(lessons, *, minimum_occurrences, outcome_path):
        captured["lessons"] = lessons
        captured["outcomes"] = outcome_path
        return {
            "lesson_count": 0,
            "scenarios": [],
            "intervention_effectiveness": {},
        }

    monkeypatch.setattr(memory, "analyze_trends", fake_analyze)
    args = SimpleNamespace(
        memory_command="trends",
        root=str(tmp_path),
        lessons=None,
        minimum_occurrences=1,
        output_format="text",
    )

    assert memory._cmd_memory(args) == 0
    assert captured["lessons"] == tmp_path / ".ter" / "session-lessons.jsonl"
    assert captured["outcomes"] == tmp_path / ".ter" / "intervention-outcomes.jsonl"
    assert "Recorded lessons: 0" in capsys.readouterr().out
