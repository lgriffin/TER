from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ter_calculator.commands import history, integrate, memory, optimize, production


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_memory_command_all_modes(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(
        memory,
        "build_index",
        lambda root, output: {
            "file_count": 1,
            "commit_count": 2,
            "index_path": "i",
            "chunk_count": 3,
            "duplicate_group_count": 1,
        },
    )
    assert (
        memory._cmd_memory(
            ns(
                memory_command="index",
                root=str(tmp_path),
                output=None,
                index_path=None,
                output_format="text",
            )
        )
        == 0
    )
    monkeypatch.setattr(
        memory,
        "inspect_index",
        lambda path: {
            "root": "r",
            "chunk_count": 3,
            "file_count": 1,
            "commit_count": 2,
            "duplicate_group_count": 1,
            "semantic_duplicate_group_count": 2,
        },
    )
    memory._cmd_memory(
        ns(
            memory_command="inspect",
            root=str(tmp_path),
            index_path=None,
            output_format="text",
        )
    )
    monkeypatch.setattr(
        memory,
        "search_index",
        lambda *a: {
            "matches": [
                {
                    "path": "a.py",
                    "start_line": 1,
                    "end_line": 2,
                    "score": 0.9,
                    "source_type": "code",
                    "excerpt": "hello\nworld",
                }
            ],
            "risk_flags": [{"type": "duplicate", "path": "a.py"}],
        },
    )
    memory._cmd_memory(
        ns(
            memory_command="search",
            root=str(tmp_path),
            index_path=None,
            query="x",
            limit=2,
            minimum_score=0,
            output_format="text",
        )
    )
    monkeypatch.setattr(
        memory,
        "analyze_trends",
        lambda *a, **k: {
            "lesson_count": 2,
            "scenarios": [{"message": "loop"}],
            "intervention_effectiveness": {
                "refresh": {
                    "compliance_rate": 0.5,
                    "improvement_rate": 0.25,
                    "mean_ter_delta": 0.1,
                }
            },
        },
    )
    memory._cmd_memory(
        ns(
            memory_command="trends",
            root=str(tmp_path),
            lessons=None,
            minimum_occurrences=1,
            output_format="text",
        )
    )
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(ValueError):
        memory._cmd_memory(ns(memory_command=None, output_format="json"))


def test_integrate_formats(monkeypatch, tmp_path: Path, capsys) -> None:
    result_dir = tmp_path / "r"
    result_dir.mkdir()
    gate = SimpleNamespace(passed=True, to_dict=lambda: {"passed": True})
    monkeypatch.setattr(integrate, "load_results", lambda p: ([{"x": 1}], []))
    monkeypatch.setattr(integrate, "evaluate_gate", lambda r, g: gate)
    monkeypatch.setattr(integrate, "build_sarif", lambda r, g: {"sarif": 1})
    monkeypatch.setattr(integrate, "build_github_annotations", lambda g: "::warning::x")
    monkeypatch.setattr(integrate, "build_step_summary", lambda g, p: "summary")
    for fmt in ("json", "sarif", "github", "summary"):
        args = ns(
            result_dir=str(result_dir),
            minimum_ter=0.5,
            maximum_waste_ratio=0.5,
            output=None,
            format=fmt,
            quiet=False,
        )
        assert integrate._cmd_integrate(args) == 0
        assert integrate._default_output(result_dir, fmt).exists()
    assert "artifact written" in capsys.readouterr().out
    with pytest.raises(ValueError):
        integrate._cmd_integrate(ns(result_dir=str(tmp_path / "missing")))


def test_history_dispatch_and_prints(monkeypatch, capsys) -> None:
    monkeypatch.setattr(history, "_list", lambda a: 7)
    assert history._cmd_history(ns(history_command="list")) == 7
    with pytest.raises(ValueError):
        history._cmd_history(ns(history_command="bad"))
    history._print_profile({"sessions": 0})
    history._print_profile(
        {
            "sessions": 2,
            "project": "p",
            "average_ter": 0.8,
            "total_tokens": 100,
            "waste_tokens": 20,
            "total_cost_usd": 1.2,
            "waste_cost_usd": 0.2,
            "main_waste_source": "loop",
        }
    )
    assert "Average TER" in capsys.readouterr().out


def test_optimize_and_production(monkeypatch, capsys, tmp_path: Path) -> None:
    class Store:
        def __init__(self, *a):
            pass

        def query(self, **k):
            return []

        def predict(self, *a, **k):
            return {"available": False}

        def close(self):
            pass

    policy = SimpleNamespace(
        to_dict=lambda: {
            "project": "p",
            "sample_size": 0,
            "confidence": "low",
            "evidence": {"average_ter": 0, "waste_ratio": 0},
            "thresholds": {"similarity": 0.5, "confidence": 0.5, "restatement": 0.5},
            "token_budget": {"soft_limit": 1, "recommended": 2, "hard_limit": 3},
        }
    )
    monkeypatch.setattr(optimize, "TERHistoryStore", Store)
    monkeypatch.setattr(optimize, "learn_policy", lambda *a, **k: policy)
    monkeypatch.setattr(optimize, "personalize_policy", lambda p, x: p)
    monkeypatch.setattr(optimize, "save_policy", lambda p, o: Path(o))
    assert (
        optimize._cmd_optimize(
            ns(
                db=None,
                project="p",
                minimum_samples=1,
                prompt="x",
                neighbors=2,
                output=str(tmp_path / "p.json"),
                quiet=False,
                output_format="text",
            )
        )
        == 0
    )
    report = SimpleNamespace(
        db_path="d",
        schema_version=1,
        integrity="ok",
        journal_mode="wal",
        writable=True,
        secure_permissions=True,
        issues=[],
        healthy=True,
    )
    monkeypatch.setattr(
        production.RuntimeConfig, "from_env", lambda db: SimpleNamespace(db_path="d")
    )
    monkeypatch.setattr(
        production, "TERHistoryStore", lambda p: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(production, "inspect_database", lambda p: report)
    assert production._cmd_doctor(ns(db=None, output_format="text")) == 0
    assert "production readiness" in capsys.readouterr().out
