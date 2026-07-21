from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ter_calculator.commands import optimize, release, watch
from ter_calculator.hook_monitor import HookConfig, HookSessionState
from ter_calculator.intervention import _evaluate_active_interventions
from ter_calculator.repository_memory import _risk_flags, load_index, search_index


def _watch_args(**overrides):
    values = {
        "stream": True,
        "output_format": "text",
        "latest": False,
        "project_path": None,
        "quiet": True,
        "verbose": False,
        "poll_interval": 0.01,
        "log_file": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_watch_embedding_and_latest_error_paths(monkeypatch, tmp_path, capsys):
    import ter_calculator.real_time as real_time

    monkeypatch.setattr(
        real_time,
        "load_embedding_model",
        lambda: (_ for _ in ()).throw(ImportError("embedding unavailable")),
    )
    assert watch._cmd_watch(_watch_args()) == 1
    assert "embedding unavailable" in capsys.readouterr().err

    monkeypatch.setattr(real_time, "load_embedding_model", lambda: object())
    import ter_calculator.loader as loader

    latest = tmp_path / "latest.jsonl"
    latest.write_text("", encoding="utf-8")
    monkeypatch.setattr(loader, "find_latest_session", lambda _: latest)

    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise KeyboardInterrupt

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(real_time, "SessionMonitor", Monitor)
    assert watch._cmd_watch(_watch_args(latest=True)) == 0


def test_watch_directory_resolution_logging_and_runtime_error(
    monkeypatch, tmp_path, capsys
):
    import ter_calculator.real_time as real_time

    monkeypatch.setattr(real_time, "load_embedding_model", lambda: object())
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    sibling = tmp_path / "session.jsonl"
    sibling.write_text("", encoding="utf-8")
    log_path = tmp_path / "signals.jsonl"

    class Signal:
        session_id = "session-123"
        timestamp = 1.0
        is_live = True
        aggregate_ter = 0.7
        raw_ratio = 0.8
        message_index = 2
        drift = SimpleNamespace(value="stable")
        drift_magnitude = 0.0
        warnings = []
        warning_level = SimpleNamespace(value="none")
        total_tokens = 0
        aligned_tokens = 0
        waste_tokens = 0
        phase_ter = {}
        waste_sources = {}

    class Monitor:
        def __init__(self, *args, on_signal=None, **kwargs):
            self.on_signal = on_signal

        def run(self):
            self.on_signal(Signal())
            raise RuntimeError("monitor failed")

        def stop(self):
            pass

    monkeypatch.setattr(real_time, "SessionMonitor", Monitor)
    result = watch._cmd_watch(
        _watch_args(project_path=str(session_dir), log_file=str(log_path))
    )
    assert result == 1
    assert "monitor failed" in capsys.readouterr().err
    assert json.loads(log_path.read_text(encoding="utf-8"))["ter"] == 0.7


def test_optimize_prompt_output_and_text_rendering(monkeypatch, tmp_path, capsys):
    closed = []

    class Store:
        def __init__(self, path):
            self.path = path

        def query(self, **kwargs):
            return [object()]

        def predict(self, prompt, project, k):
            return {"prompt": prompt, "k": k}

        def close(self):
            closed.append(True)

    policy_data = {
        "project": "demo",
        "sample_size": 1,
        "confidence": "low",
        "evidence": {"average_ter": 0.75, "waste_ratio": 0.20},
        "thresholds": {"similarity": 0.8, "confidence": 0.7, "restatement": 0.9},
        "token_budget": {"soft_limit": 100, "recommended": 200, "hard_limit": 300},
    }

    class Policy:
        def to_dict(self):
            return policy_data

    monkeypatch.setattr(optimize, "TERHistoryStore", Store)
    monkeypatch.setattr(optimize, "learn_policy", lambda *args, **kwargs: Policy())
    monkeypatch.setattr(
        optimize, "personalize_policy", lambda policy, prediction: policy
    )
    output = tmp_path / "policy.json"
    monkeypatch.setattr(optimize, "save_policy", lambda policy, path: output)

    args = SimpleNamespace(
        db="history.db",
        project="demo",
        minimum_samples=1,
        prompt="fix retries",
        neighbors=3,
        output=str(output),
        quiet=False,
        output_format="text",
    )
    assert optimize._cmd_optimize(args) == 0
    assert closed
    assert "Adaptive policy written" in capsys.readouterr().out

    args.output = None
    assert optimize._cmd_optimize(args) == 0
    assert "TER Adaptive Optimization Policy" in capsys.readouterr().out


def test_release_validation_errors_and_baseline_summary(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(
        result_dir=str(tmp_path / "missing"),
        baseline=None,
        minimum_sessions=1,
        minimum_ter=0.5,
        maximum_waste_ratio=0.5,
        maximum_ter_drop=0.1,
        maximum_waste_increase=0.1,
        output=None,
        format="json",
        quiet=True,
    )
    with pytest.raises(ValueError, match="does not exist"):
        release._cmd_release(args)

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    args.result_dir = str(result_dir)
    monkeypatch.setattr(release, "load_results", lambda _: ([], ["bad"]))
    with pytest.raises(ValueError, match="No valid"):
        release._cmd_release(args)

    monkeypatch.setattr(release, "load_results", lambda _: ([object()], ["bad"]))
    monkeypatch.setattr(release, "build_release_snapshot", lambda *a, **k: {"ter": 0.8})
    assessment = SimpleNamespace(passed=False, to_dict=lambda: {"passed": False})
    monkeypatch.setattr(release, "evaluate_release", lambda *a, **k: assessment)
    monkeypatch.setattr(release, "build_file_checksums", lambda _: {"a": "hash"})
    monkeypatch.setattr(release, "build_release_summary", lambda *a: "summary")
    written = {}
    monkeypatch.setattr(
        release, "atomic_write_text", lambda path, text: written.update(text=text)
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"snapshot": {"ter": 0.7}}), encoding="utf-8")
    args.baseline = str(baseline)
    args.format = "summary"
    args.output = str(tmp_path / "manifest.txt")
    args.quiet = False
    assert release._cmd_release(args) == 2
    assert written["text"] == "summary"
    assert "Release artifact written" in capsys.readouterr().out


def test_active_intervention_waits_then_records_followed_outcome(monkeypatch, tmp_path):
    state = HookSessionState(session_id="s")
    state.active_interventions = [
        {
            "intervention_id": "int-1",
            "session_id": "s",
            "action": "refresh_context",
            "issued_at": 1.0,
            "baseline": {
                "timestamp": 1.0,
                "ter": 0.4,
                "waste_ratio": 0.5,
                "repeated_tool_calls": 3,
            },
            "reason": "degraded",
            "related_memory_ids": [],
            "evaluation_due_after_events": 2,
            "events_since_issue": 0,
        }
    ]
    config = HookConfig(outcome_store=str(tmp_path / "outcomes.jsonl"))
    captured = []
    monkeypatch.setattr(
        "ter_calculator.intervention.append_intervention_outcome",
        lambda path, record, post, compliance: captured.append(compliance),
    )
    metrics = {"ter": 0.5, "waste_ratio": 0.3, "repeated_tool_calls": 1}
    _evaluate_active_interventions({"metrics": metrics}, state, config)
    assert len(state.active_interventions) == 1
    _evaluate_active_interventions(
        {
            "metrics": metrics,
            "assistant_message": "Objective and next action are clear",
        },
        state,
        config,
    )
    assert state.active_interventions == []
    assert captured[0].acknowledged is True
    assert captured[0].followed is True


def test_repository_risk_flags_handle_duplicate_semantic_and_defect_edges(tmp_path):
    chunks = [
        {
            "chunk_id": "a",
            "fingerprint": "same",
            "path": "a.py",
            "start_line": 1,
        },
        {
            "chunk_id": "b",
            "fingerprint": "same",
            "path": "b.py",
            "start_line": 1,
        },
    ]
    payload = {
        "chunks": chunks,
        "duplicate_groups": [["a", "b"]],
        "semantic_duplicate_groups": [["a", "b"]],
    }
    matches = [
        {
            "fingerprint": "same",
            "path": "a.py",
            "start_line": 1,
            "score": 0.9,
            "excerpt": "Fix regression in duplicated retry path",
        }
    ]
    assert {flag["type"] for flag in _risk_flags(matches, payload)} == {
        "duplicate_pattern",
        "semantic_duplicate_pattern",
        "prior_defect_or_fix",
    }

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_index(bad)
    with pytest.raises(ValueError, match="greater than zero"):
        search_index(bad, "query", limit=0)
