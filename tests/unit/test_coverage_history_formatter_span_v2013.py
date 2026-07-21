from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from ter_calculator.commands import history
from ter_calculator import formatter
from ter_calculator.span_segmentation import (
    TextSegment,
    SegmentationConfig,
    segment_text,
    _split_oversized,
    _split_by_words,
    _merge_small,
)


def ns(**k):
    return SimpleNamespace(**k)


class FakeStore:
    def __init__(self, *a):
        self.path = Path("db")
        self.closed = False

    def query(self, **k):
        return [
            SimpleNamespace(
                timestamp=0,
                aggregate_ter=0.8,
                token_count=100,
                waste_tokens=20,
                cost_usd=1.2,
                project="p",
                session_id="s",
            )
        ]

    def profile(self, p):
        return {
            "sessions": 1,
            "project": p,
            "average_ter": 0.8,
            "total_tokens": 100,
            "waste_tokens": 20,
            "total_cost_usd": 1.2,
            "waste_cost_usd": 0.2,
            "main_waste_source": "loop",
        }

    def predict(self, *a, **k):
        return {
            "available": True,
            "predicted_ter": 0.7,
            "confidence": "high",
            "neighbors": 2,
            "sample_size": 3,
            "recommendation": "go",
        }

    def integrity_check(self):
        return "ok"

    def backup(self, o):
        return Path(o)

    def put(self, r):
        self.record = r

    def close(self):
        self.closed = True


def test_history_list_profile_predict_dashboard_backup_restore(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(history, "_store", lambda a: FakeStore())
    base = dict(db=None, project="p", output_format="text")
    assert history._list(ns(**base, min_ter=None, max_ter=None, limit=10)) == 0
    assert history._profile(ns(**base)) == 0
    assert history._predict(ns(**base, prompt="x", neighbors=2)) == 0
    assert history._cmd_dashboard(ns(db=None, project="p", limit=10)) == 0
    assert history._backup(ns(db=None, output=str(tmp_path / "b.db"))) == 0
    monkeypatch.setattr(history.TERHistoryStore, "restore", lambda b, d, force: Path(d))
    monkeypatch.setattr(
        "ter_calculator.production.RuntimeConfig.from_env",
        lambda db: SimpleNamespace(db_path=tmp_path / "r.db"),
    )
    assert history._restore(ns(db=None, backup="x", force=True)) == 0
    out = capsys.readouterr().out
    assert "Predicted TER" in out and "TER trend" in out and "Backup written" in out

    # JSON and unavailable prediction branches
    monkeypatch.setattr(
        FakeStore,
        "predict",
        lambda self, *a, **k: {"available": False, "sample_size": 0},
    )
    history._list(
        ns(**{**base, "output_format": "json"}, min_ter=None, max_ter=None, limit=1)
    )
    history._profile(ns(**{**base, "output_format": "json"}))
    history._predict(ns(**base, prompt="x", neighbors=1))
    assert "No comparable" in capsys.readouterr().out


def test_history_record(monkeypatch, tmp_path, capsys):
    session_file = tmp_path / "s.jsonl"
    session_file.write_text("{}\n")
    fake_result = SimpleNamespace(
        session_id="s",
        aggregate_ter=0.8,
        phase_scores={},
        total_tokens=10,
        waste_tokens=2,
        economics=None,
    )
    monkeypatch.setattr(
        "ter_calculator.analyze_pipeline.analyze_session", lambda a: fake_result
    )
    monkeypatch.setattr(
        "ter_calculator.analyze_pipeline.default_analyze_args", lambda p: object()
    )
    monkeypatch.setattr(
        "ter_calculator.loader.load_session",
        lambda p: SimpleNamespace(user_prompts=["hello"]),
    )
    monkeypatch.setattr(history, "waste_breakdown", lambda r: {})
    monkeypatch.setattr(history, "_store", lambda a: FakeStore())
    assert (
        history._record(
            ns(session_path=str(session_file), prompt=None, project=None, db=None)
        )
        == 0
    )
    assert "Recorded" in capsys.readouterr().out


def test_formatter_dispatch_fallback_and_helpers(monkeypatch):
    r = SimpleNamespace(
        total_tokens=100,
        waste_tokens=20,
        aggregate_ter=0.8,
        economics=None,
        classified_spans=[],
        waste_patterns=[],
    )
    monkeypatch.setattr("ter_calculator.formatter_json.format_json", lambda x: "json")
    monkeypatch.setattr("ter_calculator.formatter_html.format_html", lambda x: "html")
    monkeypatch.setattr("ter_calculator.formatter_text.format_text", lambda x: "text")
    assert formatter.format_ter_result(r, "json") == "json"
    assert formatter.format_ter_result(r, "html") == "html"
    assert formatter.format_ter_result(r, "text", False) == "text"
    monkeypatch.setattr(
        "ter_calculator.formatter_rich.format_rich",
        lambda x: (_ for _ in ()).throw(UnicodeEncodeError("x", "x", 0, 1, "x")),
    )
    assert formatter.format_ter_result(r) == "text"
    monkeypatch.setattr(
        "ter_calculator.formatter_json.format_comparison_json", lambda x: "cj"
    )
    monkeypatch.setattr(
        "ter_calculator.formatter_text.format_comparison_text", lambda x: "ct"
    )
    assert (
        formatter.format_comparison([r], "json") == "cj"
        and formatter.format_comparison([r], use_rich=False) == "ct"
    )
    monkeypatch.setattr(
        "ter_calculator.formatter_json.format_grouped_json", lambda a, b: "gj"
    )
    monkeypatch.setattr(
        "ter_calculator.formatter_text.format_grouped_text", lambda a, b: "gt"
    )
    assert (
        formatter.format_grouped_analysis(r, [], "json") == "gj"
        and formatter.format_grouped_analysis(r, [], use_rich=False) == "gt"
    )
    assert formatter._compute_group_aggregates([])["weighted_ter"] == 0
    assert formatter._pattern_pricing("repetitive_read") == "input"
    assert formatter._pattern_pricing("reasoning_loop") == "output"
    assert formatter._compute_waste_cost(r) == 0


def test_span_segmentation_edges():
    text = "First sentence is here. Second sentence is much longer and has several words.\n\nThird paragraph follows with more content."
    segs = segment_text(
        text, SegmentationConfig(enabled=True, min_tokens=2, max_tokens=8)
    )
    assert segs and all(s.char_end > s.char_start for s in segs)
    whole = TextSegment(text, 0, len(text))
    assert _split_oversized(text, whole, 5)
    assert _split_by_words(text, whole, 3)
    assert _split_by_words("   ", TextSegment("   ", 0, 3), 3) == []
    assert _merge_small(text, [], 2, 10) == []
    merged = _merge_small(
        "a b c d", [TextSegment("a", 0, 1), TextSegment("b c d", 2, 7)], 2, 10
    )
    assert len(merged) == 1
