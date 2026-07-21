import json
import time
from pathlib import Path

import pytest

from ter_calculator.acceleration import AnalysisCache, QuickAnalyser, hash_file


def test_hash_file_and_missing(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert hash_file(p) == hash_file(str(p))
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "none")


def test_analysis_cache_hit_miss_stats_expiry_corruption_and_clear(tmp_path):
    c = AnalysisCache(tmp_path / "cache")
    calls = []
    assert c.get_or_compute("abcdef", lambda: calls.append(1) or {"x": 1}) == {"x": 1}
    assert c.get_or_compute("abcdef", lambda: calls.append(2)) == {"x": 1}
    assert calls == [1]
    stats = c.cache_stats()
    assert (stats.hit_count, stats.miss_count, stats.entry_count) == (1, 1, 1)

    pkl, meta = c._key_paths("abcdef")
    m = json.loads(meta.read_text())
    m["timestamp"] = time.time() - 7200
    meta.write_text(json.dumps(m))
    assert c._read("abcdef", ttl_hours=1) is None
    assert not pkl.exists()

    c._write("badpickle", {"x": 2}, 24)
    pkl, meta = c._key_paths("badpickle")
    pkl.write_bytes(b"not pickle")
    assert c._read("badpickle", 24) is None

    c._write("badmeta", 3, 24)
    _, meta = c._key_paths("badmeta")
    meta.write_text("{")
    assert c._read("badmeta", 24) is None
    c.clear_all()
    assert c.cache_stats().entry_count == 0


def test_cache_invalidate_by_hash_and_metadata(tmp_path):
    source = tmp_path / "session.jsonl"
    source.write_text("x")
    c = AnalysisCache(tmp_path / "cache")
    key = hash_file(source) + "extra"
    c._write(key, 1, 24)
    c._write("zzmeta", 2, 24)
    _, meta = c._key_paths("zzmeta")
    raw = json.loads(meta.read_text())
    raw["source_path"] = str(source)
    meta.write_text(json.dumps(raw))
    c.invalidate(str(source))
    assert c.cache_stats().entry_count == 0


def write_session(path: Path, lines):
    path.write_text(
        "\n".join(json.dumps(x) if isinstance(x, dict) else x for x in lines),
        encoding="utf-8",
    )


def test_quick_analyser_empty_all_aligned_and_scored(tmp_path):
    q = QuickAnalyser(top_n_keywords=5)
    assert q.top_n_keywords == 5
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert q.analyse_quick(str(empty))["total_tokens"] == 0

    no_keywords = tmp_path / "no_kw.jsonl"
    write_session(
        no_keywords,
        [
            {"sessionId": "s", "message": {"role": "user", "content": "the and with"}},
            {"sessionId": "s", "message": {"role": "assistant", "content": "hello"}},
        ],
    )
    assert q.analyse_quick(str(no_keywords))["waste_tokens"] == 0

    p = tmp_path / "s.jsonl"
    write_session(
        p,
        [
            "bad json",
            {
                "sessionId": "sid",
                "timestamp": "1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "fix parser bug quickly"}],
                },
            },
            {
                "sessionId": "sid",
                "timestamp": "2",
                "message": {
                    "role": "assistant",
                    "requestId": "r",
                    "content": [{"type": "thinking", "thinking": "fix parser bug"}],
                },
            },
            {
                "sessionId": "sid",
                "timestamp": "2",
                "message": {
                    "role": "assistant",
                    "requestId": "r",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file": "parser.py"},
                        }
                    ],
                },
            },
            {
                "sessionId": "sid",
                "timestamp": "3",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "unrelated prose"}],
                },
            },
        ],
    )
    out = q.analyse_quick(str(p), {"similarity_threshold": 0.2})
    assert out["session_id"] == "sid" and out["total_tokens"] > 0
    assert out["method"] == "quick_keyword"
    parsed = q._parse_session(str(p))
    assert {s["phase"] for s in parsed["spans"]} >= {
        "reasoning",
        "tool_use",
        "generation",
    }
    with pytest.raises(FileNotFoundError):
        q.analyse_quick(str(tmp_path / "missing"))


def test_quick_helpers():
    q = QuickAnalyser(2)
    assert q._extract_keywords([]) == set()
    assert q._extract_keywords(["alpha alpha beta gamma"]) == {"alpha", "beta"}
    assert q._keyword_overlap_score("alpha beta", {"alpha", "gamma"}) == 0.5
    assert q._keyword_overlap_score("x", set()) == 0
    result = q._compute_result(
        "s",
        [
            {"phase": "reasoning", "token_count": 10, "label": "aligned"},
            {"phase": "other", "token_count": 5, "label": "waste"},
        ],
    )
    assert result["aligned_tokens"] == 10 and result["waste_tokens"] == 5
