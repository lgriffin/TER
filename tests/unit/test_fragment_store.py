"""Tests for fragment_store.py — content-addressable fragment storage."""

import time
import numpy as np
import pytest

from ter_calculator.models import Fragment, SpanPhase, TokenSpan
from ter_calculator.fragment_store import (
    FragmentStore,
    FragmentShardingEngine,
    _normalize_text,
    _compute_hash,
)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test_fragments.db"
    s = FragmentStore(db_path=db)
    yield s
    s.close()


def _make_fragment(text="hello world", phase=SpanPhase.REASONING, **kwargs):
    defaults = {
        "id": _compute_hash(_normalize_text(text)),
        "text": _normalize_text(text),
        "token_count": max(1, len(text) // 4),
        "phase": phase,
        "origin_session": "test-session",
        "created_at": time.time(),
        "embedding": np.random.randn(384).astype(np.float32),
    }
    defaults.update(kwargs)
    return Fragment(**defaults)


class TestNormalization:
    def test_whitespace_collapse(self):
        assert _normalize_text("  hello   world  ") == "hello world"

    def test_unicode_nfc(self):
        text_nfd = "é"  # e + combining accent
        text_nfc = "é"  # precomposed e-acute
        assert _normalize_text(text_nfd) == _normalize_text(text_nfc)

    def test_deterministic_hash(self):
        h1 = _compute_hash("hello world")
        h2 = _compute_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_content_different_hash(self):
        h1 = _compute_hash("hello")
        h2 = _compute_hash("world")
        assert h1 != h2


class TestFragmentStore:
    def test_put_and_get(self, store):
        frag = _make_fragment()
        store.put(frag)
        retrieved = store.get(frag.id)
        assert retrieved is not None
        assert retrieved.id == frag.id
        assert retrieved.text == frag.text

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_exists(self, store):
        frag = _make_fragment()
        assert not store.exists(frag.id)
        store.put(frag)
        assert store.exists(frag.id)

    def test_put_many(self, store):
        frags = [_make_fragment(f"text {i}") for i in range(5)]
        store.put_many(frags)
        assert store.count() == 5

    def test_get_many(self, store):
        frags = [_make_fragment(f"text {i}") for i in range(3)]
        store.put_many(frags)
        ids = [f.id for f in frags]
        retrieved = store.get_many(ids)
        assert len(retrieved) == 3

    def test_get_many_empty(self, store):
        assert store.get_many([]) == []

    def test_content_addressable_dedup(self, store):
        frag1 = _make_fragment("same content")
        frag2 = _make_fragment("same content")
        assert frag1.id == frag2.id
        store.put(frag1)
        store.put(frag2)
        assert store.count() == 1

    def test_find_by_session(self, store):
        f1 = _make_fragment("text a", origin_session="session-1")
        f2 = _make_fragment("text b", origin_session="session-2")
        store.put(f1)
        store.put(f2)
        result = store.find_by_session("session-1")
        assert len(result) == 1
        assert result[0].origin_session == "session-1"

    def test_find_by_phase(self, store):
        f1 = _make_fragment("reasoning text", phase=SpanPhase.REASONING)
        f2 = _make_fragment("tool text", phase=SpanPhase.TOOL_USE)
        store.put(f1)
        store.put(f2)
        result = store.find_by_phase(SpanPhase.REASONING)
        assert len(result) == 1

    def test_gc_removes_expired(self, store):
        old = _make_fragment("old text", created_at=1.0, ttl_seconds=1)
        store.put(old)
        assert store.count() == 1
        removed = store.gc(max_age_hours=0)
        assert removed == 1
        assert store.count() == 0

    def test_gc_keeps_fresh(self, store):
        fresh = _make_fragment("fresh text", created_at=time.time(), ttl_seconds=99999)
        store.put(fresh)
        removed = store.gc(max_age_hours=0)
        assert removed == 0
        assert store.count() == 1

    def test_embedding_roundtrip(self, store):
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        frag = _make_fragment("emb test", embedding=emb)
        store.put(frag)
        retrieved = store.get(frag.id)
        assert retrieved is not None
        assert retrieved.embedding is not None
        np.testing.assert_array_almost_equal(retrieved.embedding, emb)

    def test_null_embedding(self, store):
        frag = _make_fragment("no emb", embedding=None)
        store.put(frag)
        retrieved = store.get(frag.id)
        assert retrieved is not None
        assert retrieved.embedding is None

    def test_all_ids(self, store):
        frags = [_make_fragment(f"id test {i}") for i in range(3)]
        store.put_many(frags)
        ids = store.all_ids()
        assert len(ids) == 3


class TestFragmentShardingEngine:
    def test_shard_creates_fragments(self, store):
        spans = [
            TokenSpan(
                text="hello world",
                phase=SpanPhase.REASONING,
                position=0,
                token_count=3,
                source_message_uuid="msg-1",
            ),
            TokenSpan(
                text="goodbye world",
                phase=SpanPhase.GENERATION,
                position=1,
                token_count=3,
                source_message_uuid="msg-1",
            ),
        ]
        engine = FragmentShardingEngine(store)
        fragments = engine.shard(spans, "test-session", embed=False)
        assert len(fragments) == 2
        assert store.count() == 2

    def test_shard_deduplicates(self, store):
        spans = [
            TokenSpan(
                text="duplicate text",
                phase=SpanPhase.REASONING,
                position=0,
                token_count=3,
                source_message_uuid="msg-1",
            ),
            TokenSpan(
                text="duplicate text",
                phase=SpanPhase.REASONING,
                position=1,
                token_count=3,
                source_message_uuid="msg-2",
            ),
        ]
        engine = FragmentShardingEngine(store)
        fragments = engine.shard(spans, "test-session", embed=False)
        assert len(fragments) == 2
        assert store.count() == 1

    def test_shard_skips_empty(self, store):
        spans = [
            TokenSpan(
                text="",
                phase=SpanPhase.REASONING,
                position=0,
                token_count=0,
                source_message_uuid="msg-1",
            ),
            TokenSpan(
                text="   ",
                phase=SpanPhase.REASONING,
                position=1,
                token_count=0,
                source_message_uuid="msg-2",
            ),
        ]
        engine = FragmentShardingEngine(store)
        fragments = engine.shard(spans, "test-session", embed=False)
        assert len(fragments) == 0

    def test_shard_reuses_existing(self, store):
        existing = _make_fragment("existing content")
        store.put(existing)

        spans = [
            TokenSpan(
                text="existing content",
                phase=SpanPhase.REASONING,
                position=0,
                token_count=4,
                source_message_uuid="msg-1",
            ),
        ]
        engine = FragmentShardingEngine(store)
        fragments = engine.shard(spans, "new-session", embed=False)
        assert len(fragments) == 1
        assert fragments[0].origin_session == existing.origin_session
        assert store.count() == 1
