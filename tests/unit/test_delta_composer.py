"""Tests for delta_composer.py — reference-based prompt composition."""

import time
import pytest

from ter_calculator.models import (
    Fragment,
    FragmentManifest,
    InvalidationEvent,
    PromptTemplate,
    SpanPhase,
)
from ter_calculator.delta_composer import (
    LocalCache,
    compose_delta,
    resolve_prompt,
    create_template_from_session,
)
from ter_calculator.fragment_store import FragmentStore, _normalize_text, _compute_hash


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "delta_test.db"
    s = FragmentStore(db_path=db)
    yield s
    s.close()


def _frag(text, **kwargs):
    normalized = _normalize_text(text)
    defaults = {
        "id": _compute_hash(normalized),
        "text": normalized,
        "token_count": max(1, len(text) // 4),
        "phase": SpanPhase.REASONING,
        "origin_session": "test",
        "created_at": time.time(),
    }
    defaults.update(kwargs)
    return Fragment(**defaults)


class TestLocalCache:
    def test_put_and_get(self):
        cache = LocalCache(max_size=10)
        frag = _frag("hello")
        cache.put(frag)
        assert cache.has(frag.id)
        assert cache.get(frag.id) is frag

    def test_get_nonexistent(self):
        cache = LocalCache()
        assert cache.get("nope") is None
        assert not cache.has("nope")

    def test_lru_eviction(self):
        cache = LocalCache(max_size=2)
        f1 = _frag("first")
        f2 = _frag("second")
        f3 = _frag("third")
        cache.put(f1)
        cache.put(f2)
        cache.put(f3)
        assert not cache.has(f1.id)
        assert cache.has(f2.id)
        assert cache.has(f3.id)
        assert cache.size == 2

    def test_lru_access_refreshes(self):
        cache = LocalCache(max_size=2)
        f1 = _frag("first")
        f2 = _frag("second")
        f3 = _frag("third")
        cache.put(f1)
        cache.put(f2)
        cache.get(f1.id)  # refresh f1
        cache.put(f3)     # should evict f2, not f1
        assert cache.has(f1.id)
        assert not cache.has(f2.id)
        assert cache.has(f3.id)

    def test_evict(self):
        cache = LocalCache()
        frag = _frag("evict me")
        cache.put(frag)
        cache.evict(frag.id)
        assert not cache.has(frag.id)

    def test_invalidate_from_events(self):
        cache = LocalCache()
        f1 = _frag("one")
        f2 = _frag("two")
        cache.put(f1)
        cache.put(f2)
        events = [
            InvalidationEvent(
                fragment_id=f1.id,
                timestamp=time.time(),
                reason="updated",
            )
        ]
        count = cache.invalidate_from_events(events)
        assert count == 1
        assert not cache.has(f1.id)
        assert cache.has(f2.id)

    def test_clear(self):
        cache = LocalCache()
        cache.put(_frag("a"))
        cache.put(_frag("b"))
        cache.clear()
        assert cache.size == 0

    def test_put_updates_existing(self):
        cache = LocalCache()
        f1 = _frag("original")
        f2 = Fragment(
            id=f1.id,
            text="updated",
            token_count=2,
            phase=SpanPhase.REASONING,
            origin_session="test",
            created_at=time.time(),
        )
        cache.put(f1)
        cache.put(f2)
        assert cache.size == 1
        assert cache.get(f1.id).text == "updated"


class TestComposeDelta:
    def test_all_cache_misses(self, store):
        f1 = _frag("content one")
        f2 = _frag("content two")
        store.put(f1)
        store.put(f2)

        template = PromptTemplate(
            template_text=f"{{{{{f1.id}}}}} and {{{{{f2.id}}}}}",
            required_fragment_ids=[f1.id, f2.id],
        )
        cache = LocalCache()
        delta = compose_delta(template, store, cache)

        assert delta.manifest.cache_hits == 0
        assert delta.manifest.cache_misses == 2
        assert len(delta.delta_fragments) == 2
        assert delta.compression_ratio == 0.0

    def test_all_cache_hits(self, store):
        f1 = _frag("cached content")
        store.put(f1)

        template = PromptTemplate(
            template_text=f"{{{{{f1.id}}}}}",
            required_fragment_ids=[f1.id],
        )
        cache = LocalCache()
        cache.put(f1)
        delta = compose_delta(template, store, cache)

        assert delta.manifest.cache_hits == 1
        assert delta.manifest.cache_misses == 0
        assert len(delta.delta_fragments) == 0
        assert delta.compression_ratio == 1.0

    def test_mixed_hits_and_misses(self, store):
        f1 = _frag("cached")
        f2 = _frag("not cached")
        store.put(f1)
        store.put(f2)

        template = PromptTemplate(
            template_text="test",
            required_fragment_ids=[f1.id, f2.id],
        )
        cache = LocalCache()
        cache.put(f1)
        delta = compose_delta(template, store, cache)

        assert delta.manifest.cache_hits == 1
        assert delta.manifest.cache_misses == 1
        assert len(delta.delta_fragments) == 1
        assert delta.total_tokens_saved > 0


class TestResolvePrompt:
    def test_resolve_with_cache(self, store):
        f1 = _frag("hello")
        f2 = _frag("world")
        store.put(f1)
        store.put(f2)

        template = PromptTemplate(
            template_text=f"{{{{{f1.id}}}}} {{{{{f2.id}}}}}",
            required_fragment_ids=[f1.id, f2.id],
        )
        cache = LocalCache()
        delta = compose_delta(template, store, cache)
        result = resolve_prompt(delta, cache)

        assert "hello" in result
        assert "world" in result
        assert "{{" not in result

    def test_resolve_updates_cache(self, store):
        frag = _frag("new content")
        store.put(frag)

        template = PromptTemplate(
            template_text=f"{{{{{frag.id}}}}}",
            required_fragment_ids=[frag.id],
        )
        cache = LocalCache()
        delta = compose_delta(template, store, cache)
        resolve_prompt(delta, cache)

        assert cache.has(frag.id)


class TestCreateTemplate:
    def test_creates_placeholders(self):
        from types import SimpleNamespace

        blocks = [
            SimpleNamespace(text="hello world", block_type="text",
                            tool_name=None, tool_input=None, tool_use_id=None),
        ]
        messages = [
            SimpleNamespace(
                uuid="m1", role="user",
                content_blocks=blocks,
            ),
        ]
        session = SimpleNamespace(
            session_id="test",
            file_path="test.jsonl",
            messages=messages,
            timestamp=None,
            total_tokens=10,
            user_prompts=["hello world"],
        )
        frag = _frag("hello world")
        template = create_template_from_session(session, [frag])

        assert frag.id in template.required_fragment_ids
        assert "{{" in template.template_text
