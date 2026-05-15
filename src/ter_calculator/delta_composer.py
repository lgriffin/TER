"""Reference-based prompt composition with delta transmission.

Implements the Delta Prompt Composer and SLM Execution Runtime from the
Token Aware Microprompt Orchestrator patent: composes prompts by citing
immutable fragment references, transmitting only fragments not already
cached at the target.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Sequence

from .fragment_store import FragmentStore
from .models import (
    DeltaPrompt,
    Fragment,
    FragmentManifest,
    InvalidationEvent,
    PromptTemplate,
    Session,
)

_PLACEHOLDER_RE = re.compile(r"\{\{([a-f0-9]{64})\}\}")


class LocalCache:
    """LRU cache of fragments keyed by fragment ID."""

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, Fragment] = OrderedDict()

    def has(self, fragment_id: str) -> bool:
        return fragment_id in self._data

    def get(self, fragment_id: str) -> Fragment | None:
        if fragment_id not in self._data:
            return None
        self._data.move_to_end(fragment_id)
        return self._data[fragment_id]

    def put(self, fragment: Fragment) -> None:
        if fragment.id in self._data:
            self._data.move_to_end(fragment.id)
            self._data[fragment.id] = fragment
        else:
            if len(self._data) >= self._max_size:
                self._data.popitem(last=False)
            self._data[fragment.id] = fragment

    def evict(self, fragment_id: str) -> None:
        self._data.pop(fragment_id, None)

    def invalidate_from_events(
        self, events: Sequence[InvalidationEvent]
    ) -> int:
        count = 0
        for event in events:
            if event.fragment_id in self._data:
                del self._data[event.fragment_id]
                count += 1
        return count

    @property
    def size(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


def compose_delta(
    template: PromptTemplate,
    store: FragmentStore,
    local_cache: LocalCache,
) -> DeltaPrompt:
    cache_hits = 0
    cache_misses = 0
    delta_fragments: list[Fragment] = []
    total_tokens = 0
    cached_tokens = 0

    for frag_id in template.required_fragment_ids:
        cached = local_cache.get(frag_id)
        if cached is not None:
            cache_hits += 1
            cached_tokens += cached.token_count
            total_tokens += cached.token_count
        else:
            from_store = store.get(frag_id)
            if from_store is not None:
                delta_fragments.append(from_store)
                total_tokens += from_store.token_count
                cache_misses += 1
            else:
                cache_misses += 1

    delta_tokens = sum(f.token_count for f in delta_fragments)
    compression = 1.0 - (delta_tokens / total_tokens) if total_tokens > 0 else 0.0

    manifest = FragmentManifest(
        fragment_ids=list(template.required_fragment_ids),
        total_tokens=total_tokens,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )

    return DeltaPrompt(
        template=template,
        manifest=manifest,
        delta_fragments=delta_fragments,
        total_tokens_saved=cached_tokens,
        compression_ratio=compression,
    )


def resolve_prompt(delta: DeltaPrompt, local_cache: LocalCache) -> str:
    lookup: dict[str, str] = {}

    for frag_id in delta.template.required_fragment_ids:
        cached = local_cache.get(frag_id)
        if cached is not None:
            lookup[frag_id] = cached.text

    for frag in delta.delta_fragments:
        lookup[frag.id] = frag.text
        local_cache.put(frag)

    result = delta.template.template_text
    for frag_id, text in lookup.items():
        result = result.replace("{{" + frag_id + "}}", text)

    return result


def create_template_from_session(
    session: Session,
    fragments: list[Fragment],
) -> PromptTemplate:
    parts: list[str] = []
    for msg in session.messages:
        for block in msg.content_blocks:
            if block.text:
                parts.append(block.text)

    full_text = "\n".join(parts)

    sorted_frags = sorted(fragments, key=lambda f: len(f.text), reverse=True)

    required_ids: list[str] = []
    for frag in sorted_frags:
        placeholder = "{{" + frag.id + "}}"
        if frag.text in full_text:
            full_text = full_text.replace(frag.text, placeholder, 1)
            required_ids.append(frag.id)

    return PromptTemplate(
        template_text=full_text,
        required_fragment_ids=required_ids,
    )
