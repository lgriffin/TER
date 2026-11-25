"""Harness-made Claude Code messages are not model turns (TER-ANL-044)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.in_memory import InMemoryPriceBook
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import PriceEntry, Rates
from ter.domain.costing import price_session
from ter.domain.lean.analysis import explain
from ter.domain.lean.inventory import context_inventory

RATES = Rates(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)


def _assistant(n: int, model: str, text: str, tokens: int) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": f"m{n}",
        "parentUuid": f"m{n - 1}",
        "sessionId": "s",
        "requestId": f"r{n}",
        "timestamp": f"2026-11-01T10:00:0{n}.000Z",
        "message": {
            "id": f"a{n}",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": tokens,
                "output_tokens": tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    }


def _session(tmp_path: Path, second_model: str, tokens: int = 0) -> Path:
    lines = [
        {
            "type": "user",
            "uuid": "m1",
            "parentUuid": None,
            "sessionId": "s",
            "timestamp": "2026-11-01T10:00:01.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        },
        _assistant(2, "claude-sonnet-5", "Hello.", 100),
        _assistant(3, second_model, "API Error: request was aborted.", tokens),
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


def _cost(path: Path) -> Any:
    trace = ClaudeCodeJsonlSource().read(path)
    analysis = explain(list(trace.events), RegexTokenizer())
    book = InMemoryPriceBook([PriceEntry("claude-sonnet-5", date(2025, 1, 1), RATES)])
    return trace, price_session(
        analysis.steps,
        context_inventory(analysis.steps, analysis.findings),
        book,
        analysis.classifications,
    )


@pytest.mark.req("TER-ANL-044")
class TestHarnessMessages:
    def test_a_synthetic_message_carries_no_usage(self, tmp_path: Path) -> None:
        trace, cost = _cost(_session(tmp_path, "<synthetic>"))
        assert [e.usage is not None for e in trace.events] == [False, True, False]
        assert cost.unpriced_turns == 0 and cost.unpriced_models == ()

    def test_negative_an_unknown_model_is_still_counted_unpriced(
        self, tmp_path: Path
    ) -> None:
        _, cost = _cost(_session(tmp_path, "claude-unknown-9"))
        assert cost.unpriced_turns == 1
        assert cost.unpriced_models == ("claude-unknown-9",)

    def test_boundary_tokens_on_a_synthetic_message_are_still_not_priced(
        self, tmp_path: Path
    ) -> None:
        _, with_tokens = _cost(_session(tmp_path, "<synthetic>", tokens=50))
        _, without = _cost(_session(tmp_path, "<synthetic>"))
        assert with_tokens.usd == without.usd > 0
