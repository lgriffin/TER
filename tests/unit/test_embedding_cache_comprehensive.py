import builtins
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import ter_calculator.embedding_cache as ec
from ter_calculator.models import SpanPhase, TokenSpan


def span(text, phase=SpanPhase.REASONING, pos=0, tokens=10):
    return TokenSpan(
        text=text,
        phase=phase,
        position=pos,
        token_count=tokens,
        source_message_uuid=str(pos),
    )


class FakeModel:
    def __init__(self):
        self.calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.vstack(
            [
                np.full(ec.EMBEDDING_DIM, i + 1, dtype=np.float64)
                for i, _ in enumerate(texts)
            ]
        )


def test_estimate_tokens_tiktoken_and_fallback(monkeypatch):
    class Enc:
        def encode(self, text):
            return [1, 2, 3]

    monkeypatch.setattr(ec, "_TIKTOKEN_ENC", Enc())
    assert ec.estimate_tokens("abc") == 3
    monkeypatch.setattr(ec, "_TIKTOKEN_ENC", None)
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    assert ec.estimate_tokens("abcdefgh") == 2
    assert ec.estimate_tokens("") == 1


def test_model_loader_caches_and_sets_environment(monkeypatch):
    ec._MODEL_CACHE.clear()
    made = []
    mod = types.ModuleType("sentence_transformers")

    class ST:
        def __init__(self, name):
            made.append(name)

    mod.SentenceTransformer = ST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    first = ec.get_embedding_model("demo")
    second = ec.get_embedding_model("demo")
    assert first is second and made == ["demo"]


def test_model_loader_helpful_import_error(monkeypatch):
    ec._MODEL_CACHE.clear()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        ec.get_embedding_model("missing")


def test_merge_adjacent_spans_boundaries_and_empty():
    assert ec.merge_adjacent_spans([]) == []
    spans = [
        span("a", pos=4, tokens=2),
        span("b", pos=8, tokens=3),
        span("c", SpanPhase.TOOL_USE, 11, 4),
    ]
    merged = ec.merge_adjacent_spans(spans)
    assert len(merged) == 2
    assert merged[0].text == "a b"
    assert merged[0].source_indices == (0, 1)
    assert (
        merged[0].start_position,
        merged[0].end_position,
        merged[0].total_token_count,
    ) == (4, 8, 5)
    assert merged[1].phase == SpanPhase.TOOL_USE.value


def test_cache_complete_lifecycle_and_bad_files(tmp_path, monkeypatch):
    cache = ec.EmbeddingCache(tmp_path)
    v = np.arange(4, dtype=np.float64)
    cache.put("hello", v)
    cache.flush()
    assert cache.size == 1
    assert cache.get("hello").dtype == np.float32
    assert cache.get("absent") is None
    assert cache.get_many(["hello", "absent"])["absent"] is None
    cache.put_many(["a", "b"], np.ones((2, 4), dtype=np.float32))
    cache.flush()
    reloaded = ec.EmbeddingCache(tmp_path)
    assert reloaded.size == 3
    h = cache.content_hash("hello")
    cache._npy_path(h).write_text("not numpy")
    assert cache.get("hello") is None
    assert not cache._npy_path(h).exists()
    cache.clear()
    assert cache.size == 0 and tmp_path.exists()


def test_cache_corrupt_index_and_flush_oserror(tmp_path, monkeypatch):
    (tmp_path / "index.json").write_text("{")
    cache = ec.EmbeddingCache(tmp_path)
    assert cache.size == 0

    def boom(*a, **k):
        raise OSError("no")

    monkeypatch.setattr(builtins, "open", boom)
    cache.flush()  # handled internally


def test_filter_short_spans_and_defaults():
    spans = [span("x", tokens=1), span("long", tokens=10)]
    emb, skipped = ec.filter_short_spans(spans, 10)
    assert emb == [spans[1]]
    assert skipped[0].span_index == 0
    assert skipped[0].default_label == "low_signal"


def test_detect_device_cuda_mps_cpu_and_missing_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda n: "GPU"
    )
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert ec.detect_device().device == "cuda"
    torch.cuda.is_available = lambda: False
    torch.backends.mps.is_available = lambda: True
    assert ec.detect_device().device == "mps"
    torch.backends.mps.is_available = lambda: False
    assert ec.detect_device().device == "cpu"
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def test_configure_and_compute_embeddings():
    model = FakeModel()
    cfg = ec.DeviceConfig("cpu", "CPU", 3)
    assert ec.configure_model_device(model, cfg) is model
    assert model.device == "cpu"
    empty = ec.compute_batch_embeddings([], model)
    assert empty.shape == (0, ec.EMBEDDING_DIM)
    out = ec.compute_batch_embeddings(
        ["a", "b"], model, device_config=cfg, show_progress=True
    )
    assert out.shape == (2, ec.EMBEDDING_DIM) and out.dtype == np.float32
    assert model.calls[-1][1]["batch_size"] == 3
    out = ec.compute_batch_embeddings(["a"], model, batch_size=7)
    assert model.calls[-1][1]["batch_size"] == 7


def test_embed_spans_merge_cache_and_skips(tmp_path):
    model = FakeModel()
    cache = ec.EmbeddingCache(tmp_path)
    spans = [
        span("tiny", tokens=1),
        span("one", tokens=10),
        span("two", tokens=11),
        span("tool", SpanPhase.TOOL_USE, 3, 12),
    ]
    result, skipped = ec.embed_spans(
        spans, model, cache=cache, min_token_count=10, merge=True
    )
    assert skipped[0].span_index == 0
    assert np.all(result[0].embedding == 0)
    assert result[1].embedding is result[2].embedding
    assert len(model.calls) == 1 and len(model.calls[0][0]) == 2
    # second invocation is fully cached
    ec.embed_spans(spans, model, cache=cache, min_token_count=10, merge=True)
    assert len(model.calls) == 1


def test_embed_spans_without_merge_and_all_skipped(tmp_path):
    model = FakeModel()
    cache = ec.EmbeddingCache(tmp_path)
    spans = [span("a", tokens=10), span("b", tokens=10)]
    ec.embed_spans(spans, model, cache=cache, merge=False, min_token_count=10)
    assert len(model.calls[0][0]) == 2
    model2 = FakeModel()
    short = [span("x", tokens=1)]
    result, skipped = ec.embed_spans(short, model2, cache=cache, min_token_count=5)
    assert len(skipped) == 1 and model2.calls == []
