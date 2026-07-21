from types import SimpleNamespace
import sys, types
import numpy as np
import ter_calculator.acceleration as a


def test_session_watcher_snapshot_poll_callbacks(tmp_path, monkeypatch):
    watcher = a.SessionWatcher()
    assert watcher._snapshot(tmp_path) == {}
    old = tmp_path / "old.jsonl"
    old.write_text("x")
    watcher._known_files = {str(old): 0}
    new = tmp_path / "new.jsonl"
    new.write_text("y")
    events = []
    monkeypatch.setattr(a.time, "time", lambda: 123.0)
    watcher._poll(tmp_path, events.append)
    assert {e.event_type for e in events} == {
        a.WatchEventType.NEW_SESSION,
        a.WatchEventType.MODIFIED_SESSION,
    }
    watcher._handle_event(
        events[0], lambda e: (_ for _ in ()).throw(RuntimeError("callback"))
    )
    called = []
    watcher._analyser_fn = lambda path: called.append(path)
    watcher._handle_event(events[0], None)
    assert called
    watcher._analyser_fn = lambda path: (_ for _ in ()).throw(RuntimeError("analysis"))
    watcher._handle_event(events[0], None)


def test_session_watcher_watch_creates_and_stops(tmp_path, monkeypatch):
    watcher = a.SessionWatcher()
    missing = tmp_path / "missing"
    monkeypatch.setattr(a.time, "sleep", lambda n: watcher.stop())
    watcher.watch(missing, interval=0)
    assert missing.exists() and watcher._running is False


def test_parallel_embed_small_parallel_and_fallback(monkeypatch):
    monkeypatch.setattr(
        a,
        "_embed_single_process",
        lambda texts, model: [np.array([len(x)], dtype=np.float32) for x in texts],
    )
    assert a.parallel_embed([], n_workers=2) == []
    assert len(a.parallel_embed(["x"], n_workers=2)) == 1

    class Pool:
        def __init__(self, processes):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *x):
            pass

        def map(self, fn, args):
            return [[[1, 2] for _ in chunk] for chunk, _ in args]

    monkeypatch.setattr(a.multiprocessing, "Pool", Pool)
    out = a.parallel_embed([str(i) for i in range(100)], n_workers=2)
    assert len(out) == 100 and out[0].dtype == np.float32

    class BadPool(Pool):
        def map(self, *x):
            raise RuntimeError("bad")

    monkeypatch.setattr(a.multiprocessing, "Pool", BadPool)
    assert len(a.parallel_embed([str(i) for i in range(100)], n_workers=2)) == 100


def test_embed_worker_and_single_process_fake_module(monkeypatch):
    class Vec:
        def __init__(self, x):
            self.x = x

        def tolist(self):
            return self.x

    class Model:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, **kw):
            return [np.array([i, 1], dtype=np.float32) for i, _ in enumerate(texts)]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    assert a._embed_worker(([], "m")) == []
    assert a._embed_worker((["a", "b"], "m")) == [[0, 1], [1, 1]]
    out = a._embed_single_process(["a"], "m")
    assert out[0].dtype == np.float32
