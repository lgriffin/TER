from types import SimpleNamespace
import json
import sys
import pytest
import ter_calculator.cli as cli


def ns(**kw):
    d = dict(
        latest=False,
        session_path="s.jsonl",
        output_format="text",
        quiet=True,
        verbose=False,
        group=False,
        mode="relaxed",
        budget=100,
        relevance_threshold=0.1,
        project_path=None,
        poll_interval=0.01,
        stream=True,
        log_file=None,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def test_context_dispatch_and_resolve(monkeypatch, capsys):
    assert cli._cmd_context(ns(context_command=None)) == 1
    assert cli._cmd_context(ns(context_command="bad")) == 1
    monkeypatch.setattr(cli, "_cmd_context_store", lambda a: 7)
    assert cli._cmd_context(ns(context_command="store")) == 7
    import ter_calculator.loader as ld

    monkeypatch.setattr(ld, "find_latest_session", lambda p: "/tmp/latest.jsonl")
    assert (
        cli._resolve_session_path(ns(latest=True, session_path="x"))
        == "/tmp/latest.jsonl"
    )


def _patch_session(monkeypatch):
    import ter_calculator.loader as ld

    sess = SimpleNamespace(session_id="sess")
    monkeypatch.setattr(ld, "load_session", lambda p: sess)
    monkeypatch.setattr(ld, "segment_spans", lambda s: [1, 2])
    return sess


def test_context_store(monkeypatch, capsys):
    assert cli._cmd_context_store(ns(session_path=None)) == 1
    _patch_session(monkeypatch)
    import ter_calculator.fragment_store as fs

    class Store:
        def count(self):
            return 3

        def close(self):
            self.closed = True

    class Engine:
        def __init__(self, s):
            pass

        def shard(self, sp, sid):
            return [
                SimpleNamespace(origin_session="sess"),
                SimpleNamespace(origin_session="other"),
            ]

    monkeypatch.setattr(fs, "FragmentStore", Store)
    monkeypatch.setattr(fs, "FragmentShardingEngine", Engine)
    assert cli._cmd_context_store(ns()) == 0
    assert "Fragments created" in capsys.readouterr().out


def test_context_graph_text_json(monkeypatch, capsys):
    _patch_session(monkeypatch)
    import ter_calculator.fragment_store as fs, ter_calculator.context_graph as cg

    class Store:
        def close(self):
            pass

    class Engine:
        def __init__(self, s):
            pass

        def shard(self, *a):
            return [1]

    class Edge:
        source_id = "a" * 20
        target_id = "b" * 20
        edge_type = SimpleNamespace(value="dependency")
        weight = 0.5

    class Graph:
        node_count = 2
        edge_count = 1

        def build_from_session(self, f):
            pass

        def all_edges(self):
            return [Edge()]

        def detect_cycles(self):
            return [["x"]]

        def topological_sort(self):
            return ["a", "b"]

        def close(self):
            pass

    monkeypatch.setattr(fs, "FragmentStore", Store)
    monkeypatch.setattr(fs, "FragmentShardingEngine", Engine)
    monkeypatch.setattr(cg, "ContextGraph", Graph)
    assert cli._cmd_context_graph(ns(output_format="json")) == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == 2
    assert cli._cmd_context_graph(ns(output_format="text")) == 0
    assert "Cycles detected" in capsys.readouterr().out


def test_context_optimize_delta_check(monkeypatch, capsys):
    sess = _patch_session(monkeypatch)
    import ter_calculator.intent as intent, ter_calculator.fragment_store as fs, ter_calculator.context_graph as cg, ter_calculator.budget_optimizer as bo

    monkeypatch.setattr(intent, "extract_intent", lambda s: "intent")

    class Store:
        def close(self):
            pass

        def find_by_session(self, s):
            return [1]

    class Engine:
        def __init__(self, s):
            pass

        def shard(self, *a):
            return [SimpleNamespace(id="f")]

    class Graph:
        def build_from_session(self, f):
            pass

        def close(self):
            pass

    monkeypatch.setattr(fs, "FragmentStore", Store)
    monkeypatch.setattr(fs, "FragmentShardingEngine", Engine)
    monkeypatch.setattr(cg, "ContextGraph", Graph)
    monkeypatch.setattr(
        bo,
        "recommend_context",
        lambda *a, **k: SimpleNamespace(
            budget_ceiling=100,
            selected_fragment_ids=["f"],
            budget_used=10,
            total_relevance=0.8,
            pruned_count=1,
            reasoning="ok",
        ),
    )
    assert cli._cmd_context_optimize(ns()) == 0
    import ter_calculator.delta_composer as dc

    monkeypatch.setattr(dc, "LocalCache", lambda: object())
    monkeypatch.setattr(
        dc,
        "create_template_from_session",
        lambda *a: SimpleNamespace(required_fragment_ids=["f"]),
    )
    monkeypatch.setattr(
        dc,
        "compose_delta",
        lambda *a: SimpleNamespace(
            manifest=SimpleNamespace(cache_hits=1, cache_misses=0),
            delta_fragments=[],
            total_tokens_saved=9,
            compression_ratio=0.5,
        ),
    )
    assert cli._cmd_context_delta(ns()) == 0
    import ter_calculator.loader as ld, ter_calculator.consistency as co

    monkeypatch.setattr(ld, "discover_subagents", lambda p: ["sub"])
    monkeypatch.setattr(co, "monitor_group_consistency", lambda *a: [])
    assert cli._cmd_context_check(ns(group=True)) == 0
    assert "No version skew" in capsys.readouterr().out


def test_watch_errors_and_stream(monkeypatch, tmp_path, capsys):
    import ter_calculator.real_time as rt

    monkeypatch.setattr(rt, "load_embedding_model", lambda: object())
    assert cli._cmd_watch(ns(project_path=None, latest=False)) == 1
    assert cli._cmd_watch(ns(project_path=str(tmp_path / "missing"))) == 1
    f = tmp_path / "s.jsonl"
    f.write_text("")

    class Monitor:
        def __init__(self, *a, on_signal=None, **k):
            self.on_signal = on_signal

        def run(self):
            raise KeyboardInterrupt

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(rt, "SessionMonitor", Monitor)
    assert cli._cmd_watch(ns(project_path=str(f), stream=True)) == 0
    assert "Stopped monitoring" in capsys.readouterr().out


def test_watch_import_and_init_errors(monkeypatch, tmp_path):
    import ter_calculator.real_time as rt

    monkeypatch.setattr(
        rt,
        "load_embedding_model",
        lambda: (_ for _ in ()).throw(ImportError("no model")),
    )
    assert cli._cmd_watch(ns(project_path=str(tmp_path))) == 1
    monkeypatch.setattr(rt, "load_embedding_model", lambda: object())
    f = tmp_path / "s.jsonl"
    f.write_text("")
    monkeypatch.setattr(
        rt,
        "SessionMonitor",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("init")),
    )
    assert cli._cmd_watch(ns(project_path=str(f))) == 1
