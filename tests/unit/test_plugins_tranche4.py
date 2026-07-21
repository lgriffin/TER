from argparse import Namespace
from types import SimpleNamespace
import sys
import types
import pytest
import ter_calculator.plugins as p


class Detector:
    name = "det"
    description = "d"

    def detect(self, spans):
        return []


class Formatter:
    format_name = "fmt"

    def format_result(self, result):
        return "r"

    def format_comparison(self, results):
        return "c"


class Middleware:
    name = "mw"

    def pre_process(self, session):
        return session

    def post_process(self, result):
        return result


def test_config_resolution_merge_and_validation(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    (home / ".terrc").write_text('[output]\nformat="json"\n')
    monkeypatch.setattr(p.Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(cwd)
    cfg = p.TERConfig.load()
    assert cfg.output.format == "json"
    merged = cfg.merge_cli_args(
        Namespace(
            similarity_threshold=0.2,
            confidence_threshold=0.3,
            restatement_threshold=0.4,
            phase_weights="0.1,0.2,0.7",
            format="text",
            color=False,
        )
    )
    assert (
        merged.thresholds.similarity == 0.2
        and merged.weights.generation == 0.7
        and not merged.output.color
    )
    kept = cfg.merge_cli_args(Namespace())
    assert kept.output.format == "json" and kept.plugins is not cfg.plugins


def test_registry_type_errors_duplicates_and_dispatch(caplog):
    r = p.PluginRegistry()
    r.reset()
    for method in (
        r.register_waste_detector,
        r.register_formatter,
        r.register_middleware,
    ):
        with pytest.raises(TypeError):
            method(object())
    d = Detector()
    f = Formatter()
    m = Middleware()
    r.register_waste_detector(d)
    r.register_waste_detector(Detector())
    r.register_formatter(f)
    replacement = Formatter()
    r.register_formatter(replacement)
    r.register_middleware(m)
    r.register_middleware(Middleware())
    assert (
        len(r.get_waste_detectors()) == 1
        and r.get_formatter("fmt") is replacement
        and len(r.get_middleware()) == 1
    )
    r._register_by_kind(object(), "unknown", "ep")
    assert "Unknown plugin kind" in caplog.text


def test_discover_plugins_modern_legacy_and_failures(monkeypatch):
    r = p.PluginRegistry()
    r.reset()

    class EP:
        def __init__(self, name, obj=None, fail=False):
            self.name = name
            self.obj = obj
            self.fail = fail

        def load(self):
            if self.fail:
                raise RuntimeError("bad")
            return self.obj

    groups = {
        "x.waste_detectors": [EP("d", Detector)],
        "x.formatters": [EP("f", Formatter())],
        "x.middleware": [EP("m", Middleware), EP("bad", fail=True)],
    }
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda **kw: groups.get(kw["group"], []))
    r.discover_plugins("x")
    assert (
        len(r.get_waste_detectors()) == 1
        and r.get_formatter("fmt")
        and len(r.get_middleware()) == 1
    )
    r.reset()

    def legacy(**kw):
        if kw:
            raise TypeError
        return groups

    monkeypatch.setattr(md, "entry_points", legacy)
    r.discover_plugins("x")
    assert len(r.get_waste_detectors()) == 1


def test_load_from_config_success_and_failures(tmp_path, monkeypatch):
    r = p.PluginRegistry()
    r.reset()
    mod = types.ModuleType("fake_plugins")
    mod.Detector = Detector
    mod.Formatter = Formatter
    mod.Middleware = Middleware
    sys.modules["fake_plugins"] = mod
    cfg = tmp_path / "ter.toml"
    cfg.write_text(
        '[plugins]\nwaste_detectors=["fake_plugins.Detector","bad"]\nformatters=["fake_plugins.Formatter"]\nmiddleware=["fake_plugins.Middleware"]\n'
    )
    r.load_from_config(cfg)
    assert (
        len(r.get_waste_detectors()) == 1
        and r.get_formatter("fmt")
        and len(r.get_middleware()) == 1
    )
    bad = tmp_path / "bad.toml"
    bad.write_text("[")
    r.load_from_config(bad)


def test_decorator_failure_paths(monkeypatch):
    class Bad:
        def __init__(self):
            raise RuntimeError("no")

    assert p.waste_detector(Bad) is Bad
    assert p.output_formatter(Bad) is Bad
