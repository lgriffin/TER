from pathlib import Path

import pytest

from ter_calculator.plugins import (
    PluginRegistry,
    TERConfig,
    _first_not_none,
    _import_dotted_path,
    _parse_phase_weights,
    output_formatter,
    waste_detector,
)


def test_config_defaults_explicit_invalid_and_search(tmp_path, monkeypatch):
    assert TERConfig.load(tmp_path / "missing").thresholds.similarity == 0.4
    bad = tmp_path / "bad.toml"
    bad.write_text("[")
    assert TERConfig.load(bad).config_path is None
    good = tmp_path / "ter.toml"
    good.write_text("""
[thresholds]
similarity=0.7
confidence=0.8
restatement=0.9
[weights]
reasoning=0.2
tool_use=0.5
generation=0.3
[output]
format="json"
color=false
[plugins]
waste_detectors=["x.y"]
formatters=["a.b"]
middleware=["m.n"]
""")
    cfg = TERConfig.load(good)
    assert cfg.thresholds.similarity == 0.7 and cfg.output.format == "json"
    assert cfg.plugins.middleware == ["m.n"] and cfg.config_path == good
    monkeypatch.chdir(tmp_path)
    assert TERConfig.load().config_path == good


def test_helpers():
    assert _first_not_none(None, None, 3, 4) == 3
    with pytest.raises(ValueError):
        _first_not_none(None)
    assert _parse_phase_weights("0.2,0.3,0.5") == (0.2, 0.3, 0.5)
    assert _parse_phase_weights([0.2, 0.3, 0.5]) == (0.2, 0.3, 0.5)
    with pytest.raises(ValueError):
        _parse_phase_weights("1,2")
    with pytest.raises(ValueError):
        _parse_phase_weights("a,b,c")
    assert _import_dotted_path("pathlib.Path") is Path
    with pytest.raises((ImportError, AttributeError)):
        _import_dotted_path("pathlib.Nope")


def test_registry_registration_lookup_reset_and_decorators():
    r = PluginRegistry()
    r.reset()

    class D:
        name = "det"
        description = "test"

        def detect(self, spans):
            return []

    class F:
        format_name = "fmt"

        def format_result(self, result):
            return "x"

        def format_comparison(self, results):
            return "x"

    class M:
        name = "mw"

        def pre_process(self, x):
            return x

        def post_process(self, x):
            return x

    d, f, m = D(), F(), M()
    r.register_waste_detector(d)
    r.register_formatter(f)
    r.register_middleware(m)
    assert r.get_waste_detectors() == [d]
    assert r.get_formatter("fmt") is f and r.get_formatter("none") is None
    assert r.get_middleware() == [m]
    r.reset()
    assert r.get_waste_detectors() == []

    @waste_detector
    class DecoratedD(D):
        name = "decorated-det"

    @output_formatter
    class DecoratedF(F):
        format_name = "decorated-fmt"

    assert any(x.name == "decorated-det" for x in r.get_waste_detectors())
    assert r.get_formatter("decorated-fmt") is not None
    r.reset()
