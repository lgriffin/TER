"""The test-first steps (TER-TST-019, TER-TST-020).

``python dev.py red`` passes only when the new tests fail, and ``python dev.py
green`` only when they pass and each cites its requirement. The tests run
``scripts/tdd.py`` on small test files written in a temporary directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
MARK = "@pytest.mark." + "req"  # spelled apart so the citation scan skips it


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ter_tdd", ROOT / "scripts/tdd.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tdd = _load_script()


def _write(tmp_path: Path, body: str) -> str:
    (tmp_path / "test_new.py").write_text("import pytest\n\n" + body, encoding="utf-8")
    return "test_new.py"


@pytest.mark.req("TER-TST-019")
@pytest.mark.parametrize(
    ("body", "expected", "message"),
    [
        ("def test_new():\n    assert 1 == 2\n", 0, "RED:"),
        ("def test_new():\n    assert 1 == 1\n", 1, "pass already"),
        ("import not_a_module_anywhere\n\ndef test_new(): pass\n", 1, "did not run"),
        ("def helper(): pass\n", 1, "no test was selected"),
        (
            "@pytest.fixture\ndef broken():\n    raise RuntimeError\n\n"
            "def test_new(broken):\n    assert 1 == 2\n",
            1,
            "only setup or teardown errored",
        ),
        (
            "@pytest.fixture\ndef broken():\n    raise RuntimeError\n\n"
            "def test_new(broken): pass\n\ndef test_other():\n    assert 1 == 2\n",
            0,
            "RED:",
        ),
    ],
    ids=[
        "fails",
        "passes-already",
        "collection-error",
        "no-test",
        "setup-error-only",
        "one-fails-beside-a-setup-error",
    ],
)
def test_red_holds_only_when_the_new_test_fails(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    body: str,
    expected: int,
    message: str,
) -> None:
    target = _write(tmp_path, body)
    assert tdd.red([target], cwd=tmp_path) == expected
    out, err = capfd.readouterr()
    assert message in out + err


@pytest.mark.req("TER-TST-020")
def test_green_holds_when_the_tests_pass_and_cite_a_requirement(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    target = _write(tmp_path, f'{MARK}("TER-AAA-001")\ndef test_new(): pass\n')
    assert tdd.green([target], cwd=tmp_path) == 0
    assert "GREEN:" in capfd.readouterr().out


@pytest.mark.req("TER-TST-020")
def test_green_fails_on_a_failing_test(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    target = _write(tmp_path, f'{MARK}("TER-AAA-001")\ndef test_new(): assert False\n')
    assert tdd.green([target], cwd=tmp_path) == 1
    assert "NOT GREEN: pytest exited 1" in capfd.readouterr().err


@pytest.mark.req("TER-TST-020")
def test_green_names_a_passing_test_that_cites_no_requirement(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    target = _write(
        tmp_path,
        f'{MARK}("TER-AAA-001")\ndef test_cited(): pass\n\ndef test_bare(): pass\n',
    )
    assert tdd.green([target], cwd=tmp_path) == 1
    err = capfd.readouterr().err
    assert "test_new.py::test_bare" in err and "test_cited" not in err


@pytest.mark.req("TER-TST-020")
@pytest.mark.parametrize(
    ("body", "outcome"),
    [
        ("@pytest.mark.skip\ndef test_new(): pass\n", "skipped"),
        ("@pytest.mark.xfail\ndef test_new(): assert False\n", "xfailed"),
    ],
    ids=["skipped", "expected-failure"],
)
def test_green_fails_on_a_test_that_did_not_pass(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], body: str, outcome: str
) -> None:
    target = _write(tmp_path, f'{MARK}("TER-AAA-001")\n' + body)
    assert tdd.green([target], cwd=tmp_path) == 1
    assert f"test_new.py::test_new ({outcome})" in capfd.readouterr().err


@pytest.mark.req("TER-TST-020")
def test_green_names_uncited_tests_even_when_another_fails(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    target = _write(
        tmp_path,
        f'{MARK}("TER-AAA-001")\ndef test_cited(): assert False\n\n'
        "def test_bare(): pass\n",
    )
    assert tdd.green([target], cwd=tmp_path) == 1
    err = capfd.readouterr().err
    assert "test_new.py::test_cited (failed)" in err
    assert "cite no requirement" in err and "test_new.py::test_bare" in err


def test_uncited_reads_a_trace_document() -> None:
    trace = {
        "requirements": {"TER-AAA-001": [{"nodeid": "t.py::a", "outcome": "passed"}]}
    }
    assert tdd.uncited(["t.py::a", "t.py::b"], trace) == ["t.py::b"]
    assert tdd.uncited(["t.py::a"], {}) == ["t.py::a"]
    assert tdd.not_passed(["t.py::a", "t.py::b"], trace) == []


def test_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert tdd.main(["blue", "t.py"]) == 2
    assert tdd.main(["red"]) == 2
    assert "usage:" in capsys.readouterr().err
