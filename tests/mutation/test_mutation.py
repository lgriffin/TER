"""The mutation score ratchet over ``ter.domain`` (TER-TST-015, TER-TST-016).

``scripts/mutation.py`` reads ``mutmut results`` into a score per module and
fails when one scores below ``tests/mutation/baseline.json``. The tests here
hold that arithmetic and the CI wiring on small inputs; mutmut itself runs in
the ``Mutation score`` workflow.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ter_mutation", ROOT / "scripts/mutation.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


mutation = _load_script()
Score = mutation.Score
Selection = mutation.Selection

RESULTS = """\
    ter.domain.scoring.x_ter__mutmut_1: killed
    ter.domain.scoring.x_ter__mutmut_2: survived
    ter.domain.scoring.x_ter__mutmut_3: timeout
    ter.domain.scoring.xǁScoreǁratio__mutmut_1: caught by type check
    ter.domain.lean.control.x_limits__mutmut_1: no tests
    ter.domain.lean.control.x_limits__mutmut_2: killed
    ter.domain.lean.control.x_limits__mutmut_3: skipped
    ter.domain.lean.x_register__mutmut_1: killed
""".splitlines()


@pytest.mark.req("TER-TST-015")
def test_a_score_is_the_share_of_mutants_the_tests_catch() -> None:
    assert mutation.scores(RESULTS) == {
        # killed, timed out and type-checked mutants are caught
        "ter.domain.scoring": Score(killed=3, mutants=4),
        # a mutant no test reaches is missed; a skipped one does not count
        "ter.domain.lean.control": Score(killed=1, mutants=2),
        "ter.domain.lean": Score(killed=1, mutants=1),
    }


@pytest.mark.req("TER-TST-015")
def test_a_selection_keeps_its_modules_and_a_package_its_submodules() -> None:
    package = mutation.scores(RESULTS, Selection(prefixes=("ter.domain.lean",)))
    assert set(package) == {"ter.domain.lean", "ter.domain.lean.control"}
    single = mutation.scores(RESULTS, Selection(exact=("ter.domain.lean",)))
    assert set(single) == {"ter.domain.lean"}
    assert Selection(exact=("ter.domain.lean",)).patterns() == ["ter.domain.lean.x*"]
    assert Selection(prefixes=("ter.domain",)).patterns() == ["ter.domain.*"]


@pytest.mark.req("TER-TST-015")
def test_an_unfinished_run_is_refused_unless_it_is_outside_the_selection() -> None:
    unfinished = [*RESULTS, "    ter.domain.signal.x_fire__mutmut_1: not checked"]
    with pytest.raises(SystemExit, match="did not finish"):
        mutation.scores(unfinished)
    assert "ter.domain.signal" not in mutation.scores(
        unfinished, Selection(exact=("ter.domain.scoring",))
    )


@pytest.mark.req("TER-TST-015")
def test_a_module_below_its_baseline_or_without_one_fails() -> None:
    baseline = {"ter.domain.scoring": {"score": 0.75}}
    assert mutation.drops({"ter.domain.scoring": Score(3, 4)}, baseline) == []
    assert mutation.drops({"ter.domain.scoring": Score(4, 4)}, baseline) == []
    (drop,) = mutation.drops({"ter.domain.scoring": Score(2, 4)}, baseline)
    assert "scores 50.0% (2/4), below its baseline 75.0%" in drop
    (missing,) = mutation.drops({"ter.domain.events": Score(1, 1)}, baseline)
    assert "ter.domain.events has no baseline" in missing


@pytest.mark.req("TER-TST-015")
def test_the_baseline_is_raised_but_never_lowered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(mutation, "BASELINE", baseline)
    known = {"ter.domain.a", "ter.domain.b"}
    first = {"ter.domain.a": Score(1, 2), "ter.domain.b": Score(1, 4)}
    mutation.write_baseline(
        {**first, "ter.domain.gone": Score(1, 1)}, known | {"ter.domain.gone"}
    )
    # A module not measured stays, and one no longer in the domain is dropped.
    mutation.write_baseline({"ter.domain.a": Score(2, 2)}, known)
    held = json.loads(baseline.read_text("utf-8"))["modules"]
    assert held == {
        "ter.domain.a": {"killed": 2, "mutants": 2, "score": 1.0},
        "ter.domain.b": {"killed": 1, "mutants": 4, "score": 0.25},
    }
    # A whole-domain update is held to the committed scores too.
    with pytest.raises(SystemExit, match="ter.domain.b would go down"):
        mutation.write_baseline(
            {"ter.domain.a": Score(2, 2), "ter.domain.b": Score(0, 4)}, known
        )
    assert json.loads(baseline.read_text("utf-8"))["modules"] == held


@pytest.mark.req("TER-TST-015")
def test_a_selected_module_the_run_did_not_score_fails() -> None:
    baseline = {
        "ter.domain.lean.steps": {"score": 0.8},
        "ter.domain.events": {"score": 0.9},
    }
    measured = {"ter.domain.events": Score(9, 10)}
    whole = mutation.unmeasured(measured, baseline, Selection())
    assert whole == ["ter.domain.lean.steps produced no mutant results"]
    package = Selection(prefixes=("ter.domain.lean",))
    assert mutation.unmeasured(measured, baseline, package) == whole
    assert (
        mutation.unmeasured(measured, baseline, Selection(exact=("ter.domain.events",)))
        == []
    )
    # All of a module's mutants skipped: it has no score, so it is unmeasured.
    skipped = ["    ter.domain.new.x_f__mutmut_1: skipped"]
    new = Selection(exact=("ter.domain.new",))
    assert mutation.unmeasured(mutation.scores(skipped, new), {}, new) == [
        "ter.domain.new produced no mutant results"
    ]


@pytest.mark.req("TER-TST-015")
def test_a_failing_mutmut_command_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="partial", stderr="")

    monkeypatch.setattr(mutation.subprocess, "run", failed)
    with pytest.raises(SystemExit, match="mutmut results exited 1"):
        mutation._mutmut("results", "--all", "true")


@pytest.mark.req("TER-TST-016")
def test_a_branch_mutates_only_the_domain_modules_with_functions_it_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    domain = tmp_path / "src" / "ter" / "domain"
    (domain / "lean").mkdir(parents=True)
    for name, text in {
        "__init__.py": "",
        "scoring.py": "def ter() -> int:\n    return 1\n",
        "constants.py": "LIMIT = 3\n",
        "lean/__init__.py": "class A:\n    def go(self) -> None: ...\n",
        "lean/untouched.py": "def f() -> None: ...\n",
    }.items():
        (domain / name).write_text(text, encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-qm",
        "base",
        "--allow-empty",
    )
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "domain")
    monkeypatch.setattr(mutation, "ROOT", tmp_path)
    monkeypatch.setattr(mutation, "DOMAIN", domain)
    assert mutation.changed_modules("HEAD~1") == [
        "ter.domain.lean",
        "ter.domain.lean.untouched",
        "ter.domain.scoring",
    ]
    assert mutation.changed_modules("HEAD") == []


@pytest.mark.req("TER-TST-015")
def test_the_committed_baseline_covers_every_domain_module_with_functions() -> None:
    held = mutation.load_baseline()
    with_functions = {
        module
        for module in mutation.domain_modules()
        if mutation._defines_functions(mutation.module_source(module))
    }
    assert with_functions <= set(held), sorted(with_functions - set(held))
    assert set(held) <= set(mutation.domain_modules()), "baseline names a lost module"


@pytest.mark.req("TER-TST-016")
def test_shards_split_the_domain_into_parts_of_about_the_same_size() -> None:
    baseline = {"a": {"mutants": 90}, "b": {"mutants": 50}, "c": {"mutants": 40}}
    modules = ["a", "b", "c", "new"]
    parts = [mutation.shard(modules, baseline, k, 2) for k in (1, 2)]
    assert parts == [["a", "new"], ["b", "c"]]
    assert sorted(m for part in parts for m in part) == sorted(modules)
    with pytest.raises(SystemExit, match="does not exist"):
        mutation.shard(modules, baseline, 3, 2)


def _workflow() -> dict[str, Any]:
    text = (ROOT / ".github/workflows/mutation.yml").read_text("utf-8")
    loaded: dict[str, Any] = yaml.safe_load(text)
    return loaded


@pytest.mark.req("TER-TST-016")
def test_ci_checks_changed_modules_on_a_pull_request_and_the_whole_domain_otherwise() -> (
    None
):
    workflow = _workflow()
    triggers = workflow[True]  # YAML reads the bare key `on` as true
    assert {"pull_request", "push", "schedule", "workflow_dispatch"} <= set(triggers)
    changed, domain = workflow["jobs"]["changed"], workflow["jobs"]["domain"]
    assert changed["if"] == "github.event_name == 'pull_request'"
    assert domain["if"] == "github.event_name != 'pull_request'"
    assert changed["steps"][0]["with"]["fetch-depth"] == 0  # the diff needs the base
    assert changed["steps"][-1]["run"] == (
        'python scripts/mutation.py --changed "origin/${{ github.base_ref }}"'
    )
    shards = domain["strategy"]["matrix"]["shard"]
    assert domain["steps"][-1]["run"] == (
        f"python scripts/mutation.py --shard ${{{{ matrix.shard }}}}/{len(shards)}"
    )
    assert shards == list(range(1, len(shards) + 1))


def test_the_mutation_config_mutates_the_domain_with_the_fast_suites() -> None:
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["tool"][
        "mutmut"
    ]
    assert config["source_paths"] == ["src/ter/domain/"]
    selection = config["pytest_add_cli_args_test_selection"]
    assert "not embeddings" in selection
    assert "--test-order=declared" in config["pytest_add_cli_args"]
