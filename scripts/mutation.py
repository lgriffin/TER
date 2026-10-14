"""Mutation score of TER's pure domain, held to a committed baseline.

mutmut changes ``src/ter/domain`` one small edit at a time (a ``<`` for a
``<=``, a ``+ 1`` dropped, a constant changed) and runs the tests that reach
the changed function. A mutant the tests catch is killed; one they miss
survived, and is a behaviour no test pins. The score of a module is the
share of its mutants killed (a mutant no test reaches counts as missed).
``tests/mutation/baseline.json`` holds each module's score, and the check
fails when any module scores lower, so the score can only go up::

    python scripts/mutation.py                     # every domain module
    python scripts/mutation.py ter.domain.scoring  # some modules (prefixes)
    python scripts/mutation.py --changed origin/main  # modules a branch changes
    python scripts/mutation.py --update            # raise the baseline after
                                                   # adding tests

``python dev.py mutation`` runs it. The configuration (what is mutated, which
tests run) is ``[tool.mutmut]`` in pyproject.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "mutation" / "baseline.json"
DOMAIN = ROOT / "src" / "ter" / "domain"
#: mutmut names a mutant ``<module>.x_<function>__mutmut_<n>``, or
#: ``<module>.xǁ<Class>ǁ<method>__mutmut_<n>`` for a method.
_MUTANT = re.compile(r"^\s*(?P<module>[\w.]+?)\.xǁ?[^:]*__mutmut_\d+: (?P<status>.+)$")
#: Statuses that mean the tests noticed the change.
CAUGHT = {"killed", "timeout", "caught by type check"}
#: Statuses that do not count either way.
IGNORED = {"skipped"}
#: Statuses that mean the run did not finish; the check refuses them.
UNFINISHED = {"not checked", "check was interrupted by user"}


@dataclass(frozen=True)
class Score:
    killed: int
    mutants: int

    @property
    def ratio(self) -> float:
        return self.killed / self.mutants if self.mutants else 1.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "killed": self.killed,
            "mutants": self.mutants,
            "score": round(self.ratio, 4),
        }


@dataclass(frozen=True)
class Selection:
    """What a run mutates: modules and their submodules (``prefixes``), and
    single modules (``exact``); nothing named means the whole domain."""

    prefixes: tuple[str, ...] = ()
    exact: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.prefixes or self.exact)

    def patterns(self) -> list[str]:
        """mutmut's mutant-name patterns (fnmatch) for this selection."""
        return [f"{p}.*" for p in self.prefixes] + [f"{m}.x*" for m in self.exact]

    def holds(self, module: str) -> bool:
        if not self:
            return True
        return module in self.exact or any(
            module == p or module.startswith(f"{p}.") for p in self.prefixes
        )


WHOLE_DOMAIN = Selection()


def scores(
    results: Iterable[str], selection: Selection = WHOLE_DOMAIN
) -> dict[str, Score]:
    """Each selected module's score from ``mutmut results --all true`` lines
    (mutmut lists the mutants of modules it did not run as not checked)."""
    counts: dict[str, Counter[str]] = {}
    for line in results:
        match = _MUTANT.match(line)
        if match is None or not selection.holds(match["module"]):
            continue
        status = match["status"].strip()
        if status in UNFINISHED:
            raise SystemExit(f"mutation run did not finish: {line.strip()}")
        if status in IGNORED:
            continue
        counter = counts.setdefault(match["module"], Counter())
        counter["mutants"] += 1
        counter["killed"] += status in CAUGHT
    return {
        module: Score(killed=c["killed"], mutants=c["mutants"])
        for module, c in sorted(counts.items())
    }


def drops(
    measured: Mapping[str, Score], baseline: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """Every module that scores below its baseline, or has none."""
    found = []
    for module, score in measured.items():
        held = baseline.get(module)
        if held is None:
            found.append(f"{module} has no baseline: run with --update and commit it")
        elif round(score.ratio, 4) < float(held["score"]):
            found.append(
                f"{module} scores {score.ratio:.1%} ({score.killed}/{score.mutants}),"
                f" below its baseline {float(held['score']):.1%}"
            )
    return found


def domain_modules() -> list[str]:
    """Every module under ``src/ter/domain``, as ``ter.domain.x.y``."""
    modules = []
    for path in sorted(DOMAIN.rglob("*.py")):
        parts = path.relative_to(ROOT / "src").with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


def changed_modules(base: str) -> list[str]:
    """The domain modules a branch changes against ``base``."""
    names = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD", "--", "src/ter/domain"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.split()
    known = set(domain_modules())
    picked = []
    for name in names:
        path = ROOT / name
        parts = Path(name).relative_to("src").with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        # mutmut mutates functions; a module with none has no mutants.
        if module in known and path.exists() and _defines_functions(path):
            picked.append(module)
    return sorted(set(picked))


def module_source(module: str) -> Path:
    """The file of ``ter.domain.x``: ``x.py``, or ``x/__init__.py``."""
    path = ROOT / "src" / Path(*module.split("."))
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def _defines_functions(path: Path) -> bool:
    return (
        re.search(r"^\s*(async\s+)?def ", path.read_text("utf-8"), re.MULTILINE)
        is not None
    )


def shard(
    modules: Iterable[str], baseline: Mapping[str, Mapping[str, float]], k: int, n: int
) -> list[str]:
    """Shard ``k`` of ``n`` (1-based): modules dealt heaviest first to the
    shard with the fewest baseline mutants, so the shards take about as long."""
    if not 1 <= k <= n:
        raise SystemExit(f"shard {k}/{n} does not exist")
    weight = {m: int(baseline.get(m, {}).get("mutants", 0)) for m in modules}
    loads = [0] * n
    dealt: list[list[str]] = [[] for _ in range(n)]
    for module in sorted(weight, key=lambda m: (-weight[m], m)):
        lightest = loads.index(min(loads))
        dealt[lightest].append(module)
        loads[lightest] += weight[module]
    return sorted(dealt[k - 1])


def _shard_arg(text: str) -> tuple[int, int]:
    k, _, n = text.partition("/")
    try:
        return int(k), int(n)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not K/N") from None


def load_baseline() -> dict[str, dict[str, float]]:
    if not BASELINE.exists():
        return {}
    data: dict[str, dict[str, float]] = json.loads(
        BASELINE.read_text(encoding="utf-8")
    )["modules"]
    return data


def write_baseline(measured: Mapping[str, Score], keep: bool) -> None:
    """Record ``measured``; with ``keep``, modules not measured stay as they
    were and none goes down."""
    modules = dict(load_baseline()) if keep else {}
    for module, score in measured.items():
        held = modules.get(module)
        if held is not None and round(score.ratio, 4) < float(held["score"]):
            raise SystemExit(f"{module} would go down: fix the tests instead")
        modules[module] = score.as_dict()
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "tool": "mutmut",
        "about": "Mutation score per ter.domain module; scripts/mutation.py",
        "modules": dict(sorted(modules.items())),
    }
    BASELINE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _mutmut(*args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "mutmut", *args],
        cwd=ROOT,
        check=False,
        capture_output=args[0] == "results",
        text=True,
    )
    if args[0] == "results":
        return done.stdout
    if done.returncode != 0:
        raise SystemExit(f"mutmut {args[0]} exited {done.returncode}")
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("modules", nargs="*", help="module prefixes to mutate")
    parser.add_argument("--changed", metavar="BASE", help="the modules changed vs BASE")
    parser.add_argument(
        "--shard", type=_shard_arg, metavar="K/N", help="one of N parts of the domain"
    )
    parser.add_argument("--update", action="store_true", help="raise the baseline")
    parser.add_argument(
        "--max-children", default=str(os.cpu_count() or 1), help="mutmut workers"
    )
    args = parser.parse_args(argv)

    exact: list[str] = changed_modules(args.changed) if args.changed else []
    if args.shard:
        with_functions = [
            m for m in domain_modules() if _defines_functions(module_source(m))
        ]
        exact += shard(with_functions, load_baseline(), *args.shard)
    selection = Selection(prefixes=tuple(args.modules), exact=tuple(exact))
    if args.changed and not selection:
        print("no domain module with functions changed: nothing to mutate")
        return 0
    _mutmut("run", "--max-children", args.max_children, *selection.patterns())
    measured = scores(_mutmut("results", "--all", "true").splitlines(), selection)
    for module, score in measured.items():
        print(f"{module:<45} {score.ratio:6.1%}  {score.killed}/{score.mutants}")
    if args.update:
        write_baseline(measured, keep=bool(selection))
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0
    found = drops(measured, load_baseline())
    for problem in found:
        print(f"FAILED {problem}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
