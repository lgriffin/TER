"""``ter-req``: lint, trace and report on the EARS requirement catalogue.

Usage::

    ter-req lint   [--catalogue requirements] [--tests tests] [--strict]
    ter-req trace  --results req-trace.json [--gate L0] [--summary PATH]
    ter-req points [--out docs/ter4/points.md] [--check]
    ter-req report [--results req-trace.json] [--gate L0] [--out PATH]

``lint`` checks EARS grammar, the vision points in ``points.yaml`` (definition
of done, rules, two-way links, verification of done points) and, with
``--tests``, that every ``@pytest.mark.req`` in the test tree cites a known id
(a static backward trace). ``points`` writes the living points index, or with
``--check`` fails when the committed index is stale. ``trace`` is the CI gate: it fails when a verified requirement at or
below the gate level has no passing test, or a test cites an unknown id.
Exit status is 0 on success, 1 when a check fails and 2 on bad input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from ter.adapters.driven.requirements_yaml import (
    Catalogue,
    CatalogueError,
    RepositoryChecks,
    load_catalogue,
)
from ter.domain.maturity import Maturity
from ter.domain.points import lint_points, summarise
from ter.domain.requirements import (
    TestOutcome,
    backward_trace,
    lint_catalogue,
    trace,
)

from .req_report import render_markdown, render_points_index

DEFAULT_CATALOGUE = Path("requirements")
DEFAULT_INDEX = Path("docs/ter4/points.md")
MARKER = re.compile(r"mark\.req\(\s*((?:[\"'][^\"']*[\"']\s*,?\s*)+)\)")
MARKER_ID = re.compile(r"[\"']([^\"']*)[\"']")
#: A BDD scenario's citation: ``@req:TER-ANL-001`` (see ``pytest_req``).
FEATURE_TAG = re.compile(r"(?<!\S)@req:(\S+)")

Results = dict[str, list[TestOutcome]]


class InputError(ValueError):
    """A results file is missing or malformed."""


def scan_citations(root: Path) -> dict[str, list[str]]:
    """Find ``mark.req("...")`` citations in Python files under ``root``,
    and ``@req:...`` tags in its ``.feature`` files.

    Returns id -> ["path:line", ...].
    """
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in MARKER.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            for rid in MARKER_ID.findall(match.group(1)):
                found.setdefault(rid, []).append(f"{path.as_posix()}:{line}")
    for path in sorted(root.rglob("*.feature")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, text in enumerate(lines, start=1):
            if text.lstrip().startswith("#"):
                continue
            for rid in FEATURE_TAG.findall(text):
                found.setdefault(rid, []).append(f"{path.as_posix()}:{number}")
    return sorted_found(found)


def sorted_found(found: dict[str, list[str]]) -> dict[str, list[str]]:
    return {rid: found[rid] for rid in sorted(found)}


def load_results(path: Path) -> Results:
    """Read the JSON written by ``pytest --req-trace``."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{path}: cannot read trace results: {exc}") from exc
    mapping = document.get("requirements") if isinstance(document, dict) else None
    if not isinstance(mapping, dict):
        raise InputError(f"{path}: expected a 'requirements' mapping")
    results: Results = {}
    for rid, entries in mapping.items():
        if not isinstance(entries, list):
            raise InputError(f"{path}: {rid} must map to a list")
        outcomes = []
        for entry in entries:
            if not isinstance(entry, dict) or not {"nodeid", "outcome"} <= set(entry):
                raise InputError(f"{path}: {rid} has an entry without nodeid/outcome")
            outcomes.append(TestOutcome(str(entry["nodeid"]), str(entry["outcome"])))
        results[str(rid)] = outcomes
    return results


def _cmd_lint(args: argparse.Namespace, catalogue: Catalogue) -> int:
    issues = lint_catalogue(catalogue.requirements, catalogue.vocabulary)
    if catalogue.points:
        checks = RepositoryChecks(args.root)
        issues += lint_points(catalogue.points, catalogue.requirements, checks)
    for issue in issues:
        source = catalogue.sources.get(issue.requirement_id)
        prefix = f"{source}: " if source else ""
        print(f"{prefix}{issue}")
    errors = [i for i in issues if args.strict or not i.warning]
    warnings = len(issues) - len(errors)
    failed = bool(errors)
    if args.tests is not None:
        unknown = backward_trace(catalogue.requirements, scan_citations(args.tests))
        for rid, where in unknown:
            print(
                f"{rid}: [TRACE-UNKNOWN] cited but not in the catalogue: {', '.join(where)}"
            )
        failed = failed or bool(unknown)
    print(
        f"{'FAIL' if failed else 'OK'}: {len(catalogue.requirements)} requirements, "
        f"{len(catalogue.points)} points, {len(errors)} errors, {warnings} warnings"
    )
    return 1 if failed else 0


def _cmd_trace(args: argparse.Namespace, catalogue: Catalogue) -> int:
    results = load_results(args.results)
    report = trace(catalogue.requirements, results, args.gate)
    for gap in report.forward_gaps:
        print(
            f"{gap.requirement.id}: [TRACE-FORWARD] verified at {gap.requirement.level.code}: {gap.reason}"
        )
    for rid, where in report.unknown_ids:
        print(
            f"{rid}: [TRACE-UNKNOWN] cited but not in the catalogue: {', '.join(where)}"
        )
    for rid in report.promotable:
        print(f"{rid}: [TRACE-PROMOTE] planned, but a citing test passes")
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(
                render_markdown(catalogue.requirements, results, report) + "\n"
            )
    gated = sum(
        1 for r in catalogue.requirements if r.verified and args.gate.permits(r.level)
    )
    verdict = "OK" if report.ok else "FAIL"
    print(f"{verdict}: gate {args.gate.code}, {gated} verified requirements checked")
    return 0 if report.ok else 1


def _cmd_points(args: argparse.Namespace, catalogue: Catalogue) -> int:
    if not catalogue.points:
        raise InputError(f"{args.catalogue}: no points.yaml in the catalogue")
    text = render_points_index(catalogue.points, catalogue.requirements)
    summary = summarise(catalogue.points)
    counts = ", ".join(f"{n} {s.value}" for s, n in summary.by_status.items())
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        if current != text:
            print(f"FAIL: {args.out} is stale; run `ter-req points` and commit it")
            return 1
        print(f"OK: {args.out} is up to date ({counts})")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({counts})")
    return 0


def _cmd_report(args: argparse.Namespace, catalogue: Catalogue) -> int:
    results: Results = load_results(args.results) if args.results else {}
    report = trace(catalogue.requirements, results, args.gate)
    text = render_markdown(
        catalogue.requirements, results, report, with_results=args.results is not None
    )
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def _add_location_options(parser: argparse.ArgumentParser, default: object) -> None:
    """--catalogue and --root, accepted before or after the subcommand."""
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=DEFAULT_CATALOGUE if default is None else default,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".") if default is None else default,
        help="repository root for test/ci checks",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ter-req",
        description="Lint, trace and report on the TER EARS requirement catalogue.",
    )
    _add_location_options(parser, None)
    sub = parser.add_subparsers(dest="command", required=True)

    lint = sub.add_parser("lint", help="check EARS grammar and catalogue invariants")
    lint.add_argument(
        "--tests", type=Path, default=None, help="also scan this tree for req markers"
    )
    lint.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings (pending merges) as errors",
    )
    _add_location_options(lint, argparse.SUPPRESS)
    lint.set_defaults(handler=_cmd_lint)

    gate = sub.add_parser(
        "trace", help="forward/backward trace gate over pytest results"
    )
    gate.add_argument("--results", type=Path, required=True)
    gate.add_argument("--gate", type=Maturity.parse, default=Maturity.MEASURED)
    gate.add_argument(
        "--summary", type=Path, default=None, help="append a Markdown report here"
    )
    _add_location_options(gate, argparse.SUPPRESS)
    gate.set_defaults(handler=_cmd_trace)

    points = sub.add_parser("points", help="write the vision points index")
    points.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    points.add_argument(
        "--check", action="store_true", help="fail when the committed index is stale"
    )
    _add_location_options(points, argparse.SUPPRESS)
    points.set_defaults(handler=_cmd_points)

    report = sub.add_parser(
        "report", help="Markdown coverage report per maturity level"
    )
    report.add_argument("--results", type=Path, default=None)
    report.add_argument("--gate", type=Maturity.parse, default=Maturity.MEASURED)
    report.add_argument("--out", type=Path, default=None)
    _add_location_options(report, argparse.SUPPRESS)
    report.set_defaults(handler=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalogue = load_catalogue(args.catalogue)
        code: int = args.handler(args, catalogue)
    except (CatalogueError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
