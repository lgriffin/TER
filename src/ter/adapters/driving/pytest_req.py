"""Pytest plugin: record which tests verify which EARS requirement.

Tests cite requirements with ``@pytest.mark.req("TER-ANL-001")`` (several ids
may be given in one marker). Run pytest with ``--req-trace=PATH`` to write a
JSON map of requirement id to the citing tests and their outcomes::

    {"schema": "ter.req-trace/1",
     "requirements": {"TER-ANL-001": [{"nodeid": "tests/...", "outcome": "passed"}]}}

BDD scenarios cite them with a ``@req:TER-ANL-001`` tag in the ``.feature``
file (pytest-bdd), which becomes the same marker, so a scenario counts in
the trace like any other test.

``python -m ter.adapters.driving.req_cli trace`` reads that file. The plugin
is registered by the repository's root ``conftest.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

TRACE_SCHEMA = "ter.req-trace/1"
NOT_RUN = "not-run"
#: The Gherkin tag that cites a requirement: ``@req:TER-ANL-001``.
REQ_TAG = "req:"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ter-req", "TER requirement traceability")
    group.addoption(
        "--req-trace",
        metavar="PATH",
        default=None,
        help="write requirement id -> citing tests and outcomes as JSON to PATH",
    )


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption("--req-trace", default=None)
    if path:
        config.pluginmanager.register(
            RequirementTracer(Path(str(path))), "ter-req-tracer"
        )


@pytest.hookimpl(optionalhook=True, tryfirst=True)
def pytest_bdd_apply_tag(
    tag: str, function: Callable[..., object]
) -> Callable[..., object] | None:
    """Turn a scenario's ``@req:<id>`` tag into ``@pytest.mark.req(<id>)``.

    Other tags are left to pytest-bdd, which makes each one a marker.
    """
    if not tag.startswith(REQ_TAG):
        return None
    rid = tag.removeprefix(REQ_TAG)
    if not rid:
        raise pytest.UsageError(f"the tag @{tag} names no requirement id")
    marked: Callable[..., object] = pytest.mark.req(rid)(function)
    return marked


def cited_ids(item: pytest.Item) -> list[str]:
    """Requirement ids cited by ``item``'s ``req`` markers.

    Raises:
        pytest.UsageError: If a marker has no id or a non-string id.
    """
    ids: list[str] = []
    for marker in item.iter_markers("req"):
        if not marker.args or not all(isinstance(a, str) for a in marker.args):
            raise pytest.UsageError(
                f"{item.nodeid}: @pytest.mark.req needs one or more string ids"
            )
        ids.extend(str(a) for a in marker.args)
    return list(dict.fromkeys(ids))


class RequirementTracer:
    """Collects ``req`` citations and test outcomes, then writes the trace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.citations: dict[str, list[str]] = {}
        self.outcomes: dict[str, str] = {}

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        for item in items:
            ids = cited_ids(item)
            if ids:
                self.citations[item.nodeid] = ids

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.nodeid not in self.citations:
            return
        outcome: str = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        if report.when == "call":
            self.outcomes[report.nodeid] = outcome
        elif report.outcome != "passed":
            # A failing or skipping setup/teardown overrides a passing call.
            self.outcomes[report.nodeid] = outcome

    def mapping(self) -> dict[str, list[dict[str, str]]]:
        """Requirement id -> [{nodeid, outcome}], sorted for stable output."""
        result: dict[str, list[dict[str, str]]] = {}
        for nodeid, ids in self.citations.items():
            for rid in ids:
                outcome = self.outcomes.get(nodeid, NOT_RUN)
                result.setdefault(rid, []).append(
                    {"nodeid": nodeid, "outcome": outcome}
                )
        return {
            rid: sorted(v, key=lambda e: e["nodeid"])
            for rid, v in sorted(result.items())
        }

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema": TRACE_SCHEMA, "requirements": self.mapping()}
        self.path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
