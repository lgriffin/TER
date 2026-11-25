"""The install itself: scripts start, report their versions, and every
subcommand they offer has a smoke journey (TER-TST-006)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import Cli

pytestmark = pytest.mark.smoke

#: Every subcommand, and the smoke test that runs it end to end. A new
#: subcommand fails test_every_subcommand_has_a_journey until it has one.
TER4_COMMANDS = {
    "a3": "test_ter4_analysis.py::test_a3_writes_html_and_json",
    "capabilities": "test_install.py::test_capabilities_lists_every_port",
    "context": "test_ter4_analysis.py::test_context_bundle_and_report",
    "control": "test_ter4_control_and_policy.py::test_control_charts_from_measures",
    "corpus": "test_ter4_analysis.py::test_corpus_import_redacts_into_a_manifest",
    "explain": "test_ter4_analysis.py::test_explain_json_names_its_format",
    "hook": "test_ter4_hooks.py::test_hook_records_a_payload_into_the_event_log",
    "hooks": "test_ter4_hooks.py::test_hooks_check_correlates_recordings",
    "observe": "test_ter4_hooks.py::test_observe_reads_the_live_event_log",
    "policy": "test_ter4_control_and_policy.py::test_policy_approve_shadow_effectiveness",
    "record": "test_ter4_analysis.py::test_record_export_then_verify",
    "route": "test_ter4_analysis.py::test_route_json_names_its_format",
}
TER3_COMMANDS = {
    "a3": "test_ter3.py::test_a3_and_explain",
    "analyze": "test_ter3.py::test_analysis_commands",
    "benchmark": "test_ter3.py::test_benchmark_scores_labelled_predictions",
    "benchmark-compare": "test_ter3.py::test_benchmark_compare_reports_the_delta",
    "budget": "test_ter3.py::test_budget_json",
    "compare": "test_ter3.py::test_analysis_commands",
    "context": "test_ter3.py::test_context_store_round_trip",
    "explain": "test_ter3.py::test_a3_and_explain",
    "hook": "test_ter3.py::test_hook_monitor_answers_claude_code",
    "list": "test_ter3.py::test_list_runs_without_sessions",
    "present": "test_ter3.py::test_analysis_commands",
    "report": "test_ter3.py::test_analysis_commands",
    "visualize": "test_ter3.py::test_analysis_commands",
    "watch": "test_ter3.py::test_watch_follows_a_session",
}


def _subcommands(usage: str) -> set[str]:
    match = re.search(r"\{([a-z0-9,-]+)\}", usage)
    assert match, usage
    return set(match.group(1).split(","))


@pytest.mark.req("TER-TST-006")
def test_versions(cli: Cli) -> None:
    assert "ter.event/0.5" in cli.ter4("--version").ok().out
    assert cli.ter("--version").ok().out.strip()
    assert cli.ter_req("--help").ok().out


@pytest.mark.req("TER-TST-006")
def test_the_installed_package_is_what_runs(cli: Cli) -> None:
    where = cli.run(
        [cli.python, "-c", "import ter, ter_calculator; print(ter.__file__)"]
    ).ok()
    if cli.installed:
        assert "site-packages" in where.out, where.out
    else:
        assert "src" in where.out, where.out


@pytest.mark.req("TER-TST-006")
def test_every_subcommand_has_a_journey(cli: Cli) -> None:
    assert _subcommands(cli.ter4("--help").ok().out) == set(TER4_COMMANDS)
    assert _subcommands(cli.ter("--help").ok().out) == set(TER3_COMMANDS)
    for journey in {*TER4_COMMANDS.values(), *TER3_COMMANDS.values()}:
        module, _, name = journey.partition("::")
        source = (Path(__file__).parent / module).read_text(encoding="utf-8")
        assert f"def {name}(" in source, f"no smoke test {journey}"


@pytest.mark.req("TER-TST-006")
@pytest.mark.parametrize("command", sorted(TER4_COMMANDS))
def test_every_ter4_subcommand_has_help(cli: Cli, command: str) -> None:
    assert "usage:" in cli.ter4(command, "--help").ok().out


@pytest.mark.req("TER-TST-006")
@pytest.mark.parametrize("command", sorted(TER3_COMMANDS))
def test_every_ter3_subcommand_has_help(cli: Cli, command: str) -> None:
    assert "usage:" in cli.ter(command, "--help").ok().out


@pytest.mark.req("TER-TST-006")
def test_capabilities_lists_every_port(cli: Cli) -> None:
    out = cli.ter4("capabilities").ok().out
    for port in ("SessionSource", "PriceBook", "InterventionLedger"):
        assert port in out


@pytest.mark.req("TER-TST-007")
def test_an_unknown_command_fails_with_usage(cli: Cli) -> None:
    result = cli.ter4("no-such-command")
    assert result.code == 2 and "usage:" in result.err
