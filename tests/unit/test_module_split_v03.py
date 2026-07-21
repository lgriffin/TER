"""Architecture and compatibility tests for the v03 module split."""

from ter_calculator import acceleration
from ter_calculator.cli import main


def test_acceleration_public_api_is_preserved():
    expected = {
        "AnalysisCache",
        "CacheStats",
        "QuickAnalyser",
        "SessionWatcher",
        "WatchEvent",
        "WatchEventType",
        "parallel_embed",
        "hash_file",
    }
    assert expected.issubset(set(acceleration.__all__))
    for name in expected:
        assert hasattr(acceleration, name)


def test_acceleration_components_have_focused_modules():
    assert acceleration.AnalysisCache.__module__.endswith("acceleration.cache")
    assert acceleration.QuickAnalyser.__module__.endswith("acceleration.quick_analyser")
    assert acceleration.SessionWatcher.__module__.endswith(
        "acceleration.session_watcher"
    )


def test_command_implementations_are_extracted():
    from ter_calculator.commands import (
        analyze,
        budget,
        context,
        hook,
        listing,
        report,
        watch,
    )

    assert analyze._cmd_analyze.__module__ == "ter_calculator.commands.analyze"
    assert report._cmd_report.__module__ == "ter_calculator.commands.report"
    assert watch._cmd_watch.__module__ == "ter_calculator.commands.watch"
    assert context._cmd_context.__module__ == "ter_calculator.commands.context"
    assert budget._cmd_budget.__module__ == "ter_calculator.commands.budget"
    assert listing._cmd_list.__module__ == "ter_calculator.commands.listing"
    assert hook._cmd_hook.__module__ == "ter_calculator.commands.hook"


def test_cli_entrypoint_still_dispatches_help(capsys):
    assert main([]) == 1
    assert "usage:" in capsys.readouterr().out


def test_benchmark_command_module_and_cli_wrapper():
    import ter_calculator.cli as cli
    from ter_calculator.commands import benchmark

    assert callable(benchmark._cmd_benchmark)
    assert callable(cli._cmd_benchmark)
