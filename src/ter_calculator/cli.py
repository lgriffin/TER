"""CLI entry point for TER Calculator."""

from __future__ import annotations

import argparse
import io
import sys

from . import __version__


def _setup_stdout_encoding():
    """Ensure stdout can handle Unicode on Windows."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )


def _add_analysis_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared analysis arguments used by both analyze and report."""
    parser.add_argument(
        "session_path",
        nargs="?",
        default=None,
        help="Path to a JSONL session file (optional if --latest is used)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent session (based on file modification time)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.40,
        help="Cosine similarity threshold for alignment (default: 0.40)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Classifier confidence threshold (default: 0.75)",
    )
    parser.add_argument(
        "--restatement-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for context restatement (default: 0.85)",
    )
    parser.add_argument(
        "--phase-weights",
        type=str,
        default="0.3,0.4,0.3",
        help="Phase weights as r,t,g (default: 0.3,0.4,0.3)",
    )
    parser.add_argument(
        "--no-waste-patterns",
        action="store_true",
        help="Disable waste pattern detection",
    )
    parser.add_argument(
        "--cost-model",
        type=str,
        default="sonnet",
        help="Cost model: 'sonnet' (default) or custom 'input,output,cache_read,cache_write' rates per MTok",
    )
    parser.add_argument(
        "--no-input-analysis",
        action="store_true",
        help="Disable input analysis (user/model token breakdown, drift, and alignment)",
    )
    parser.add_argument(
        "--prompt-similarity-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for flagging redundant prompts (default: 0.75)",
    )
    parser.add_argument(
        "--cost-weighted",
        action="store_true",
        help="Include cost-weighted TER analysis",
    )
    parser.add_argument(
        "--check-overthinking",
        action="store_true",
        help="Analyze reasoning efficiency and detect overthinking",
    )
    parser.add_argument(
        "--fine-segmentation",
        action="store_true",
        help="Split long reasoning and response blocks into provenance-preserving semantic units",
    )
    parser.add_argument(
        "--segment-min-tokens",
        type=int,
        default=12,
        help="Minimum fine-segment size before adjacent units are merged (default: 12)",
    )
    parser.add_argument(
        "--segment-max-tokens",
        type=int,
        default=180,
        help="Maximum target fine-segment size (default: 180)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ter",
        description="Token Efficiency Ratio calculator for Claude Code sessions",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-essential output"
    )

    subparsers = parser.add_subparsers(dest="command")

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a Claude Code session"
    )
    _add_analysis_args(analyze_parser)
    analyze_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        dest="analysis_output",
        metavar="FILE",
        default=None,
        help="Write analysis output to FILE; recommended for --format html",
    )
    analyze_parser.add_argument(
        "--group",
        action="store_true",
        help="Include subagent sessions in grouped analysis",
    )

    # report — Markdown summary (same analysis pipeline as analyze)
    report_parser = subparsers.add_parser(
        "report",
        help="Print a Markdown summary (headline metrics, calibration, top waste, next steps)",
    )
    _add_analysis_args(report_parser)
    report_parser.add_argument(
        "-o",
        "--output",
        dest="report_output",
        metavar="FILE",
        default=None,
        help="Write Markdown to FILE instead of stdout (e.g. report.md)",
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="Compare TER across multiple sessions"
    )
    compare_parser.add_argument(
        "session_paths", nargs="+", help="Paths to JSONL session files"
    )
    compare_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    compare_parser.add_argument(
        "--sort",
        choices=["ter", "tokens", "waste"],
        default="ter",
        help="Sort order (default: ter)",
    )
    compare_parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compare exactly two sessions as before/after (Markdown delta; uses default analyze thresholds)",
    )

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List available sessions")
    list_parser.add_argument(
        "project_path",
        nargs="?",
        default=None,
        help="Path to Claude Code project directory",
    )
    list_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    list_parser.add_argument(
        "--limit", type=int, default=20, help="Maximum sessions to list (default: 20)"
    )

    # watch subcommand
    watch_parser = subparsers.add_parser(
        "watch", help="Monitor active sessions in real-time"
    )
    watch_parser.add_argument(
        "project_path",
        nargs="?",
        default=None,
        help="Path to Claude Code project directory (optional if --latest is used)",
    )
    watch_parser.add_argument(
        "--latest",
        action="store_true",
        help="Watch the most recent session (based on file modification time)",
    )
    watch_parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between polls (default: 2.0)",
    )
    watch_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    watch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to custom sentence-transformers model (optional)",
    )
    watch_parser.add_argument(
        "--log",
        dest="log_file",
        metavar="FILE",
        default=None,
        help="Append signals as JSONL to FILE for later analysis",
    )
    watch_parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming line-by-line output instead of live dashboard",
    )

    # benchmark subcommand
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate labeled classifier predictions and calibrate thresholds",
    )
    benchmark_parser.add_argument("benchmark_path", help="Path to benchmark JSONL")
    benchmark_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override predicted labels using score >= threshold as waste",
    )
    benchmark_parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for confidence intervals (default: 1000)",
    )
    benchmark_parser.add_argument("--seed", type=int, default=17)
    benchmark_parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="F-beta objective used for threshold calibration (default: 0.5)",
    )
    benchmark_parser.add_argument(
        "--minimum-precision",
        type=float,
        default=0.0,
        help="Minimum precision constraint for threshold recommendation",
    )
    benchmark_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    benchmark_parser.add_argument("-o", "--output", default=None)

    regression_parser = subparsers.add_parser(
        "benchmark-compare",
        help="Compare baseline and candidate benchmark predictions with release gates",
    )
    regression_parser.add_argument("baseline_path", help="Baseline benchmark JSONL")
    regression_parser.add_argument("candidate_path", help="Candidate benchmark JSONL")
    regression_parser.add_argument("--baseline-threshold", type=float, default=None)
    regression_parser.add_argument("--candidate-threshold", type=float, default=None)
    regression_parser.add_argument("--minimum-precision", type=float, default=0.0)
    regression_parser.add_argument("--maximum-precision-drop", type=float, default=0.0)
    regression_parser.add_argument("--maximum-recall-drop", type=float, default=1.0)
    regression_parser.add_argument("--maximum-f0-5-drop", type=float, default=0.0)
    regression_parser.add_argument("--maximum-accuracy-drop", type=float, default=1.0)
    regression_parser.add_argument(
        "--maximum-false-positive-increase", type=int, default=0
    )
    regression_parser.add_argument("--seed", type=int, default=17)
    regression_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    regression_parser.add_argument("-o", "--output", default=None)

    # budget subcommand
    budget_parser = subparsers.add_parser(
        "budget", help="Get token budget recommendations for a task"
    )
    budget_parser.add_argument(
        "intent_text", help="Task description for budget estimation"
    )
    budget_parser.add_argument(
        "--use-history",
        action="store_true",
        help="Enable historical learning from past sessions",
    )
    budget_parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help="Custom path to budget_history.json",
    )
    budget_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # context subcommand — Token Aware Microprompt Orchestrator
    context_parser = subparsers.add_parser(
        "context",
        help="Context fragment orchestration (store, graph, optimize, delta, check)",
    )
    context_sub = context_parser.add_subparsers(dest="context_command")

    ctx_store = context_sub.add_parser(
        "store", help="Shard a session into content-addressable fragments"
    )
    ctx_store.add_argument(
        "session_path", nargs="?", default=None, help="Path to a JSONL session file"
    )
    ctx_store.add_argument("--latest", action="store_true")

    ctx_graph = context_sub.add_parser(
        "graph", help="Build and display the context graph for a session"
    )
    ctx_graph.add_argument(
        "session_path", nargs="?", default=None, help="Path to a JSONL session file"
    )
    ctx_graph.add_argument("--latest", action="store_true")
    ctx_graph.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
    )

    ctx_opt = context_sub.add_parser(
        "optimize", help="Run knapsack optimization on session fragments"
    )
    ctx_opt.add_argument(
        "session_path", nargs="?", default=None, help="Path to a JSONL session file"
    )
    ctx_opt.add_argument("--latest", action="store_true")
    ctx_opt.add_argument(
        "--budget", type=int, required=True, help="Token budget ceiling"
    )
    ctx_opt.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.1,
        help="Minimum relevance score (default: 0.1)",
    )

    ctx_delta = context_sub.add_parser(
        "delta", help="Show delta prompt composition for a session"
    )
    ctx_delta.add_argument(
        "session_path", nargs="?", default=None, help="Path to a JSONL session file"
    )
    ctx_delta.add_argument("--latest", action="store_true")

    ctx_check = context_sub.add_parser(
        "check", help="Run consistency check across sessions"
    )
    ctx_check.add_argument(
        "session_path", nargs="?", default=None, help="Path to a JSONL session file"
    )
    ctx_check.add_argument("--latest", action="store_true")
    ctx_check.add_argument(
        "--group", action="store_true", help="Include subagent sessions"
    )
    ctx_check.add_argument(
        "--mode",
        choices=["strict", "relaxed"],
        default="relaxed",
        help="Consistency mode (default: relaxed)",
    )

    # visualize subcommand — SVG chart generation
    visualize_parser = subparsers.add_parser(
        "visualize", help="Generate SVG chart visualizations from a session analysis"
    )
    _add_analysis_args(visualize_parser)
    visualize_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        default=None,
        help="Directory to write SVG files (default: <session>_charts/)",
    )
    visualize_parser.add_argument(
        "--charts",
        type=str,
        default=None,
        help="Comma-separated chart names to generate (default: all). "
        "Available: key_metrics,waste_breakdown,composition,phase_scores,"
        "waste_patterns,positional_ter,economics",
    )

    # present subcommand — Marp slide deck
    present_parser = subparsers.add_parser(
        "present", help="Generate a Marp presentation summarizing session analysis"
    )
    _add_analysis_args(present_parser)
    present_parser.add_argument(
        "-o",
        "--output",
        dest="present_output",
        metavar="FILE",
        default=None,
        help="Output Marp Markdown file (default: <session>.ter-slides.md)",
    )

    # hook subcommand — Claude Code hook utilities
    hook_parser = subparsers.add_parser(
        "hook",
        help="Claude Code hook utilities",
    )
    hook_sub = hook_parser.add_subparsers(dest="hook_command")

    hook_monitor = hook_sub.add_parser(
        "monitor",
        help="PostToolUse hook: reads event from stdin, outputs guidance JSON",
    )
    hook_monitor.add_argument(
        "--min-repetitive-reads",
        type=int,
        default=3,
        help="File read count to trigger alert (default: 3)",
    )
    hook_monitor.add_argument(
        "--min-edit-fragments",
        type=int,
        default=3,
        help="Consecutive same-file edits to trigger alert (default: 3)",
    )
    hook_monitor.add_argument(
        "--min-repeated-commands",
        type=int,
        default=3,
        help="Repeated bash command count to trigger alert (default: 3)",
    )
    hook_monitor.add_argument(
        "--min-duplicate-calls",
        type=int,
        default=2,
        help="Duplicate tool call count to trigger alert (default: 2)",
    )
    hook_monitor.add_argument(
        "--no-bash-antipatterns",
        action="store_true",
        help="Disable bash anti-pattern checking",
    )
    hook_monitor.add_argument(
        "--state-dir",
        type=str,
        default=None,
        help="Override state file directory (default: system temp)",
    )

    args = parser.parse_args(argv)

    _setup_stdout_encoding()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "watch":
            return _cmd_watch(args)
        if args.command == "budget":
            return _cmd_budget(args)
        if args.command == "benchmark":
            return _cmd_benchmark(args)
        if args.command == "benchmark-compare":
            return _cmd_benchmark_compare(args)
        if args.command == "visualize":
            return _cmd_visualize(args)
        if args.command == "present":
            return _cmd_present(args)
        if args.command == "context":
            return _cmd_context(args)
        if args.command == "hook":
            return _cmd_hook(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return 1

    return 0


# Thin compatibility wrappers preserve the historical private imports used by
# downstream integrations and the project's existing test suite. Command
# behavior lives in ``ter_calculator.commands``.


def _cmd_analyze(args) -> int:
    from .commands import analyze as command

    original = command._cmd_analyze_group
    command._cmd_analyze_group = globals()["_cmd_analyze_group"]
    try:
        return command._cmd_analyze(args)
    finally:
        command._cmd_analyze_group = original


def _cmd_analyze_group(args) -> int:
    from .commands import analyze as command

    original = command._cmd_analyze
    command._cmd_analyze = globals()["_cmd_analyze"]
    try:
        return command._cmd_analyze_group(args)
    finally:
        command._cmd_analyze = original


def _cmd_compare(args) -> int:
    from .commands.analyze import _cmd_compare as implementation

    return implementation(args)


def _cmd_report(args) -> int:
    from .commands.report import _cmd_report as implementation

    return implementation(args)


def _cmd_list(args) -> int:
    from .commands.listing import _cmd_list as implementation

    return implementation(args)


def _cmd_budget(args) -> int:
    from .commands.budget import _cmd_budget as implementation

    return implementation(args)


def _cmd_benchmark(args) -> int:
    from .commands.benchmark import _cmd_benchmark as implementation

    return implementation(args)


def _cmd_benchmark_compare(args) -> int:
    from .commands.benchmark_compare import _cmd_benchmark_compare as implementation

    return implementation(args)


_last_was_live = False


def _signal_to_dict(signal) -> dict:
    from .commands.watch import _signal_to_dict as implementation

    return implementation(signal)


def _print_signal(signal, fmt, log_fh=None):
    global _last_was_live
    from .commands import watch as command

    command._last_was_live = _last_was_live
    try:
        return command._print_signal(signal, fmt, log_fh)
    finally:
        _last_was_live = command._last_was_live


def _cmd_watch(args) -> int:
    from .commands import watch as command

    original_print = command._print_signal
    command._print_signal = globals()["_print_signal"]
    try:
        return command._cmd_watch(args)
    finally:
        command._print_signal = original_print


def _resolve_session_path(args) -> str | None:
    from .commands.context import _resolve_session_path as implementation

    return implementation(args)


def _cmd_context(args) -> int:
    from .commands import context as command

    originals = {}
    for name in (
        "_cmd_context_store",
        "_cmd_context_graph",
        "_cmd_context_optimize",
        "_cmd_context_delta",
        "_cmd_context_check",
    ):
        originals[name] = getattr(command, name)
        setattr(command, name, globals()[name])
    try:
        return command._cmd_context(args)
    finally:
        for name, value in originals.items():
            setattr(command, name, value)


def _cmd_context_store(args) -> int:
    from .commands.context import _cmd_context_store as implementation

    return implementation(args)


def _cmd_context_graph(args) -> int:
    from .commands.context import _cmd_context_graph as implementation

    return implementation(args)


def _cmd_context_optimize(args) -> int:
    from .commands.context import _cmd_context_optimize as implementation

    return implementation(args)


def _cmd_context_delta(args) -> int:
    from .commands.context import _cmd_context_delta as implementation

    return implementation(args)


def _cmd_context_check(args) -> int:
    from .commands.context import _cmd_context_check as implementation

    return implementation(args)


def _cmd_visualize(args) -> int:
    from .commands.visualize import _cmd_visualize as implementation

    return implementation(args)


def _cmd_present(args) -> int:
    from .commands.present import _cmd_present as implementation

    return implementation(args)


def _cmd_hook(args) -> int:
    from .commands.hook import _cmd_hook as implementation

    return implementation(args)


def _cmd_hook_monitor(args) -> int:
    from .commands.hook import _cmd_hook_monitor as implementation

    return implementation(args)


if __name__ == "__main__":
    sys.exit(main())
