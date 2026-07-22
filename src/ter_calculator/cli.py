"""CLI entry point for TER Calculator."""

from __future__ import annotations

import argparse
import io
import json
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

    # batch subcommand — Phase 1 portfolio analytics
    batch_parser = subparsers.add_parser(
        "batch",
        help="Analyze a folder of JSONL sessions and build aggregate artifacts",
    )
    batch_parser.add_argument("input_dir", help="Folder containing .jsonl sessions")
    batch_parser.add_argument(
        "-o",
        "--output-dir",
        default="ter-results",
        help="Artifact directory (default: ter-results)",
    )
    batch_parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: min(8, CPU count))",
    )
    batch_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only analyze .jsonl files directly inside input_dir",
    )
    batch_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze files even when a valid output already exists",
    )
    batch_parser.add_argument(
        "--ter-buckets",
        type=int,
        default=20,
        help="Number of TER distribution buckets (default: 20 = 5%% each)",
    )
    batch_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Command summary format (default: text)",
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

    # history subcommand — Phase 4 cross-session intelligence
    history_parser = subparsers.add_parser(
        "history", help="Persistent TER history, project profiles, and predictions"
    )
    history_sub = history_parser.add_subparsers(dest="history_command")

    history_record = history_sub.add_parser(
        "record", help="Analyze and record one session"
    )
    history_record.add_argument("session_path")
    history_record.add_argument("--project", default=None)
    history_record.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt used only as a hashed fingerprint",
    )
    history_record.add_argument(
        "--db", default=None, help="Override history database path"
    )

    history_list = history_sub.add_parser("list", help="List recorded sessions")
    history_list.add_argument("--project", default=None)
    history_list.add_argument("--min-ter", type=float, default=None)
    history_list.add_argument("--max-ter", type=float, default=None)
    history_list.add_argument("--limit", type=int, default=20)
    history_list.add_argument("--db", default=None)
    history_list.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )

    history_profile = history_sub.add_parser(
        "profile", help="Summarize systemic project waste"
    )
    history_profile.add_argument("--project", default=None)
    history_profile.add_argument("--db", default=None)
    history_profile.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )

    history_predict = history_sub.add_parser(
        "predict", help="Predict TER from similar historical prompts"
    )
    history_predict.add_argument("prompt")
    history_predict.add_argument("--project", required=True)
    history_predict.add_argument("--neighbors", type=int, default=5)
    history_predict.add_argument("--db", default=None)
    history_predict.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )

    history_backup = history_sub.add_parser(
        "backup", help="Create an integrity-checked SQLite backup"
    )
    history_backup.add_argument("output")
    history_backup.add_argument("--db", default=None)

    history_restore = history_sub.add_parser(
        "restore", help="Restore an integrity-checked SQLite backup"
    )
    history_restore.add_argument("backup")
    history_restore.add_argument("--db", default=None)
    history_restore.add_argument("--force", action="store_true")

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help=(
            "Build a portfolio dashboard from .ter.json results, or show "
            "cross-session TER and cost trends"
        ),
    )
    dashboard_parser.add_argument(
        "result_dir",
        nargs="?",
        help="Directory containing existing .ter.json result files",
    )
    dashboard_parser.add_argument(
        "--ter-buckets",
        type=int,
        default=20,
        help="Number of buckets in the portfolio TER distribution (default: 20)",
    )
    dashboard_parser.add_argument(
        "--output",
        default=None,
        help="Portfolio dashboard output path (default: RESULT_DIR/ter-dashboard.html)",
    )
    dashboard_parser.add_argument("--project", default=None)
    dashboard_parser.add_argument("--limit", type=int, default=30)
    dashboard_parser.add_argument("--db", default=None)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Validate production configuration and history storage"
    )
    doctor_parser.add_argument("--db", default=None)
    doctor_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Learn project-specific thresholds, budgets, and intervention policy",
    )
    optimize_parser.add_argument("--project", required=True)
    optimize_parser.add_argument("--db", default=None)
    optimize_parser.add_argument(
        "--minimum-samples",
        type=int,
        default=5,
        help="Samples required before recommendations are considered usable",
    )
    optimize_parser.add_argument(
        "--prompt",
        default=None,
        help="Optionally personalize token budgets using a private prompt fingerprint",
    )
    optimize_parser.add_argument("--neighbors", type=int, default=5)
    optimize_parser.add_argument(
        "--output",
        default=None,
        help="Atomically write the learned policy as JSON",
    )
    optimize_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )

    integrate_parser = subparsers.add_parser(
        "integrate",
        help="Export CI/CD integration artifacts and enforce TER quality gates",
    )
    integrate_parser.add_argument(
        "result_dir", help="Directory containing existing *.ter.json results"
    )
    integrate_parser.add_argument(
        "--format",
        choices=["json", "sarif", "github", "summary"],
        default="json",
        help="Integration artifact format (default: json)",
    )
    integrate_parser.add_argument(
        "--minimum-ter",
        type=float,
        default=0.0,
        help="Fail when weighted TER is below this value",
    )
    integrate_parser.add_argument(
        "--maximum-waste-ratio",
        type=float,
        default=1.0,
        help="Fail when waste ratio exceeds this value",
    )
    integrate_parser.add_argument("--output", default=None, help="Artifact output path")

    release_parser = subparsers.add_parser(
        "release-check",
        help="Build a reproducible release manifest and enforce regression gates",
    )
    release_parser.add_argument(
        "result_dir", help="Directory containing existing *.ter.json results"
    )
    release_parser.add_argument(
        "--baseline", default=None, help="Prior release manifest"
    )
    release_parser.add_argument("--minimum-sessions", type=int, default=1)
    release_parser.add_argument("--minimum-ter", type=float, default=0.0)
    release_parser.add_argument("--maximum-waste-ratio", type=float, default=1.0)
    release_parser.add_argument("--maximum-ter-drop", type=float, default=1.0)
    release_parser.add_argument("--maximum-waste-increase", type=float, default=1.0)
    release_parser.add_argument("--format", choices=["json", "summary"], default="json")
    release_parser.add_argument("--output", default=None)

    memory_parser = subparsers.add_parser(
        "memory",
        help="Index and query project-scoped repository memory",
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    memory_index = memory_sub.add_parser(
        "index", help="Index repository files and Git history"
    )
    memory_index.add_argument("root", nargs="?", default=".")
    memory_index.add_argument("--output", default=None)
    memory_index.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    memory_search = memory_sub.add_parser(
        "search", help="Retrieve similar code, fixes, and defects"
    )
    memory_search.add_argument("query")
    memory_search.add_argument("--root", default=".")
    memory_search.add_argument("--index", dest="index_path", default=None)
    memory_search.add_argument("--limit", type=int, default=8)
    memory_search.add_argument("--minimum-score", type=float, default=0.10)
    memory_search.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    memory_inspect = memory_sub.add_parser(
        "inspect", help="Inspect repository memory coverage"
    )
    memory_inspect.add_argument("--root", default=".")
    memory_inspect.add_argument("--index", dest="index_path", default=None)
    memory_inspect.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    memory_trends = memory_sub.add_parser(
        "trends", help="Aggregate recurring patterns from session lessons"
    )
    memory_trends.add_argument("--root", default=".")
    memory_trends.add_argument("--lessons", default=None)
    memory_trends.add_argument("--outcomes", default=None)
    memory_trends.add_argument("--minimum-occurrences", type=int, default=2)
    memory_trends.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    memory_tune = memory_sub.add_parser(
        "tune", help="Preview or apply transparent per-repository threshold tuning"
    )
    memory_tune.add_argument("--root", default=".")
    memory_tune.add_argument("--minimum-samples", type=int, default=8)
    memory_tune.add_argument("--apply", action="store_true")
    memory_tune.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
    )
    memory_dashboard = memory_sub.add_parser(
        "dashboard", help="Write a static intervention effectiveness dashboard"
    )
    memory_dashboard.add_argument("--root", default=".")
    memory_dashboard.add_argument("--output", default=None)
    memory_dashboard.add_argument(
        "--format", dest="output_format", choices=["text", "json"], default="text"
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
        "--min-denied-calls",
        type=int,
        default=2,
        help="Denied calls of the same tool before circuit breaker (default: 2)",
    )
    hook_monitor.add_argument(
        "--min-reasoning-loops",
        type=int,
        default=2,
        help="Consecutive repetitive assistant messages before guidance (default: 2)",
    )
    hook_monitor.add_argument(
        "--reasoning-similarity-threshold",
        type=float,
        default=0.88,
        help="Similarity threshold for reasoning loop detection (default: 0.88)",
    )
    hook_monitor.add_argument(
        "--cost-per-1k-tokens",
        type=float,
        default=0.003,
        help="Estimated USD cost per 1,000 tokens used for effectiveness economics",
    )
    hook_monitor.add_argument(
        "--no-bash-antipatterns",
        action="store_true",
        help="Disable bash anti-pattern checking",
    )
    hook_monitor.add_argument(
        "--no-live-efficiency",
        action="store_true",
        help="Disable rolling live-efficiency degradation detection",
    )
    hook_monitor.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Number of recent events used by live efficiency (default: 10)",
    )
    hook_monitor.add_argument(
        "--efficiency-threshold",
        type=float,
        default=0.72,
        help="Rolling efficiency threshold for intervention (default: 0.72)",
    )
    hook_monitor.add_argument(
        "--drift-threshold",
        type=float,
        default=0.12,
        help="Degrading-window drift threshold (default: 0.12)",
    )
    hook_monitor.add_argument(
        "--acceleration-threshold",
        type=float,
        default=0.10,
        help="Waste acceleration threshold (default: 0.10)",
    )
    hook_monitor.add_argument(
        "--intervention-cooldown",
        type=int,
        default=8,
        help="Minimum events between refresh interventions (default: 8)",
    )
    hook_monitor.add_argument(
        "--min-repeated-failures",
        type=int,
        default=2,
        help="Identical failed actions before mandatory replan (default: 2)",
    )
    hook_monitor.add_argument(
        "--no-project-memory",
        action="store_true",
        help="Disable repository-memory retrieval for prompts",
    )
    hook_monitor.add_argument(
        "--memory-index", default=None, help="Explicit repository memory index"
    )
    hook_monitor.add_argument("--memory-limit", type=int, default=4)
    hook_monitor.add_argument("--memory-minimum-score", type=float, default=0.18)
    hook_monitor.add_argument(
        "--lesson-store", default=None, help="Session lesson JSONL path"
    )
    hook_monitor.add_argument(
        "--outcome-store", default=None, help="Intervention outcome JSONL path"
    )
    hook_monitor.add_argument(
        "--policy-mode",
        choices=["observe", "suggest", "warn", "block"],
        default="suggest",
    )
    hook_monitor.add_argument("--ter-drop-warning", type=float, default=None)
    hook_monitor.add_argument("--ter-drop-replan", type=float, default=None)
    hook_monitor.add_argument("--waste-ratio-warning", type=float, default=None)
    hook_monitor.add_argument("--waste-ratio-replan", type=float, default=None)
    hook_monitor.add_argument("--degraded-windows-required", type=int, default=None)
    hook_monitor.add_argument("--refresh-cooldown-seconds", type=int, default=None)
    hook_monitor.add_argument("--replan-cooldown-seconds", type=int, default=None)
    hook_monitor.add_argument(
        "--pre-send-check-enabled",
        action="store_true",
        help="Opt in to synchronous duplicate/pattern checks before UserPromptSubmit",
    )
    hook_monitor.add_argument(
        "--pre-send-similarity-threshold",
        type=float,
        default=0.72,
        help="Minimum repository-memory similarity for a pre-send match",
    )
    hook_monitor.add_argument(
        "--pre-send-cooldown-seconds",
        type=int,
        default=120,
        help="Cooldown between equivalent pre-send warnings",
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
        if args.command == "context":
            return _cmd_context(args)
        if args.command == "hook":
            return _cmd_hook(args)
        if args.command == "history":
            return _cmd_history(args)
        if args.command == "dashboard":
            return _cmd_dashboard(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "optimize":
            return _cmd_optimize(args)
        if args.command == "integrate":
            return _cmd_integrate(args)
        if args.command == "release-check":
            return _cmd_release(args)
        if args.command == "memory":
            return _cmd_memory(args)
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


def _cmd_hook(args) -> int:
    from .commands.hook import _cmd_hook as implementation

    return implementation(args)


def _cmd_hook_monitor(args) -> int:
    from .commands.hook import _cmd_hook_monitor as implementation

    return implementation(args)


def _cmd_optimize(args) -> int:
    from .commands.optimize import _cmd_optimize as implementation

    return implementation(args)


def _cmd_integrate(args) -> int:
    from .commands.integrate import _cmd_integrate as implementation

    return implementation(args)


def _cmd_release(args) -> int:
    from .commands.release import _cmd_release as implementation

    return implementation(args)


def _cmd_memory(args) -> int:
    from .commands.memory import _cmd_memory as implementation

    return implementation(args)


if __name__ == "__main__":
    sys.exit(main())


def _cmd_history(args) -> int:
    from .commands.history import _cmd_history as implementation

    return implementation(args)


def _cmd_dashboard(args) -> int:
    if args.result_dir is not None:
        from pathlib import Path

        from .batch_analysis import (
            aggregate_results,
            build_dashboard_html,
            load_results,
            write_combined_jsonl,
        )

        result_dir = Path(args.result_dir)
        if not result_dir.is_dir():
            raise ValueError(f"Dashboard result directory does not exist: {result_dir}")
        if args.ter_buckets <= 0:
            raise ValueError("--ter-buckets must be greater than zero")

        results, invalid = load_results(result_dir)
        if not results:
            raise ValueError(
                f"No valid .ter.json result files found under {result_dir}"
            )

        summary = aggregate_results(results)
        (result_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        write_combined_jsonl(results, result_dir / "all-results.jsonl")
        output_path = (
            Path(args.output)
            if args.output is not None
            else result_dir / "ter-dashboard.html"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            build_dashboard_html(
                results,
                summary,
                bucket_count=args.ter_buckets,
            ),
            encoding="utf-8",
        )

        if not args.quiet:
            print(f"Dashboard written to {output_path}")
            if invalid:
                print(
                    f"Skipped {len(invalid)} invalid result file(s).", file=sys.stderr
                )
        return 0

    from .commands.history import _cmd_dashboard as implementation

    return implementation(args)


def _cmd_doctor(args) -> int:
    from .commands.production import _cmd_doctor as implementation

    return implementation(args)
