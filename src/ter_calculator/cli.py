"""CLI entry point for TER Calculator."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from . import __version__


def _setup_stdout_encoding():
    """Ensure stdout can handle Unicode on Windows."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ter",
        description="Token Efficiency Ratio calculator for Claude Code sessions",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-essential output"
    )

    subparsers = parser.add_subparsers(dest="command")

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a Claude Code session"
    )
    analyze_parser.add_argument(
        "session_path", nargs="?", default=None,
        help="Path to a JSONL session file (optional if --latest is used)"
    )
    analyze_parser.add_argument(
        "--latest", action="store_true",
        help="Analyze the most recent session (based on file modification time)"
    )
    analyze_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"],
        default="text", help="Output format (default: text)"
    )
    analyze_parser.add_argument(
        "--similarity-threshold", type=float, default=0.40,
        help="Cosine similarity threshold for alignment (default: 0.40)"
    )
    analyze_parser.add_argument(
        "--confidence-threshold", type=float, default=0.75,
        help="Classifier confidence threshold (default: 0.75)"
    )
    analyze_parser.add_argument(
        "--restatement-threshold", type=float, default=0.85,
        help="Similarity threshold for context restatement (default: 0.85)"
    )
    analyze_parser.add_argument(
        "--phase-weights", type=str, default="0.3,0.4,0.3",
        help="Phase weights as r,t,g (default: 0.3,0.4,0.3)"
    )
    analyze_parser.add_argument(
        "--no-waste-patterns", action="store_true",
        help="Disable waste pattern detection"
    )
    analyze_parser.add_argument(
        "--cost-model", type=str, default="sonnet",
        help="Cost model: 'sonnet' (default) or custom 'input,output,cache_read,cache_write' rates per MTok"
    )
    analyze_parser.add_argument(
        "--no-input-analysis", action="store_true",
        help="Disable input analysis (user/model token breakdown, drift, and alignment)"
    )
    analyze_parser.add_argument(
        "--prompt-similarity-threshold", type=float, default=0.75,
        help="Cosine similarity threshold for flagging redundant prompts (default: 0.75)"
    )
    analyze_parser.add_argument(
        "--group", action="store_true",
        help="Include subagent sessions in grouped analysis"
    )
    analyze_parser.add_argument(
        "--cost-weighted", action="store_true",
        help="Include cost-weighted TER analysis"
    )
    analyze_parser.add_argument(
        "--check-overthinking", action="store_true",
        help="Analyze reasoning efficiency and detect overthinking"
    )

    # report — Markdown summary (same analysis pipeline as analyze)
    report_parser = subparsers.add_parser(
        "report",
        help="Print a Markdown summary (headline metrics, calibration, top waste, next steps)",
    )
    report_parser.add_argument(
        "session_path", nargs="?", default=None,
        help="Path to a JSONL session file (optional if --latest is used)"
    )
    report_parser.add_argument(
        "--latest", action="store_true",
        help="Report on the most recent session (based on file modification time)"
    )
    report_parser.add_argument(
        "--similarity-threshold", type=float, default=0.40,
        help="Cosine similarity threshold for alignment (default: 0.40)"
    )
    report_parser.add_argument(
        "--confidence-threshold", type=float, default=0.75,
        help="Classifier confidence threshold (default: 0.75)"
    )
    report_parser.add_argument(
        "--restatement-threshold", type=float, default=0.85,
        help="Similarity threshold for context restatement (default: 0.85)"
    )
    report_parser.add_argument(
        "--phase-weights", type=str, default="0.3,0.4,0.3",
        help="Phase weights as r,t,g (default: 0.3,0.4,0.3)"
    )
    report_parser.add_argument(
        "--no-waste-patterns", action="store_true",
        help="Disable waste pattern detection"
    )
    report_parser.add_argument(
        "--cost-model", type=str, default="sonnet",
        help="Cost model: 'sonnet' (default) or custom rates per MTok"
    )
    report_parser.add_argument(
        "--no-input-analysis", action="store_true",
        help="Disable input analysis"
    )
    report_parser.add_argument(
        "--prompt-similarity-threshold", type=float, default=0.75,
        help="Cosine similarity threshold for redundant prompts (default: 0.75)"
    )
    report_parser.add_argument(
        "-o",
        "--output",
        dest="report_output",
        metavar="FILE",
        default=None,
        help="Write Markdown to FILE instead of stdout (e.g. report.md)",
    )
    report_parser.add_argument(
        "--cost-weighted", action="store_true",
        help="Include cost-weighted TER analysis"
    )
    report_parser.add_argument(
        "--check-overthinking", action="store_true",
        help="Analyze reasoning efficiency and detect overthinking"
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="Compare TER across multiple sessions"
    )
    compare_parser.add_argument(
        "session_paths", nargs="+", help="Paths to JSONL session files"
    )
    compare_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"],
        default="text", help="Output format (default: text)"
    )
    compare_parser.add_argument(
        "--sort", choices=["ter", "tokens", "waste"],
        default="ter", help="Sort order (default: ter)"
    )
    compare_parser.add_argument(
        "--baseline", action="store_true",
        help="Compare exactly two sessions as before/after (Markdown delta; uses default analyze thresholds)",
    )

    # list subcommand
    list_parser = subparsers.add_parser(
        "list", help="List available sessions"
    )
    list_parser.add_argument(
        "project_path", nargs="?", default=None,
        help="Path to Claude Code project directory"
    )
    list_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"],
        default="text", help="Output format (default: text)"
    )
    list_parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum sessions to list (default: 20)"
    )

    # watch subcommand
    watch_parser = subparsers.add_parser(
        "watch", help="Monitor active sessions in real-time"
    )
    watch_parser.add_argument(
        "project_path", nargs="?", default=None,
        help="Path to Claude Code project directory (optional if --latest is used)"
    )
    watch_parser.add_argument(
        "--latest", action="store_true",
        help="Watch the most recent session (based on file modification time)"
    )
    watch_parser.add_argument(
        "--poll-interval", type=float, default=2.0,
        help="Seconds between polls (default: 2.0)"
    )
    watch_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"],
        default="text", help="Output format (default: text)"
    )
    watch_parser.add_argument(
        "--model", type=str, default=None,
        help="Path to custom sentence-transformers model (optional)"
    )
    watch_parser.add_argument(
        "--log", dest="log_file", metavar="FILE", default=None,
        help="Append signals as JSONL to FILE for later analysis"
    )

    # budget subcommand
    budget_parser = subparsers.add_parser(
        "budget", help="Get token budget recommendations for a task"
    )
    budget_parser.add_argument(
        "intent_text", help="Task description for budget estimation"
    )
    budget_parser.add_argument(
        "--use-history", action="store_true",
        help="Enable historical learning from past sessions"
    )
    budget_parser.add_argument(
        "--history-path", type=str, default=None,
        help="Custom path to budget_history.json"
    )
    budget_parser.add_argument(
        "--format", dest="output_format", choices=["text", "json"],
        default="text", help="Output format (default: text)"
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


def _cmd_analyze(args) -> int:
    """Execute the analyze subcommand."""
    # Resolve --latest flag
    if args.latest:
        from .loader import find_latest_session
        args.session_path = str(find_latest_session(args.session_path))
        if not args.quiet:
            print(f"Using latest session: {args.session_path}", file=sys.stderr)
    elif args.session_path is None:
        print("Error: Either provide a session_path or use --latest", file=sys.stderr)
        return 1

    if args.group:
        return _cmd_analyze_group(args)

    from .analyze_pipeline import analyze_session
    from .formatter import format_ter_result

    result = analyze_session(args)
    print(format_ter_result(result, fmt=args.output_format))
    return 0


def _cmd_report(args) -> int:
    """Markdown one-screen summary for humans."""
    # Resolve --latest flag
    if args.latest:
        from .loader import find_latest_session
        args.session_path = str(find_latest_session(args.session_path))
        if not args.quiet:
            print(f"Using latest session: {args.session_path}", file=sys.stderr)
    elif args.session_path is None:
        print("Error: Either provide a session_path or use --latest", file=sys.stderr)
        return 1

    from .analyze_pipeline import analyze_session
    from .session_report import format_session_report_markdown

    result = analyze_session(args)
    md = format_session_report_markdown(result)
    out = getattr(args, "report_output", None)
    if out:
        Path(out).write_text(md, encoding="utf-8")
        if not args.quiet:
            print(f"Wrote {out}", file=sys.stderr)
    else:
        print(md)
    return 0


def _cmd_analyze_group(args) -> int:
    """Execute grouped analysis: parent + subagent sessions."""
    from .loader import load_session, segment_spans, discover_subagents
    from .intent import extract_intent
    from .classifier import classify_spans
    from .compute import compute_ter
    from .waste import detect_waste_patterns
    from .economics import compute_economics
    from .formatter import format_grouped_analysis

    subagent_paths = discover_subagents(args.session_path)
    if not subagent_paths:
        print("No subagent sessions found, running single-session analysis.",
              file=sys.stderr)
        # Fall back to normal analyze (without --group).
        args.group = False
        return _cmd_analyze(args)

    from .config_parse import parse_cost_model, parse_phase_weights

    phase_weights = parse_phase_weights(args.phase_weights)
    cost_model = parse_cost_model(args.cost_model)

    def _analyze_session(path):
        session = load_session(path)
        spans = segment_spans(session)
        intent = extract_intent(session)
        classified = classify_spans(
            spans, intent,
            similarity_threshold=args.similarity_threshold,
            confidence_threshold=args.confidence_threshold,
        )
        result = compute_ter(
            classified, session_id=session.session_id,
            intent=intent, phase_weights=phase_weights,
        )
        if not args.no_waste_patterns:
            result.waste_patterns = detect_waste_patterns(
                classified,
                restatement_threshold=args.restatement_threshold,
                session=session,
            )
        result.economics = compute_economics(session, classified, cost_model)
        return result

    if not args.quiet:
        print(f"Analyzing parent + {len(subagent_paths)} subagent(s)...",
              file=sys.stderr)

    parent_result = _analyze_session(args.session_path)
    subagent_results = []
    for p in subagent_paths:
        r = _analyze_session(str(p))
        # Use filename as session_id since subagents share the parent's sessionId.
        r.session_id = p.stem
        subagent_results.append(r)

    print(format_grouped_analysis(
        parent_result, subagent_results, fmt=args.output_format,
    ))
    return 0


def _cmd_compare(args) -> int:
    """Execute the compare subcommand."""
    from pathlib import Path

    from .formatter import format_comparison

    # Expand directory paths to all .jsonl files inside them.
    paths = []
    for p in args.session_paths:
        pp = Path(p)
        if pp.is_dir():
            paths.extend(sorted(str(f) for f in pp.glob("*.jsonl")))
        else:
            paths.append(p)

    if not paths:
        print("No .jsonl files found.", file=sys.stderr)
        return 1

    if getattr(args, "baseline", False):
        if len(paths) != 2:
            print(
                "Error: --baseline requires exactly two session files.",
                file=sys.stderr,
            )
            return 1
        for p in paths:
            if Path(p).is_dir():
                print(
                    "Error: --baseline requires file paths, not directories.",
                    file=sys.stderr,
                )
                return 1
        from .analyze_pipeline import analyze_session, default_analyze_args
        from .session_report import format_baseline_markdown

        ra = analyze_session(default_analyze_args(paths[0]))
        rb = analyze_session(default_analyze_args(paths[1]))
        print(format_baseline_markdown(ra, rb))
        return 0

    from .loader import load_session, segment_spans
    from .intent import extract_intent
    from .classifier import classify_spans
    from .compute import compute_ter
    from .economics import compute_economics
    from .waste import detect_waste_patterns

    results = []
    for path in paths:
        session = load_session(path)
        spans = segment_spans(session)
        intent = extract_intent(session)
        classified = classify_spans(spans, intent)
        result = compute_ter(classified, session_id=session.session_id, intent=intent)
        result.waste_patterns = detect_waste_patterns(classified, session=session)
        result.economics = compute_economics(session, classified)
        results.append(result)

    # Sort results.
    sort_key = {
        "ter": lambda r: r.aggregate_ter,
        "tokens": lambda r: r.total_tokens,
        "waste": lambda r: r.waste_tokens,
    }
    results.sort(key=sort_key[args.sort], reverse=(args.sort == "ter"))

    print(format_comparison(results, fmt=args.output_format))
    return 0


def _cmd_list(args) -> int:
    """Execute the list subcommand."""
    import json as json_mod
    from pathlib import Path
    from .loader import discover_subagents

    project_path = args.project_path
    if project_path is None:
        home = Path.home()
        claude_dir = home / ".claude" / "projects"
        if not claude_dir.exists():
            print("Error: No Claude Code projects found at ~/.claude/projects/",
                  file=sys.stderr)
            return 1
        project_path = str(claude_dir)

    project_dir = Path(project_path)
    if not project_dir.exists():
        print(f"Error: Directory not found: {project_path}", file=sys.stderr)
        return 1

    sessions = []
    for jsonl_file in sorted(project_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(sessions) >= args.limit:
            break
        # Skip subagent files — they're shown as counts on their parent.
        if "subagents" in jsonl_file.parts:
            continue
        subagent_count = len(discover_subagents(jsonl_file))
        sessions.append({
            "path": str(jsonl_file),
            "name": jsonl_file.stem,
            "size": jsonl_file.stat().st_size,
            "modified": jsonl_file.stat().st_mtime,
            "subagent_count": subagent_count,
        })

    if args.output_format == "json":
        print(json_mod.dumps(sessions, indent=2))
    else:
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Found {len(sessions)} session(s):\n")
            for i, s in enumerate(sessions, 1):
                size_kb = s["size"] / 1024
                sub_str = f", {s['subagent_count']} subagents" if s["subagent_count"] else ""
                print(f"  {i}. {s['name']} ({size_kb:.1f} KB{sub_str})")
                print(f"     {s['path']}")

    return 0


def _cmd_budget(args) -> int:
    """Execute the budget subcommand for token budget recommendations."""
    import json as json_mod
    from .adaptive_budget import recommend_budget, HistoricalBudgetAnalyzer

    if not args.intent_text.strip():
        print("Error: Intent text cannot be empty", file=sys.stderr)
        return 1

    # Load historical data if requested
    history = None
    if args.use_history:
        try:
            history = HistoricalBudgetAnalyzer(
                history_path=args.history_path if args.history_path else None
            )
        except Exception as e:
            print(f"Warning: Could not load history: {e}", file=sys.stderr)

    # Get recommendation
    try:
        rec = recommend_budget(args.intent_text, history=history)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1

    # Format output
    if args.output_format == "json":
        print(json_mod.dumps({
            "complexity": rec.complexity.value,
            "model_tier": rec.model_tier.value,
            "max_thinking_tokens": rec.max_thinking_tokens,
            "estimated_total_tokens": rec.estimated_total_tokens,
            "estimated_cost_usd": rec.estimated_cost_usd,
            "confidence": rec.confidence,
            "reasoning": rec.reasoning
        }, indent=2))
    else:
        print("Budget Recommendation")
        print("═" * 50)
        print(f"Complexity: {rec.complexity.value} ({rec.confidence:.0%} confidence)")
        print(f"Model: {rec.model_tier.value}")
        print(f"Max Thinking Tokens: {rec.max_thinking_tokens:,}")
        print(f"Est. Total Tokens: {rec.estimated_total_tokens:,}")
        print(f"Est. Cost: ${rec.estimated_cost_usd:.4f}")
        print(f"\nReasoning:\n{rec.reasoning}")

    return 0


def _signal_to_dict(signal) -> dict:
    """Convert a TERSignal to a JSON-serialisable dict."""
    from datetime import datetime, timezone

    return {
        "session_id": signal.session_id,
        "timestamp": datetime.fromtimestamp(signal.timestamp, tz=timezone.utc).isoformat(),
        "is_live": signal.is_live,
        "ter": round(signal.aggregate_ter, 4),
        "raw_ratio": round(signal.raw_ratio, 4),
        "message_index": signal.message_index,
        "drift": signal.drift.value,
        "drift_magnitude": round(signal.drift_magnitude, 4),
        "warnings": signal.warnings,
        "warning_level": signal.warning_level.value,
        "tokens": {
            "total": signal.total_tokens,
            "aligned": signal.aligned_tokens,
            "waste": signal.waste_tokens,
        },
        "phase_ter": getattr(signal, "phase_ter", {}),
        "waste_sources": getattr(signal, "waste_sources", {}),
    }


_last_was_live = False


def _print_signal(signal, fmt, log_fh=None):
    """Format and print a TER signal from live monitoring."""
    global _last_was_live
    import json as json_mod

    record = _signal_to_dict(signal)

    if log_fh is not None:
        log_fh.write(json_mod.dumps(record) + "\n")
        log_fh.flush()

    if fmt == "json":
        print(json_mod.dumps(record), flush=True)
    else:
        from datetime import datetime, timezone

        if signal.is_live and not _last_was_live:
            print("\n--- LIVE ---\n", flush=True)
        _last_was_live = signal.is_live

        tag = "LIVE" if signal.is_live else "HISTORY"
        msg_time = datetime.fromtimestamp(signal.timestamp, tz=timezone.utc).astimezone().strftime("%H:%M:%S")
        drift_arrow = "↑" if signal.drift.value == "improving" else "↓" if signal.drift.value == "degrading" else "→"
        waste_pct = (signal.waste_tokens / signal.total_tokens * 100) if signal.total_tokens > 0 else 0
        aligned_pct = (signal.aligned_tokens / signal.total_tokens * 100) if signal.total_tokens > 0 else 0

        print(f"[{msg_time}] [{tag}] [{signal.session_id[:8]}] TER: {signal.aggregate_ter:.2f} (weighted) | "
              f"Raw: {signal.raw_ratio:.2f} {drift_arrow} | "
              f"Tokens: {signal.total_tokens:,} | "
              f"Aligned: {signal.aligned_tokens:,} ({aligned_pct:.0f}%) | "
              f"Waste: {signal.waste_tokens:,} ({waste_pct:.0f}%)", flush=True)

        # Show phase breakdown if available
        if hasattr(signal, 'phase_ter') and signal.phase_ter:
            phase_strs = [f"{p[:3]}: {ter:.2f}" for p, ter in signal.phase_ter.items()]
            print(f"  Phases: {' | '.join(phase_strs)}", flush=True)

        # Show waste sources if available
        if hasattr(signal, 'waste_sources') and signal.waste_sources:
            waste_items = []
            for source, count in signal.waste_sources.items():
                if count > 0:
                    waste_items.append(f"{source}: {count}")
            if waste_items:
                print(f"  Waste: {', '.join(waste_items)}", flush=True)

        if signal.warnings:
            for warning in signal.warnings:
                print(f"  ⚠ {warning}", flush=True)


def _cmd_watch(args) -> int:
    """Execute the watch subcommand for live session monitoring."""
    from .real_time import LiveDashboard, SessionMonitor
    from pathlib import Path

    single_file = None

    if args.latest:
        from .loader import find_latest_session
        latest_session = find_latest_session(args.project_path)
        single_file = latest_session
        if not args.quiet:
            print(f"Watching latest session: {latest_session.name}", file=sys.stderr)
    elif args.project_path is None:
        print("Error: Either provide a project_path or use --latest", file=sys.stderr)
        return 1
    else:
        target = Path(args.project_path)
        if target.is_file() and target.suffix == ".jsonl":
            single_file = target
        elif target.is_dir():
            session_jsonl = target.parent / (target.name + ".jsonl")
            if session_jsonl.is_file():
                single_file = session_jsonl
                if not args.quiet:
                    print(f"Watching session file: {session_jsonl.name}", file=sys.stderr)
        elif not target.exists():
            print(f"Error: Project path not found: {target}", file=sys.stderr)
            return 1

    log_fh = None
    log_path = getattr(args, "log_file", None)
    signal_count = 0

    try:
        if log_path:
            log_fh = open(log_path, "a", encoding="utf-8")

        def on_signal(sig):
            nonlocal signal_count
            signal_count += 1
            _print_signal(sig, args.output_format, log_fh=log_fh)

        if single_file is not None:
            monitor = SessionMonitor(
                single_file,
                poll_interval=args.poll_interval,
                model=None,
                on_signal=on_signal,
            )
        else:
            monitor = LiveDashboard(
                project_dir=Path(args.project_path),
                poll_interval=args.poll_interval,
                model=None,
                on_signal=on_signal,
            )
    except Exception as e:
        print(f"Error initializing monitor: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        if log_fh:
            log_fh.close()
        return 1

    watch_target = single_file or args.project_path
    try:
        if args.output_format == "text":
            print(f"Watching: {watch_target}", flush=True)
            if log_path:
                print(f"Logging to: {log_path}", flush=True)
            print("Press Ctrl+C to stop...\n", flush=True)
        monitor.run()
    except KeyboardInterrupt:
        monitor.stop()
        if args.output_format == "text":
            print("\nStopped monitoring.")
            if log_path:
                print(f"Wrote {signal_count} signals to {log_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if log_fh:
            log_fh.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
