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
        "session_path", help="Path to a JSONL session file"
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

    # report — Markdown summary (same analysis pipeline as analyze)
    report_parser = subparsers.add_parser(
        "report",
        help="Print a Markdown summary (headline metrics, calibration, top waste, next steps)",
    )
    report_parser.add_argument(
        "session_path", help="Path to a JSONL session file"
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
    if args.group:
        return _cmd_analyze_group(args)

    from .analyze_pipeline import analyze_session
    from .formatter import format_ter_result

    result = analyze_session(args)
    print(format_ter_result(result, fmt=args.output_format))
    return 0


def _cmd_report(args) -> int:
    """Markdown one-screen summary for humans."""
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


if __name__ == "__main__":
    sys.exit(main())
