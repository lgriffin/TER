"""CLI entry point for TER Calculator."""

from __future__ import annotations

import argparse
import io
import sys

from . import __version__
from .models import SpanPhase


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
    from .loader import load_session, segment_spans
    from .intent import extract_intent
    from .classifier import classify_spans
    from .compute import compute_ter
    from .formatter import format_ter_result

    phase_weights = _parse_phase_weights(args.phase_weights)

    session = load_session(args.session_path)
    spans = segment_spans(session)
    intent = extract_intent(session)

    classified = classify_spans(
        spans, intent,
        similarity_threshold=args.similarity_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    result = compute_ter(
        classified,
        session_id=session.session_id,
        intent=intent,
        phase_weights=phase_weights,
    )

    if not args.no_waste_patterns:
        from .waste import detect_waste_patterns
        result.waste_patterns = detect_waste_patterns(
            classified,
            restatement_threshold=args.restatement_threshold,
        )

    from .economics import compute_economics
    cost_model = _parse_cost_model(args.cost_model)
    result.economics = compute_economics(session, classified, cost_model)

    print(format_ter_result(result, fmt=args.output_format))
    return 0


def _cmd_compare(args) -> int:
    """Execute the compare subcommand."""
    from .loader import load_session, segment_spans
    from .intent import extract_intent
    from .classifier import classify_spans
    from .compute import compute_ter
    from .formatter import format_comparison

    from pathlib import Path
    from .economics import compute_economics

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

    results = []
    for path in paths:
        session = load_session(path)
        spans = segment_spans(session)
        intent = extract_intent(session)
        classified = classify_spans(spans, intent)
        result = compute_ter(classified, session_id=session.session_id, intent=intent)
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
        sessions.append({
            "path": str(jsonl_file),
            "name": jsonl_file.stem,
            "size": jsonl_file.stat().st_size,
            "modified": jsonl_file.stat().st_mtime,
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
                print(f"  {i}. {s['name']} ({size_kb:.1f} KB)")
                print(f"     {s['path']}")

    return 0


def _parse_cost_model(value: str):
    """Parse cost model argument."""
    from .models import CostModel
    if value.lower() == "sonnet":
        return CostModel()
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"Cost model must be 'sonnet' or 4 comma-separated rates, got: {value}"
        )
    try:
        return CostModel(
            input_rate=float(parts[0]),
            output_rate=float(parts[1]),
            cache_read_rate=float(parts[2]),
            cache_write_rate=float(parts[3]),
        )
    except ValueError:
        raise ValueError(f"Invalid cost model rates: {value}")


def _parse_phase_weights(weights_str: str) -> dict[SpanPhase, float]:
    """Parse comma-separated phase weights."""
    parts = weights_str.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"Phase weights must be 3 comma-separated values, got: {weights_str}"
        )
    try:
        r, t, g = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        raise ValueError(f"Invalid phase weight values: {weights_str}")

    total = r + t + g
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Phase weights must sum to 1.0, got {total}: {weights_str}"
        )

    return {
        SpanPhase.REASONING: r,
        SpanPhase.TOOL_USE: t,
        SpanPhase.GENERATION: g,
    }


if __name__ == "__main__":
    sys.exit(main())
