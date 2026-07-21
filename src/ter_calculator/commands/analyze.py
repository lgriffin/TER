"""Command implementation module extracted from :mod:`ter_calculator.cli`."""

from __future__ import annotations

import sys
from pathlib import Path


def _cmd_analyze(args) -> int:
    """Execute the analyze subcommand."""
    # Resolve --latest flag
    if args.latest:
        from ..loader import find_latest_session

        args.session_path = str(find_latest_session(args.session_path))
        if not args.quiet:
            print(f"Using latest session: {args.session_path}", file=sys.stderr)
    elif args.session_path is None:
        print("Error: Either provide a session_path or use --latest", file=sys.stderr)
        return 1

    if args.group:
        return _cmd_analyze_group(args)

    from ..analyze_pipeline import analyze_session
    from ..formatter import format_ter_result

    result = analyze_session(args)
    rendered = format_ter_result(result, fmt=args.output_format)
    output_path = getattr(args, "analysis_output", None)
    if args.output_format == "html" and output_path is None:
        output_path = str(Path(args.session_path).with_suffix(".ter-report.html"))
    if output_path:
        Path(output_path).write_text(rendered, encoding="utf-8")
        if not args.quiet:
            print(
                f"Wrote {args.output_format.upper()} report: {output_path}",
                file=sys.stderr,
            )
    else:
        print(rendered)
    return 0


def _cmd_analyze_group(args) -> int:
    """Execute grouped analysis: parent + subagent sessions."""
    from ..loader import load_session, segment_spans, discover_subagents
    from ..intent import extract_intent
    from ..classifier import classify_spans
    from ..compute import compute_ter
    from ..waste import detect_waste_patterns
    from ..economics import compute_economics
    from ..formatter import format_grouped_analysis

    subagent_paths = discover_subagents(args.session_path)
    if not subagent_paths:
        print(
            "No subagent sessions found, running single-session analysis.",
            file=sys.stderr,
        )
        # Fall back to normal analyze (without --group).
        args.group = False
        return _cmd_analyze(args)

    from ..config_parse import parse_cost_model, parse_phase_weights

    phase_weights = parse_phase_weights(args.phase_weights)
    cost_model = parse_cost_model(args.cost_model)

    def _analyze_session(path):
        session = load_session(path)
        spans = segment_spans(session)
        intent = extract_intent(session)
        classified = classify_spans(
            spans,
            intent,
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
            result.waste_patterns = detect_waste_patterns(
                classified,
                restatement_threshold=args.restatement_threshold,
                session=session,
            )
        result.economics = compute_economics(session, classified, cost_model)
        return result

    if not args.quiet:
        print(
            f"Analyzing parent + {len(subagent_paths)} subagent(s)...", file=sys.stderr
        )

    parent_result = _analyze_session(args.session_path)
    subagent_results = []
    for p in subagent_paths:
        r = _analyze_session(str(p))
        # Use filename as session_id since subagents share the parent's sessionId.
        r.session_id = p.stem
        subagent_results.append(r)

    print(
        format_grouped_analysis(
            parent_result,
            subagent_results,
            fmt=args.output_format,
        )
    )
    return 0


def _cmd_compare(args) -> int:
    """Execute the compare subcommand."""

    from ..formatter import format_comparison

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
        from ..analyze_pipeline import analyze_session, default_analyze_args
        from ..session_report import format_baseline_markdown

        ra = analyze_session(default_analyze_args(paths[0]))
        rb = analyze_session(default_analyze_args(paths[1]))
        print(format_baseline_markdown(ra, rb))
        return 0

    from ..loader import load_session, segment_spans
    from ..intent import extract_intent
    from ..classifier import classify_spans
    from ..compute import compute_ter
    from ..economics import compute_economics
    from ..waste import detect_waste_patterns

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
