"""Command implementation for `ter visualize`."""

from __future__ import annotations

import sys
from pathlib import Path


def _cmd_visualize(args) -> int:
    """Generate SVG chart visualizations from a TER analysis."""
    if args.latest:
        from ..loader import find_latest_session

        args.session_path = str(find_latest_session(args.session_path))
        if not args.quiet:
            print(f"Using latest session: {args.session_path}", file=sys.stderr)
    elif args.session_path is None:
        print("Error: Either provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..analyze_pipeline import analyze_session
    from ..charts import generate_all_charts

    result = analyze_session(args)
    charts = generate_all_charts(result)

    if not charts:
        print("No charts generated (insufficient data).", file=sys.stderr)
        return 1

    output_dir = getattr(args, "output_dir", None)
    chart_filter = getattr(args, "charts", None)

    if chart_filter:
        selected = set(chart_filter.split(","))
        charts = {k: v for k, v in charts.items() if k in selected}
        if not charts:
            print(
                f"No matching charts. Available: {', '.join(generate_all_charts(result))}",
                file=sys.stderr,
            )
            return 1

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, svg in charts.items():
            path = out / f"{name}.svg"
            path.write_text(svg, encoding="utf-8")
            if not args.quiet:
                print(f"Wrote {path}", file=sys.stderr)
    else:
        session_stem = Path(args.session_path).stem
        out = Path(f"{session_stem}_charts")
        out.mkdir(parents=True, exist_ok=True)
        for name, svg in charts.items():
            path = out / f"{name}.svg"
            path.write_text(svg, encoding="utf-8")
            if not args.quiet:
                print(f"Wrote {path}", file=sys.stderr)

    if not args.quiet:
        print(
            f"\nGenerated {len(charts)} chart(s) in {out}/",
            file=sys.stderr,
        )
    return 0
