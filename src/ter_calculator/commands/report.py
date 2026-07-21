"""Command implementation module extracted from :mod:`ter_calculator.cli`."""

from __future__ import annotations

import sys
from pathlib import Path


def _cmd_report(args) -> int:
    """Markdown one-screen summary for humans."""
    # Resolve --latest flag
    if args.latest:
        from ..loader import find_latest_session

        args.session_path = str(find_latest_session(args.session_path))
        if not args.quiet:
            print(f"Using latest session: {args.session_path}", file=sys.stderr)
    elif args.session_path is None:
        print("Error: Either provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..analyze_pipeline import analyze_session
    from ..session_report import format_session_report_markdown

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
