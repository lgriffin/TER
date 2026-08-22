"""Command implementation for `ter present`."""

from __future__ import annotations

import sys
from pathlib import Path


def _cmd_present(args) -> int:
    """Generate a Marp presentation from a TER analysis."""
    if args.latest:
        from ..loader import find_latest_session

        args.session_path = str(find_latest_session(args.session_path))
        if not args.quiet:
            print(f"Using latest session: {args.session_path}", file=sys.stderr)
    elif args.session_path is None:
        print("Error: Either provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..analyze_pipeline import analyze_session
    from ..formatter_marp import format_marp

    result = analyze_session(args)
    marp_md = format_marp(result)

    output_path = getattr(args, "present_output", None)
    if output_path is None:
        output_path = str(Path(args.session_path).with_suffix(".ter-slides.md"))

    Path(output_path).write_text(marp_md, encoding="utf-8")
    if not args.quiet:
        print(f"Wrote Marp presentation: {output_path}", file=sys.stderr)
        print(
            "Render with: npx @marp-team/marp-cli " + output_path,
            file=sys.stderr,
        )
    return 0
