"""``python -m ter control``: control charts over sessions (point 201).

Commands::

    python -m ter control measure SRC... --out MEASURES.json [--pseudonymise]
                                  [--tokenizer NAME]
    python -m ter control limits MEASURES.json --out LIMITS.json
                                 [--method average|median] [--keep LIMITS.json]
                                 [--date YYYY-MM-DD]
    python -m ter control chart MEASURES.json --limits LIMITS.json
                                [--html FILE] [--json [FILE]] [--title TEXT]

``measure`` is the slow step: it explains every session (a file, a folder of
``.jsonl`` files, or a corpus from ``python -m ter corpus import``) and writes
only the control measures, content-free. ``limits`` computes natural process
limits from a measures file; with ``--keep`` it carries a developer's tuning,
rule choices and switched-off measures over from earlier limits. ``chart``
draws the XmR charts and lists every signal; it exits 0 whatever it finds.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import IO, TYPE_CHECKING

from ...domain.lean.control import (
    ControlLimitsError,
    ControlReport,
    LimitMethod,
    MeasuresDocument,
    compute_limits,
    pseudonymise,
)

if TYPE_CHECKING:
    from ...application.control import ChartSessions, MeasuredSessions

__all__ = [
    "ControlServices",
    "add_control_parser",
    "format_control",
    "run_control",
    "session_files",
]

_METHODS = {
    "average": LimitMethod.AVERAGE_MOVING_RANGE,
    "median": LimitMethod.MEDIAN_MOVING_RANGE,
}


@dataclass(frozen=True)
class ControlServices:
    """The control use cases, wired by the composition root."""

    #: ``measure(paths, tokenizer)``: explain and measure each session.
    measure: Callable[[Sequence[Path], str], "MeasuredSessions"]
    chart: "ChartSessions"


def add_control_parser(
    commands: "argparse._SubParsersAction[argparse.ArgumentParser]",
    tokenizer_help: str,
) -> None:
    control = commands.add_parser(
        "control",
        help="control charts: natural process limits and signals over sessions",
    )
    sub = control.add_subparsers(dest="control_command", required=True)

    measure = sub.add_parser(
        "measure", help="explain sessions and write their control measures"
    )
    measure.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="session .jsonl files, folders of them, or an imported corpus",
    )
    measure.add_argument("--out", type=Path, required=True, metavar="FILE")
    measure.add_argument(
        "--pseudonymise",
        action="store_true",
        help="replace session ids with short hashes before writing",
    )
    measure.add_argument("--tokenizer", default="regex", help=tokenizer_help)

    limits = sub.add_parser(
        "limits", help="compute natural process limits from a measures file"
    )
    limits.add_argument("measures", type=Path, help="ter.control-measures/1 file")
    limits.add_argument("--out", type=Path, required=True, metavar="FILE")
    limits.add_argument(
        "--method",
        choices=sorted(_METHODS),
        default="average",
        help="sigma from the average moving range (default) or the median, "
        "which one wild session cannot inflate",
    )
    limits.add_argument(
        "--keep",
        type=Path,
        metavar="FILE",
        help="carry tuning, rules and switched-off measures over from these limits",
    )
    limits.add_argument("--date", help="date the limits were computed (default: today)")

    chart = sub.add_parser("chart", help="XmR charts and signals of measured sessions")
    chart.add_argument("measures", type=Path, help="ter.control-measures/1 file")
    chart.add_argument("--limits", type=Path, required=True, metavar="FILE")
    chart.add_argument("--html", type=Path, metavar="FILE", help="write the charts")
    chart.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help="write the report as JSON to FILE (stdout when FILE is omitted)",
    )
    chart.add_argument("--title", default="Control charts")


def session_files(sources: Sequence[Path]) -> Iterator[Path]:
    """Every session transcript under ``sources``, in path order.

    A corpus folder is read from its ``sessions/``; subagent transcripts are
    part of their parent session and are skipped.
    """
    for source in sources:
        if source.is_file():
            yield source
            continue
        root = source / "sessions" if (source / "sessions").is_dir() else source
        for path in sorted(root.rglob("*.jsonl")):
            if "subagents" not in path.relative_to(root).parts:
                yield path


def _write(path: Path, text: str) -> None:
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json(value: object) -> str:
    # allow_nan=False: a policy reads these files, so no bare Infinity or NaN.
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _read_measures(path: Path) -> MeasuresDocument:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ControlLimitsError(f"{path}: no such measures file") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ControlLimitsError(f"{path}: cannot read: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ControlLimitsError(f"{path}: not JSON: {exc}") from None
    return MeasuresDocument.from_mapping(document)


def run_control(
    args: argparse.Namespace,
    services: ControlServices | None,
    out: IO[str],
    err: IO[str],
) -> int:
    if services is None:
        err.write("control is not available in this installation\n")
        return 2
    try:
        if args.control_command == "measure":
            return _measure(args, services, err)
        if args.control_command == "limits":
            return _limits(args, services, err)
        return _chart(args, services, out, err)
    except ControlLimitsError as exc:
        err.write(f"ter control: {exc}\n")
        return 2


def _measure(args: argparse.Namespace, services: ControlServices, err: IO[str]) -> int:
    missing = [s for s in args.sources if not s.exists()]
    if missing:
        err.write(f"No such session file or folder: {missing[0]}\n")
        return 2
    paths = list(session_files(args.sources))
    if not paths:
        err.write("No session transcripts (.jsonl) found\n")
        return 2
    measured = services.measure(paths, args.tokenizer)
    rows = pseudonymise(measured.rows) if args.pseudonymise else measured.rows
    if rows:
        _write(args.out, _json(MeasuresDocument(rows).as_dict()))
        err.write(f"Wrote {args.out}: {len(rows)} session(s) measured\n")
    for ref, error in measured.failures:
        err.write(f"  ! {ref}: {error}\n")
    if not rows:
        err.write("No session could be measured\n")
        return 1
    return 1 if measured.failures else 0


def _limits(args: argparse.Namespace, services: ControlServices, err: IO[str]) -> int:
    document = _read_measures(args.measures)
    keep = None if args.keep is None else services.chart.limits(args.keep)
    computed_on = args.date or date.today().isoformat()
    limits = compute_limits(
        document.rows, _METHODS[args.method], keep=keep, computed_on=computed_on
    )
    if not limits.measures:
        err.write("No measure has enough sessions for control limits\n")
        return 1
    _write(args.out, _json(limits.as_dict()))
    provisional = sum(1 for m in limits.measures if m.natural.provisional)
    err.write(
        f"Wrote {args.out}: limits for {len(limits.measures)} measure(s) from "
        f"{limits.sessions} session(s)"
        + (f", {provisional} provisional" if provisional else "")
        + "\n"
    )
    return 0


def _chart(
    args: argparse.Namespace, services: ControlServices, out: IO[str], err: IO[str]
) -> int:
    document = _read_measures(args.measures)
    report = services.chart(document.rows, args.limits)
    if report.stale:
        err.write(
            "Warning: these limits were computed with another detector set "
            f"({report.limits.detectors}, sessions {report.detectors}); "
            "recompute them\n"
        )
    wrote = False
    if args.html is not None:
        from .reports.control import render_control_html

        _write(args.html, render_control_html(report, title=args.title))
        err.write(f"Wrote {args.html}\n")
        wrote = True
    if args.json is not None:
        payload = _json(report.as_dict())
        if args.json == "-":
            out.write(payload)
        else:
            _write(Path(args.json), payload)
            err.write(f"Wrote {args.json}\n")
        wrote = True
    if not wrote:
        out.write(format_control(report))
    elif args.json != "-":
        # Files only: still say what they hold.
        err.write(format_control(report).splitlines()[0] + "\n")
    return 0


def format_control(report: ControlReport) -> str:
    """Plain-text summary: each measure's limits, then every signal."""
    lines = [
        f"Control charts · {report.sessions} sessions · "
        f"{len(report.signals)} signal(s), {len(report.firing)} would fire"
    ]
    if report.stale:
        lines.append("  ! limits are stale: computed with another detector set")
    for chart in report.charts:
        lim = chart.limits

        def f(v: float | None) -> str:
            return "-" if v is None else f"{v:.4g}"

        flags = []
        if lim.tuning is not None:
            flags.append("tuned")
        if lim.natural.provisional:
            flags.append("provisional")
        if not lim.enabled:
            flags.append("off")
        if not chart.measure.fires:
            flags.append("never fires")
        lines.append(
            f"  {chart.measure.key:<28} CL {f(lim.natural.centre):>8}  "
            f"LCL {f(lim.lcl):>8}  UCL {f(lim.ucl):>8}  "
            f"{len(chart.signals):>3} signal(s)"
            + (f"  [{', '.join(flags)}]" if flags else "")
        )
        for s in chart.signals:
            mark = "FIRE" if s.fires else ("worse" if s.unfavourable else "better")
            lines.append(
                f"      {mark:<6} {s.rule.value:<14} {s.side.value} {f(s.limit)} at "
                f"{s.session_id} ({f(s.value)})"
            )
    return "\n".join(lines) + "\n"
