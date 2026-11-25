"""``python -m ter record``: the TER Record of a session (ADR 0007).

Commands::

    python -m ter record export SESSION [--out FILE] [--tokenizer NAME]
                                [--ter MODE] [--outcome FILE]
    python -m ter record verify RECORD SESSION [--outcome FILE]

``export`` writes one ``ter.record/0.1`` object for the session
(TER-REC-001): counts, classes, rule names, confidence, judged samples and
digests, with no prompt, path, file content, tool input or tool output
(TER-REC-002), so it can leave the machine the session ran on. Prices come
from the shipped price book; a session whose model has no price there has
its turns counted as unpriced, never guessed.

``verify`` rebuilds the record from the session it names, with the
tokenizer and TER 3 mode the record says it was built with, and lists every
figure that differs (TER-REC-005). Exit 0 when the session is the one the
record's events digest names and every figure matches, 1 when not, 2 when
the record or the session cannot be read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ...application.export_record import RecordCheck
from ...domain.capabilities import CapabilityError
from ...domain.events import SessionFormatError
from ...domain.outcome import OutcomeFormatError
from ...domain.ter_record import RECORD_SCHEMA

__all__ = ["RecordServices", "add_record_parser", "format_check", "run_record"]

#: Differences listed before the rest are counted, so a wholly different
#: record does not flood the terminal.
SHOWN = 40


@dataclass(frozen=True)
class RecordServices:
    """What ``python -m ter record`` needs, wired by the composition root."""

    #: ``export(session, tokenizer, ter_mode, outcome)``: the session's record.
    export: Callable[[Path, str, str, Path | None], dict[str, object]]
    #: ``verify(record, session, outcome)``: the record against its session.
    verify: Callable[[Mapping[str, Any], Path, Path | None], RecordCheck]
    #: Names of the usable tokenizers, to check a record's ``tokenizer``.
    tokenizers: Callable[[], tuple[str, ...]]


def add_record_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    tokenizer_help: str,
    ter_modes: tuple[str, ...],
    outcome_help: str,
) -> None:
    record = commands.add_parser(
        "record",
        help="the TER Record: a content-free efficiency record of a session",
    )
    sub = record.add_subparsers(dest="record_command", required=True)
    export = sub.add_parser("export", help="write a session's ter.record/0.1 object")
    export.add_argument("path", type=Path, help="session .jsonl or GARE run")
    export.add_argument(
        "--out", type=Path, metavar="FILE", help="write to FILE (default: stdout)"
    )
    export.add_argument("--tokenizer", default="regex", help=tokenizer_help)
    export.add_argument(
        "--ter",
        choices=ter_modes,
        default="offline",
        help="how to compute TER: offline (default), model or off",
    )
    export.add_argument("--outcome", type=Path, metavar="FILE", help=outcome_help)
    verify = sub.add_parser(
        "verify", help="recompute a record from its session and list differences"
    )
    verify.add_argument("record", type=Path, help="a ter.record/0.1 JSON file")
    verify.add_argument("path", type=Path, help="the session the record names")
    verify.add_argument("--outcome", type=Path, metavar="FILE", help=outcome_help)


def format_check(check: RecordCheck) -> str:
    lines: list[str] = []
    if not check.same_session:
        lines.append(
            "Not the same session: its events digest differs from the record's."
        )
    if check.ok:
        return "Verified: the session rebuilds this record exactly.\n"
    if check.differences:
        lines.append(f"{len(check.differences)} figure(s) differ (claimed → rebuilt):")
        for path, claimed, rebuilt in check.differences[:SHOWN]:
            lines.append(
                f"  {path}: {json.dumps(claimed, ensure_ascii=False)} → "
                f"{json.dumps(rebuilt, ensure_ascii=False)}"
            )
        if len(check.differences) > SHOWN:
            lines.append(f"  … and {len(check.differences) - SHOWN} more")
    return "\n".join(lines) + "\n"


def _read_record(path: Path) -> Mapping[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not JSON: {exc}") from None
    if not isinstance(doc, dict) or doc.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"{path} is not a {RECORD_SCHEMA} record")
    return doc


def _tokenizer_of(record: Mapping[str, Any]) -> str | None:
    produced = record.get("produced_by")
    name = produced.get("tokenizer") if isinstance(produced, Mapping) else None
    return name if isinstance(name, str) else None


def run_record(
    args: argparse.Namespace,
    services: RecordServices | None,
    out: IO[str],
    err: IO[str],
) -> int:
    if services is None:
        err.write("records are not available in this installation\n")
        return 2
    path: Path = args.path
    if not path.exists():
        err.write(f"No such session file: {path}\n")
        return 2
    outcome: Path | None = args.outcome
    if outcome is not None and not outcome.is_file():
        err.write(f"No such outcome file: {outcome}\n")
        return 2
    try:
        if args.record_command == "export":
            doc = services.export(path, args.tokenizer, args.ter, outcome)
            text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
            if args.out is None:
                out.write(text)
            else:
                if args.out.parent != Path():
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text, encoding="utf-8")
                err.write(f"Wrote {args.out}\n")
            return 0
        try:
            claimed = _read_record(args.record)
        except ValueError as exc:
            err.write(f"Cannot read record: {exc}\n")
            return 2
        tokenizer = _tokenizer_of(claimed)
        if tokenizer not in services.tokenizers():
            err.write(
                f"The record names tokenizer {tokenizer!r}, which is not "
                "installed here, so it cannot be rebuilt\n"
            )
            return 2
        check = services.verify(claimed, path, outcome)
    except OutcomeFormatError as exc:
        err.write(f"Cannot read outcome: {exc}\n")
        return 2
    except (SessionFormatError, CapabilityError) as exc:
        err.write(f"Cannot read session {path}: {exc}\n")
        return 2
    out.write(format_check(check))
    return 0 if check.ok else 1
