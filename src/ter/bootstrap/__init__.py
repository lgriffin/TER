"""Composition root: the only place that chooses concrete adapters.

Entry points (CLI, hooks, CI gate) ask this package for wired use cases, and
it applies the installation's maturity ceiling while doing so. No other
module decides which capabilities are switched on.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..adapters.driving.cli import CliServices
from ..adapters.driving.cli import main as cli_main
from ..adapters.driving.record_cli import RecordServices
from ..adapters.driven.in_memory import SystemClock
from ..application.explain import ExplainedSession, ExplainSession
from ..application.export_record import RecordCheck
from ..application.route import RoutedSession, RouteSession
from .capabilities import (
    CapabilityRegistry,
    default_registry,
    detector_registry,
    repository_evidence,
)
from ..application.observe import (
    AnalyseEventLog,
    AnalyseTrace,
    ObserveEvent,
    RecordEvent,
)
from ..domain.capabilities import Capability, CapabilityProblem, UnknownCapabilityError
from ..domain.stream import StreamReport
from ..ports.driven import (
    ArchitectureContracts,
    InterventionLedger,
    InterventionPolicies,
    OutcomeSource,
    PriceBook,
    SessionSource,
    TerScorer,
    Tokenizer,
)
from ..ports.driving import EventIngest

if TYPE_CHECKING:
    from ..adapters.driven.claude_code.corpus import CorpusImport
    from ..adapters.driving.claude_hooks.check import HookCheck
    from ..adapters.driving.control_cli import ControlServices
    from ..adapters.driving.policy_cli import PolicyServices
    from ..application.control import MeasuredSessions
    from ..application.shadow import ShadowRun
    from ..domain.lean.control import ControlLimits
    from ..domain.policy_set import PolicySet

__all__ = [
    "CapabilityRegistry",
    "cli_services",
    "default_registry",
    "default_event_log_dir",
    "detector_registry",
    "main",
    "utf8_stdio",
    "make_contracts",
    "make_ingest",
    "make_outcome_source",
    "make_ledger",
    "make_policies",
    "make_recorder",
    "make_ter_scorer",
    "make_tokenizer",
    "session_folder_files",
    "session_source_for",
]

#: Environment variable that relocates the live event log.
EVENT_LOG_ENV = "TER_EVENT_LOG_DIR"


def default_event_log_dir() -> Path:
    """``$TER_EVENT_LOG_DIR``, else ``ter/events`` in the user's cache directory.

    The log holds prompts and tool output, so the default lives under the
    user's home (``$XDG_CACHE_HOME`` or ``~/.cache``), not a shared temporary
    directory.
    """
    configured = os.environ.get(EVENT_LOG_ENV)
    if configured:
        return Path(configured)
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache) if cache else Path.home() / ".cache"
    return base / "ter" / "events"


def make_tokenizer(name: str = "regex") -> Tokenizer:
    """``regex`` is offline and deterministic; ``tiktoken`` needs its encoding.

    Any ``Tokenizer.<name>`` capability works; built-ins resolve without
    scanning installed packages, so hooks stay cheap.
    """
    try:
        tokenizer = default_registry().create("Tokenizer", name)
    except UnknownCapabilityError:
        raise ValueError(f"Unknown tokenizer {name!r}") from None
    assert isinstance(tokenizer, Tokenizer)  # checked by the registry
    return tokenizer


def make_outcome_source(name: str = "junit") -> OutcomeSource:
    """An ``OutcomeSource.<name>`` capability; ``junit`` reads JUnit XML results."""
    source = default_registry().create("OutcomeSource", name)
    assert isinstance(source, OutcomeSource)  # checked by the registry
    return source


def make_policies(name: str = "json") -> InterventionPolicies:
    """An ``InterventionPolicies.<name>`` capability (TER-ARC-008): ``json``
    reads ``ter.intervention-policies/1`` files. Policies load through the
    capability registry like every other adapter, so a pack can add a source."""
    source = default_registry().create("InterventionPolicies", name)
    assert isinstance(source, InterventionPolicies)  # checked by the registry
    return source


def make_ledger(directory: Path, name: str = "jsonl") -> InterventionLedger:
    """An ``InterventionLedger.<name>`` capability (TER-INT-009): ``jsonl``
    keeps one append-only ledger file per session under ``directory``."""
    ledger = default_registry().create("InterventionLedger", name, directory)
    assert isinstance(ledger, InterventionLedger)  # checked by the registry
    return ledger


def make_contracts(name: str = "import-linter") -> ArchitectureContracts:
    """An ``ArchitectureContracts.<name>`` capability; ``import-linter`` reads
    import-linter configuration (TER-EVD-007), ``dependency-cruiser`` the
    forbidden rules of ``.dependency-cruiser.json`` (TER-EVD-015)."""
    reader = default_registry().create("ArchitectureContracts", name)
    assert isinstance(reader, ArchitectureContracts)  # checked by the registry
    return reader


def make_ter_scorer(mode: str) -> TerScorer | None:
    """``offline`` pins TER 3 to the deterministic adapters; ``model`` uses its
    sentence-transformers model; ``off`` skips TER.

    The scorer, and the tokenizer and embedder it is pinned to, are
    capabilities (``TerScorer.ter3``, ``Tokenizer.regex``, ``Embedder.hashing``).
    """
    if mode == "off":
        return None
    registry = default_registry()
    if mode == "model":
        scorer = registry.create("TerScorer", "ter3")
    elif mode == "offline":
        scorer = registry.create(
            "TerScorer",
            "ter3",
            make_tokenizer("regex"),
            registry.create("Embedder", "hashing"),
        )
    else:
        raise ValueError(f"Unknown TER mode {mode!r}")
    assert isinstance(scorer, TerScorer)  # checked by the registry
    return scorer


def make_price_book(path: Path | None = None, details: Path | None = None) -> PriceBook:
    """The price book costs are read from (ADR 0003, ADR 0008).

    None is the book shipped with TER. A ``.csv`` file is a FOCUS SKU Price
    file (``PriceBook.focus-sku-price``), whose token labels come from the
    file or from the FOCUS Cost and Usage file ``details``; any other file is
    a ``ter.price_book/1`` JSON book (``PriceBook.json``).
    """
    from ..adapters.driven.pricing import default_price_book

    if path is None:
        if details is not None:
            raise ValueError("--price-details needs a FOCUS SKU Price file (--prices)")
        return default_price_book()
    registry = default_registry()
    if path.suffix.lower() == ".csv":
        book = registry.create("PriceBook", "focus-sku-price", path, details=details)
    else:
        if details is not None:
            raise ValueError(
                "--price-details is read only beside a FOCUS SKU Price file"
            )
        book = registry.create("PriceBook", "json", path)
    assert isinstance(book, PriceBook)  # checked by the registry
    return book


def session_source_for(path: Path) -> SessionSource:
    """The agent adapter for a reference, from the ``SessionSource`` capabilities.

    A source that can tell its own files apart declares a static
    ``accepts(ref)``; the first, by name, that accepts ``path`` reads it (a
    GARE export, say). Anything else is a Claude Code transcript.
    """
    registry = default_registry()
    for cap in registry.capabilities("SessionSource"):
        if cap.name == "claude-code":
            continue
        try:
            accepts = getattr(registry.factory(cap.port, cap.name), "accepts", None)
            if not (callable(accepts) and bool(accepts(path))):
                continue
            source = registry.create(cap.port, cap.name)
        except Exception:  # a broken plugin must not stop Claude Code transcripts
            continue
        assert isinstance(source, SessionSource)  # checked by the registry
        return source
    source = registry.create("SessionSource", "claude-code")
    assert isinstance(source, SessionSource)
    return source


def session_folder_files(folder: Path) -> tuple[Path, ...]:
    """The files of the one session exported to ``folder`` (a GARE run), or none.

    A source that reads a whole folder as one session declares a static
    ``run_files(folder)``; a corpus walk passes such a folder whole and skips
    only these files, so other transcripts beside them are still found.
    """
    registry = default_registry()
    for cap in registry.capabilities("SessionSource"):
        try:
            run_files = getattr(registry.factory(cap.port, cap.name), "run_files", None)
            files = tuple(run_files(folder)) if callable(run_files) else ()
        except Exception:  # a broken plugin must not stop Claude Code transcripts
            continue
        if files:
            return files
    return ()


def make_ingest(tokenizer: str = "regex") -> EventIngest:
    """A fresh :class:`EventIngest` for analysing one recorded session.

    Every recorded path (transcript, GARE run, event log replay) applies its
    events through one of these (TER-OBS-001); the detectors it runs are the
    installed ``WasteDetector`` capabilities (TER-ARC-002).
    """
    return ObserveEvent(make_tokenizer(tokenizer), detectors=detector_registry())


def make_recorder(directory: Path) -> EventIngest:
    """The :class:`EventIngest` a hook process records live events through."""
    from ..adapters.driven.event_log import JsonlEventLog

    # Append-only: a hook's cost must not grow with the session. A hook
    # process never explains, so it skips plugin discovery too.
    return RecordEvent(make_tokenizer("regex"), JsonlEventLog(directory))


def _control_limits(path: Path | None) -> "ControlLimits | None":
    """The control limits file, or ``None`` without one."""
    if path is None:
        return None
    from ..adapters.driven.control_limits import JsonControlLimits

    return JsonControlLimits().limits(path)


def advise_from_log(
    directory: Path,
    session_id: str,
    policies: Path,
    ledger: Path,
    warn: Callable[[str], None] | None = None,
    limits: Path | None = None,
) -> tuple[str, ...]:
    """The advice due now for a live session (L4 Advisory, TER-INT-003).

    One turn of the control loop (TER-INT-013): observe the session's event
    log, analyse it, decide on its signals under the approved policy file,
    deliver the advice not given before as messages to the developer, then
    measure earlier advice. Learn is left to ``python -m ter policy
    effectiveness``: it reads the whole ledger, and a hook has nowhere to show
    its report. With ``limits``, the session is placed against its control
    limits and a policy may name a firing control signal (TER-SPC-020).
    Raises when the policy or limits file is unapproved or unreadable; the
    hook turns that into no advice. A failed Measure keeps the advice and is
    passed to ``warn``, as are stale limits.
    """
    from ..adapters.driven.event_log import JsonlEventLog
    from ..adapters.driven.intervention_channel import MessageChannel
    from ..application.loop import ControlLoop

    policy_set = make_policies().policies(policies)
    policy_set.require_approved()
    channel = MessageChannel()
    loop = ControlLoop(
        policy_set,
        make_ledger(ledger),
        SystemClock(),
        channel,
        learn=False,
        limits=_control_limits(limits),
    )
    turn = loop.turn(session_id, JsonlEventLog(directory), lambda: make_ingest("regex"))
    if warn is not None:
        for stage, error in turn.errors:
            warn(f"{stage} failed: {error}")
        if turn.stale_limits:
            warn(
                "control signals not used: the limits came from another "
                "detector set; re-measure and recompute them"
            )
    return channel.delivered()


def cli_services() -> CliServices:
    """Wire the CLI's use cases. Heavy adapters are imported on first use."""

    def analyse_transcript(path: Path, tokenizer: str) -> StreamReport:
        return AnalyseTrace(
            session_source_for(path),
            make_tokenizer(tokenizer),
            lambda: make_ingest(tokenizer),
        )(path)

    def log_sessions(directory: Path) -> tuple[str, ...]:
        from ..adapters.driven.event_log import JsonlEventLog

        return JsonlEventLog(directory).sessions()

    def analyse_log(directory: Path, session_id: str, tokenizer: str) -> StreamReport:
        from ..adapters.driven.event_log import JsonlEventLog

        log = JsonlEventLog(directory)
        return AnalyseEventLog(
            log, make_tokenizer(tokenizer), lambda: make_ingest(tokenizer)
        )(session_id)

    def hook_ingest(directory: Path) -> EventIngest:
        return make_recorder(directory)

    def hook_advise(
        directory: Path,
        session_id: str,
        policies: Path,
        ledger: Path,
        warn: Callable[[str], None],
        limits: Path | None,
    ) -> tuple[str, ...]:
        return advise_from_log(directory, session_id, policies, ledger, warn, limits)

    def explain_transcript(
        path: Path,
        tokenizer: str,
        ter: str,
        outcome: Path | None = None,
        repo: Path | None = None,
        repo_engine: str = "syntax",
        prices: PriceBook | None = None,
    ) -> ExplainedSession:
        from ..adapters.driven.claude_code import ClaudeCodeJsonlSource
        from ..adapters.driven.pricing import default_price_book

        source = session_source_for(path)
        # TER 3 scores Claude Code transcripts only; other sources get no TER
        # rather than a meaningless one (TER-SRC-017).
        scores = isinstance(source, ClaudeCodeJsonlSource)
        use_case = ExplainSession(
            source,
            make_tokenizer(tokenizer),
            make_ter_scorer(ter if scores else "off"),
            make_outcome_source() if outcome is not None else None,
            prices if prices is not None else default_price_book(),
            lambda: make_ingest(tokenizer),
            # L3: the repository as it was when the session started.
            repository_evidence(repo, repo_engine) if repo is not None else None,
            (
                (make_contracts(), make_contracts("dependency-cruiser"))
                if repo is not None
                else None
            ),
        )
        return use_case(path, outcome)

    def route_transcript(
        path: Path,
        tokenizer: str,
        profile: str | None = None,
        repo: Path | None = None,
        repo_engine: str = "syntax",
        profiles_dir: Path | None = None,
    ) -> RoutedSession:
        from ..adapters.driven.routing_profiles import (
            JsonRoutingProfiles,
            default_routing_profiles,
        )

        # Advisory routing (L3): decisions and route.escalated events for
        # analysis, never sent to a live session (TER-INT-001).
        use_case = RouteSession(
            session_source_for(path),
            make_tokenizer(tokenizer),
            default_routing_profiles()
            if profiles_dir is None
            else JsonRoutingProfiles(profiles_dir),
            lambda: make_ingest(tokenizer),
            repository_evidence(repo, repo_engine) if repo is not None else None,
            make_contracts() if repo is not None else None,
        )
        return use_case(path, profile)

    def capabilities() -> tuple[tuple[Capability, ...], tuple[CapabilityProblem, ...]]:
        registry = default_registry()
        problems = registry.check()
        return registry.capabilities(), problems

    def tokenizers() -> tuple[str, ...]:
        # Only tokenizers that load and fit the port: a broken plugin is
        # reported by `capabilities`, not crashed into by a report command.
        registry = default_registry()
        broken = {(p.key, p.target) for p in registry.check("Tokenizer")}
        return tuple(
            c.name
            for c in registry.capabilities("Tokenizer")
            if (c.key, c.target) not in broken
        )

    def import_corpus(
        sources: Sequence[Path],
        out: Path,
        labels: Path | None,
        max_tool_output: int,
        keep_tools: frozenset[str],
        quote_files: bool,
    ) -> CorpusImport:
        from ..adapters.driven.claude_code.redaction import RedactionPolicy
        from ..adapters.driven.claude_code.corpus import (
            import_corpus as run,
            read_labels,
        )

        policy = RedactionPolicy(
            max_tool_output=max_tool_output,
            keep_tools=keep_tools,
            quote_file_contents=quote_files,
        )
        return run(
            sources,
            out,
            policy=policy,
            labels=read_labels(labels) if labels is not None else None,
        )

    def hooks_check(recordings: Path, transcripts: Path) -> "HookCheck":
        from ..adapters.driven.claude_code import ClaudeCodeJsonlSource
        from ..adapters.driving.claude_hooks.check import check_recordings

        return check_recordings(recordings, transcripts, ClaudeCodeJsonlSource().read)

    def control() -> "ControlServices":
        from ..adapters.driven.control_limits import JsonControlLimits
        from ..adapters.driving.control_cli import ControlServices
        from ..application.control import ChartSessions, measure_sessions

        def measure(paths: Sequence[Path], tokenizer: str) -> "MeasuredSessions":
            # TER off: the control measures are the Lean ones, and a corpus
            # must not download a model (as scripts/corpus_findings.py).
            return measure_sessions(
                paths, lambda path: explain_transcript(path, tokenizer, "off")
            )

        return ControlServices(
            measure, ChartSessions(JsonControlLimits()), session_folder_files
        )

    from .context import context_services

    def policies() -> "PolicyServices":
        from ..adapters.driving.policy_cli import PolicyServices

        from ..application.shadow import shadow_sessions

        def shadow(
            paths: Sequence[Path],
            policies: "PolicySet",
            ledger: Path,
            tokenizer: str,
            limits: Path | None,
        ) -> "ShadowRun":
            # TER off, as for control measures: shadow mode replays a corpus
            # and must not download a model.
            return shadow_sessions(
                paths,
                lambda path: explain_transcript(path, tokenizer, "off"),
                policies,
                make_ledger(ledger),
                SystemClock(),
                _control_limits(limits),
            )

        return PolicyServices(
            make_policies,
            lambda: SystemClock().now().date(),
            shadow,
            session_folder_files,
            make_ledger,
        )

    def records() -> RecordServices:
        from ..adapters.driven.calibration import default_calibration
        from ..application.export_record import (
            check_record,
            record_of,
            ter_mode_of,
        )

        def export(
            path: Path,
            tokenizer: str,
            ter: str,
            outcome: Path | None,
            prices: Path | None = None,
            details: Path | None = None,
        ) -> dict[str, object]:
            explained = explain_transcript(
                path, tokenizer, ter, outcome, prices=make_price_book(prices, details)
            )
            return record_of(
                explained,
                tokenizer=tokenizer,
                calibration=default_calibration().calibration(),
            )

        def verify(
            record: Mapping[str, Any],
            path: Path,
            outcome: Path | None,
            prices: Path | None = None,
            details: Path | None = None,
        ) -> RecordCheck:
            produced = record.get("produced_by")
            tokenizer = (
                str(produced.get("tokenizer"))
                if isinstance(produced, Mapping)
                else "regex"
            )
            explained = explain_transcript(
                path,
                tokenizer,
                ter_mode_of(record),
                outcome,
                prices=make_price_book(prices, details),
            )
            return check_record(
                record,
                explained,
                tokenizer=tokenizer,
                calibration=default_calibration().calibration(),
            )

        return RecordServices(export, verify, tokenizers)

    return CliServices(
        records=records(),
        policies=policies(),
        control=control(),
        context=context_services(session_source_for, make_tokenizer),
        hooks_check=hooks_check,
        capabilities=capabilities,
        import_corpus=import_corpus,
        tokenizers=tokenizers,
        analyse_transcript=analyse_transcript,
        log_sessions=log_sessions,
        analyse_log=analyse_log,
        hook_ingest=hook_ingest,
        hook_advise=hook_advise,
        default_log_dir=default_event_log_dir(),
        hook_clock=SystemClock(),
        explain_transcript=explain_transcript,
        route_transcript=route_transcript,
    )


def utf8_stdio(*streams: object) -> None:
    """Write UTF-8 to standard output and error whatever the platform says.

    On Windows a piped stream defaults to the ANSI code page (cp1252), which
    cannot encode the report's symbols (``−``, ``→``): ``python -m ter
    explain --json | ...`` crashed with UnicodeEncodeError (12 Oct 2026).
    A stream that is already UTF-8, or cannot be reconfigured, is left as it
    is.
    """
    for stream in streams or (sys.stdout, sys.stderr):
        encoding = str(getattr(stream, "encoding", "") or "")
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None or encoding.lower().replace("-", "") == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            continue


def main(argv: Sequence[str] | None = None) -> int:
    utf8_stdio()
    return cli_main(argv, cli_services())
