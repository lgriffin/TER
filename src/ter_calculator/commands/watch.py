"""Command implementation module extracted from :mod:`ter_calculator.cli`."""

from __future__ import annotations

import sys
from pathlib import Path

_last_was_live = False


def _signal_to_dict(signal) -> dict:
    """Convert a TERSignal to a JSON-serialisable dict."""
    from datetime import datetime, timezone

    return {
        "session_id": signal.session_id,
        "timestamp": datetime.fromtimestamp(
            signal.timestamp, tz=timezone.utc
        ).isoformat(),
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
        msg_time = (
            datetime.fromtimestamp(signal.timestamp, tz=timezone.utc)
            .astimezone()
            .strftime("%H:%M:%S")
        )
        drift_arrow = (
            "↑"
            if signal.drift.value == "improving"
            else "↓"
            if signal.drift.value == "degrading"
            else "→"
        )
        waste_pct = (
            (signal.waste_tokens / signal.total_tokens * 100)
            if signal.total_tokens > 0
            else 0
        )
        aligned_pct = (
            (signal.aligned_tokens / signal.total_tokens * 100)
            if signal.total_tokens > 0
            else 0
        )

        print(
            f"[{msg_time}] [{tag}] [{signal.session_id[:8]}] TER: {signal.aggregate_ter:.2f} (weighted) | "
            f"Raw: {signal.raw_ratio:.2f} {drift_arrow} | "
            f"Tokens: {signal.total_tokens:,} | "
            f"Aligned: {signal.aligned_tokens:,} ({aligned_pct:.0f}%) | "
            f"Waste: {signal.waste_tokens:,} ({waste_pct:.0f}%)",
            flush=True,
        )

        # Show phase breakdown if available
        if hasattr(signal, "phase_ter") and signal.phase_ter:
            phase_strs = [f"{p[:3]}: {ter:.2f}" for p, ter in signal.phase_ter.items()]
            print(f"  Phases: {' | '.join(phase_strs)}", flush=True)

        # Show waste sources if available
        if hasattr(signal, "waste_sources") and signal.waste_sources:
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
    from ..real_time import LiveDashboard, SessionMonitor, load_embedding_model

    # Dashboard is default for text format, unless --stream is specified
    use_stream = getattr(args, "stream", False)
    use_dashboard = args.output_format == "text" and not use_stream

    # Load embedding model for accurate live classification
    try:
        model = load_embedding_model()
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    single_file = None

    if args.latest:
        from ..loader import find_latest_session

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
                    print(
                        f"Watching session file: {session_jsonl.name}", file=sys.stderr
                    )
        elif not target.exists():
            print(f"Error: Project path not found: {target}", file=sys.stderr)
            return 1

    log_fh = None
    log_path = getattr(args, "log_file", None)
    signal_count = 0

    try:
        if log_path:
            log_fh = open(log_path, "a", encoding="utf-8")

        # For dashboard mode, we'll handle on_signal differently
        if use_dashboard:
            latest_signal = [None]  # Mutable container for signal sharing

            def on_signal(sig):
                nonlocal signal_count
                signal_count += 1
                latest_signal[0] = sig
                if log_fh:
                    import json as json_mod

                    log_fh.write(json_mod.dumps(_signal_to_dict(sig)) + "\n")
                    log_fh.flush()
        else:

            def on_signal(sig):
                nonlocal signal_count
                signal_count += 1
                _print_signal(sig, args.output_format, log_fh=log_fh)

        monitor: SessionMonitor | LiveDashboard
        if single_file is not None:
            monitor = SessionMonitor(
                single_file,
                poll_interval=args.poll_interval,
                model=model,
                on_signal=on_signal,
            )
        else:
            monitor = LiveDashboard(
                project_dir=Path(args.project_path),
                poll_interval=args.poll_interval,
                model=model,
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
        if use_dashboard:
            # Dashboard mode with Rich Live display (default for text format)
            try:
                from rich.live import Live
                from rich.console import Console
                from ..dashboard import format_dashboard_with_history
            except ImportError:
                print(
                    "Error: Rich library not found. Install with: pip install rich",
                    file=sys.stderr,
                )
                print("Falling back to stream mode...", file=sys.stderr)
                use_dashboard = False
                # Fall through to stream mode below

            if use_dashboard:
                console = Console()
                print(f"Watching: {watch_target}", file=sys.stderr)
                if log_path:
                    print(f"Logging to: {log_path}", file=sys.stderr)
                print(
                    "Press Ctrl+C to stop... (use --stream for line-by-line mode)\n",
                    file=sys.stderr,
                )

                with Live(console=console, refresh_per_second=4) as live:
                    import threading
                    import time

                    stop_event = threading.Event()

                    def update_display():
                        while not stop_event.is_set():
                            if latest_signal[0] is not None:
                                recent_ter_values = []
                                if hasattr(monitor, "state"):
                                    recent_ter_values = monitor.state.recent_ter_values
                                elif (
                                    hasattr(monitor, "_monitors") and monitor._monitors
                                ):
                                    first_monitor = next(
                                        iter(monitor._monitors.values())
                                    )
                                    recent_ter_values = (
                                        first_monitor.state.recent_ter_values
                                    )

                                dashboard_renderable = format_dashboard_with_history(
                                    latest_signal[0], recent_ter_values
                                )
                                live.update(dashboard_renderable)
                            time.sleep(0.25)

                    display_thread = threading.Thread(
                        target=update_display, daemon=True
                    )
                    display_thread.start()

                    try:
                        monitor.run()
                    except KeyboardInterrupt:
                        monitor.stop()
                        raise
                    finally:
                        stop_event.set()
                        display_thread.join(timeout=1.0)

        if not use_dashboard:
            # Stream line-by-line mode (--stream flag or JSON format)
            if args.output_format == "text":
                print(f"Watching: {watch_target}", flush=True)
                if log_path:
                    print(f"Logging to: {log_path}", flush=True)
                print("Press Ctrl+C to stop...\n", flush=True)
            monitor.run()

    except KeyboardInterrupt:
        monitor.stop()
        if not use_dashboard and args.output_format == "text":
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
