"""Phase 4 history, profiles, prediction, and dashboard commands."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from ..history_store import (
    HistoryRecord,
    TERHistoryStore,
    prompt_fingerprint,
    waste_breakdown,
)


def _store(args) -> TERHistoryStore:
    return TERHistoryStore(getattr(args, "db", None))


def _cmd_history(args) -> int:
    if args.history_command == "record":
        return _record(args)
    if args.history_command == "list":
        return _list(args)
    if args.history_command == "profile":
        return _profile(args)
    if args.history_command == "predict":
        return _predict(args)
    raise ValueError("Choose a history subcommand: record, list, profile, or predict")


def _record(args) -> int:
    from ..analyze_pipeline import analyze_session, default_analyze_args
    from ..loader import load_session

    analysis_args = default_analyze_args(args.session_path)
    result = analyze_session(analysis_args)
    session = load_session(args.session_path)
    prompt = args.prompt
    if prompt is None and session.user_prompts:
        prompt = session.user_prompts[0]
    economics = result.economics
    record = HistoryRecord(
        session_id=result.session_id,
        project=args.project or str(Path(args.session_path).resolve().parent),
        timestamp=Path(args.session_path).stat().st_mtime or time.time(),
        aggregate_ter=result.aggregate_ter,
        phase_ter=result.phase_scores,
        waste_breakdown=waste_breakdown(result),
        token_count=result.total_tokens,
        waste_tokens=result.waste_tokens,
        cost_usd=economics.estimated_cost_usd if economics else 0.0,
        waste_cost_usd=economics.estimated_waste_cost_usd if economics else 0.0,
        prompt_fingerprint=prompt_fingerprint(prompt) if prompt else None,
    )
    store = _store(args)
    try:
        store.put(record)
        print(f"Recorded {record.session_id} in {store.path}")
    finally:
        store.close()
    return 0


def _list(args) -> int:
    store = _store(args)
    try:
        records = store.query(
            project=args.project,
            min_ter=args.min_ter,
            max_ter=args.max_ter,
            limit=args.limit,
        )
        if args.output_format == "json":
            print(json.dumps([r.__dict__ for r in records], indent=2, sort_keys=True))
        else:
            print("DATE        TER    TOKENS   WASTE   COST      PROJECT / SESSION")
            for r in records:
                date = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d")
                print(
                    f"{date}  {r.aggregate_ter:0.3f}  {r.token_count:7,d}  {r.waste_tokens:6,d}  ${r.cost_usd:7.3f}  {r.project} / {r.session_id}"
                )
    finally:
        store.close()
    return 0


def _profile(args) -> int:
    store = _store(args)
    try:
        profile = store.profile(args.project)
        if args.output_format == "json":
            print(json.dumps(profile, indent=2, sort_keys=True))
        else:
            _print_profile(profile)
    finally:
        store.close()
    return 0


def _predict(args) -> int:
    store = _store(args)
    try:
        prediction = store.predict(args.prompt, args.project, k=args.neighbors)
        if args.output_format == "json":
            print(json.dumps(prediction, indent=2, sort_keys=True))
        elif not prediction["available"]:
            print(
                f"No comparable prompt history for {args.project!r} (samples: {prediction['sample_size']})."
            )
        else:
            print(f"Predicted TER: {prediction['predicted_ter']:.3f}")
            print(
                f"Confidence: {prediction['confidence']} ({prediction['neighbors']} neighbors; {prediction['sample_size']} samples)"
            )
            print(f"Recommendation: {prediction['recommendation']}")
    finally:
        store.close()
    return 0


def _cmd_dashboard(args) -> int:
    store = _store(args)
    try:
        profile = store.profile(args.project)
        _print_profile(profile, title="TER Cost Optimization Dashboard")
        records = store.query(project=args.project, limit=args.limit)
        if records:
            values = list(reversed([r.aggregate_ter for r in records]))
            spark = "".join("▁▂▃▄▅▆▇█"[min(7, max(0, int(v * 8)))] for v in values)
            print(f"TER trend: {spark}")
    finally:
        store.close()
    return 0


def _print_profile(
    profile: dict[str, object], title: str = "TER Project Profile"
) -> None:
    print(title)
    print("=" * len(title))
    if not profile.get("sessions"):
        print("No recorded sessions.")
        return
    print(f"Project: {profile['project']}")
    print(f"Sessions: {profile['sessions']}")
    print(f"Average TER: {profile['average_ter']:.3f}")
    print(
        f"Tokens: {profile['total_tokens']:,} total / {profile['waste_tokens']:,} waste"
    )
    print(
        f"Cost: ${profile['total_cost_usd']:.2f} total / ${profile['waste_cost_usd']:.2f} avoidable"
    )
    print(f"Main waste source: {profile['main_waste_source'] or 'none detected'}")
