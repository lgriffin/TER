"""Command implementation module extracted from :mod:`ter_calculator.cli`."""

from __future__ import annotations

import sys


def _resolve_session_path(args) -> str | None:
    """Resolve --latest flag to a concrete session path."""
    if getattr(args, "latest", False):
        from ..loader import find_latest_session

        return str(find_latest_session(getattr(args, "session_path", None)))
    return getattr(args, "session_path", None)


def _cmd_context(args) -> int:
    """Dispatch context sub-subcommands."""
    ctx_cmd = getattr(args, "context_command", None)
    if not ctx_cmd:
        print("Usage: ter context {store|graph|optimize|delta|check}", file=sys.stderr)
        return 1

    dispatch = {
        "store": _cmd_context_store,
        "graph": _cmd_context_graph,
        "optimize": _cmd_context_optimize,
        "delta": _cmd_context_delta,
        "check": _cmd_context_check,
    }
    handler = dispatch.get(ctx_cmd)
    if handler is None:
        print(f"Unknown context command: {ctx_cmd}", file=sys.stderr)
        return 1
    return handler(args)


def _cmd_context_store(args) -> int:
    """Shard a session into content-addressable fragments."""
    session_path = _resolve_session_path(args)
    if session_path is None:
        print("Error: Provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..loader import load_session, segment_spans
    from ..fragment_store import FragmentStore, FragmentShardingEngine

    session = load_session(session_path)
    spans = segment_spans(session)
    store = FragmentStore()
    try:
        engine = FragmentShardingEngine(store)
        fragments = engine.shard(spans, session.session_id)

        new_count = len(
            [f for f in fragments if f.origin_session == session.session_id]
        )
        print(f"Session: {session.session_id}")
        print(f"Spans processed: {len(spans)}")
        print(
            f"Fragments created: {new_count} new, {len(fragments) - new_count} existing"
        )
        print(f"Total in store: {store.count()}")
    finally:
        store.close()
    return 0


def _cmd_context_graph(args) -> int:
    """Build and display the context graph for a session."""
    import json as json_mod

    session_path = _resolve_session_path(args)
    if session_path is None:
        print("Error: Provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..loader import load_session, segment_spans
    from ..fragment_store import FragmentStore, FragmentShardingEngine
    from ..context_graph import ContextGraph

    session = load_session(session_path)
    spans = segment_spans(session)
    store = FragmentStore()
    graph = ContextGraph()
    try:
        engine = FragmentShardingEngine(store)
        fragments = engine.shard(spans, session.session_id)

        graph.build_from_session(fragments)

        fmt = getattr(args, "output_format", "text")
        if fmt == "json":
            edges = [
                {
                    "source": e.source_id[:12],
                    "target": e.target_id[:12],
                    "type": e.edge_type.value,
                    "weight": round(e.weight, 4),
                }
                for e in graph.all_edges()
            ]
            print(
                json_mod.dumps(
                    {
                        "nodes": graph.node_count,
                        "edges": graph.edge_count,
                        "edge_list": edges,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Context Graph for {session.session_id}")
            print(f"  Nodes: {graph.node_count}")
            print(f"  Edges: {graph.edge_count}")
            cycles = graph.detect_cycles()
            if cycles:
                print(f"  Cycles detected: {len(cycles)}")
            else:
                print("  DAG: valid (no cycles)")
            topo = graph.topological_sort()
            print(f"  Topological order: {len(topo)} nodes")
            print("\nEdge breakdown:")
            from collections import Counter

            type_counts = Counter(e.edge_type.value for e in graph.all_edges())
            for etype, count in type_counts.most_common():
                print(f"  {etype}: {count}")
    finally:
        store.close()
        graph.close()
    return 0


def _cmd_context_optimize(args) -> int:
    """Run knapsack optimization on session fragments."""
    session_path = _resolve_session_path(args)
    if session_path is None:
        print("Error: Provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..loader import load_session, segment_spans
    from ..intent import extract_intent
    from ..fragment_store import FragmentStore, FragmentShardingEngine
    from ..context_graph import ContextGraph
    from ..budget_optimizer import recommend_context

    session = load_session(session_path)
    spans = segment_spans(session)
    intent = extract_intent(session)
    store = FragmentStore()
    graph = ContextGraph()
    try:
        engine = FragmentShardingEngine(store)
        engine.shard(spans, session.session_id)

        fragments = store.find_by_session(session.session_id)
        graph.build_from_session(fragments)

        result = recommend_context(
            session,
            intent,
            args.budget,
            store,
            graph,
            relevance_threshold=getattr(args, "relevance_threshold", 0.1),
        )

        print(f"Budget Optimization for {session.session_id}")
        print("=" * 50)
        print(f"Budget: {result.budget_ceiling:,} tokens")
        print(f"Selected: {len(result.selected_fragment_ids)} fragments")
        print(f"Tokens used: {result.budget_used:,} / {result.budget_ceiling:,}")
        print(f"Total relevance: {result.total_relevance:.4f}")
        print(f"Pruned (redundant): {result.pruned_count}")
        print(f"\n{result.reasoning}")
    finally:
        store.close()
        graph.close()
    return 0


def _cmd_context_delta(args) -> int:
    """Show delta prompt composition for a session."""
    session_path = _resolve_session_path(args)
    if session_path is None:
        print("Error: Provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..loader import load_session, segment_spans
    from ..fragment_store import FragmentStore, FragmentShardingEngine
    from ..delta_composer import (
        LocalCache,
        compose_delta,
        create_template_from_session,
    )

    session = load_session(session_path)
    spans = segment_spans(session)
    store = FragmentStore()
    try:
        engine = FragmentShardingEngine(store)
        fragments = engine.shard(spans, session.session_id)

        template = create_template_from_session(session, fragments)
        cache = LocalCache()
        delta = compose_delta(template, store, cache)

        print(f"Delta Composition for {session.session_id}")
        print("=" * 50)
        print(f"Template placeholders: {len(template.required_fragment_ids)}")
        print(f"Cache hits: {delta.manifest.cache_hits}")
        print(f"Cache misses: {delta.manifest.cache_misses}")
        print(f"Delta fragments: {len(delta.delta_fragments)}")
        print(f"Tokens saved: {delta.total_tokens_saved:,}")
        print(f"Compression ratio: {delta.compression_ratio:.1%}")
    finally:
        store.close()
    return 0


def _cmd_context_check(args) -> int:
    """Run consistency check across sessions."""
    session_path = _resolve_session_path(args)
    if session_path is None:
        print("Error: Provide a session_path or use --latest", file=sys.stderr)
        return 1

    from ..loader import discover_subagents
    from ..fragment_store import FragmentStore
    from ..context_graph import ContextGraph
    from ..consistency import (
        ConsistencyCoordinator,
        monitor_group_consistency,
    )
    from ..models import ConsistencyMode

    store = FragmentStore()
    graph = ContextGraph()
    try:
        mode = ConsistencyMode(getattr(args, "mode", "relaxed"))

        paths = [session_path]
        if getattr(args, "group", False):
            subagent_paths = discover_subagents(session_path)
            paths.extend(str(p) for p in subagent_paths)

        if len(paths) < 2:
            print("No version skew possible with a single session.")
            return 0

        skews = monitor_group_consistency(paths, store, graph)

        if not skews:
            print("No version skew detected across sessions.")
        else:
            coordinator = ConsistencyCoordinator()
            print(f"Found {len(skews)} version skew(s):\n")
            for skew in skews:
                action = coordinator.resolve_skew(skew, mode)
                print(f"  Fragment: {skew.fragment_id[:16]}...")
                print(f"  Severity: {skew.severity}")
                print(f"  Sessions: {len(skew.sessions_involved)}")
                print(f"  Action: {'BLOCK' if action.block else 'WARN'}")
                print(f"  Message: {action.message}")
                print()
    finally:
        store.close()
        graph.close()
    return 0
