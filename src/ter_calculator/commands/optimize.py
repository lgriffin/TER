"""Phase 6 adaptive optimization command."""

from __future__ import annotations

import json

from ..adaptive_optimizer import learn_policy, personalize_policy, save_policy
from ..history_store import TERHistoryStore


def _cmd_optimize(args) -> int:
    store = TERHistoryStore(args.db)
    try:
        records = store.query(project=args.project, limit=100000)
        policy = learn_policy(
            records,
            args.project,
            minimum_samples=args.minimum_samples,
        )
        if args.prompt:
            prediction = store.predict(args.prompt, args.project, k=args.neighbors)
            policy = personalize_policy(policy, prediction)
    finally:
        store.close()

    if args.output:
        path = save_policy(policy, args.output)
        if not args.quiet:
            print(f"Adaptive policy written to {path}")
    if args.output_format == "json" or not args.output:
        if args.output_format == "json":
            print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
        else:
            _print_policy(policy.to_dict())
    return 0


def _print_policy(policy: dict[str, object]) -> None:
    print("TER Adaptive Optimization Policy")
    print("================================")
    print(f"Project: {policy['project']}")
    print(f"Samples: {policy['sample_size']} ({policy['confidence']})")
    evidence = policy["evidence"]
    assert isinstance(evidence, dict)
    print(f"Historical TER: {float(evidence['average_ter']):.3f}")
    print(f"Waste ratio: {float(evidence['waste_ratio']):.1%}")
    thresholds = policy["thresholds"]
    assert isinstance(thresholds, dict)
    print(
        "Thresholds: "
        f"similarity={thresholds['similarity']}, "
        f"confidence={thresholds['confidence']}, "
        f"restatement={thresholds['restatement']}"
    )
    budget = policy["token_budget"]
    assert isinstance(budget, dict)
    print(
        "Token budget: "
        f"soft={int(budget['soft_limit']):,}, "
        f"recommended={int(budget['recommended']):,}, "
        f"hard={int(budget['hard_limit']):,}"
    )
