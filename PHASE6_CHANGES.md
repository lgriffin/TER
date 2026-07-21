# Phase 6 — Adaptive Optimization

Phase 6 turns aggregate TER history into bounded, project-specific operating policy without persisting raw prompts.

## Capabilities

- Learns similarity, confidence, and restatement thresholds from historical TER and waste ratios.
- Rebalances reasoning, tool-use, and generation weights toward historically weak phases.
- Recommends soft, target, and hard token budgets using robust historical quantiles.
- Tunes intervention thresholds when repetition, retries, or loop waste dominate.
- Optionally personalizes token budgets from privacy-preserving prompt-neighbor predictions.
- Exports policies atomically as portable JSON for CI, hooks, and team review.

## CLI

```bash
ter optimize --project my-project --format json
ter optimize --project my-project --output .ter/adaptive-policy.json
ter optimize --project my-project --prompt "refactor parser tests" --neighbors 8
```

Use `--minimum-samples` to control when a policy becomes usable. Policies remain marked `insufficient`, `experimental`, `stable`, or `mature` according to sample size.

## Privacy and safety

The optimizer uses only aggregate session metrics and hashed prompt fingerprints already stored by Phase 4. All learned thresholds are bounded to conservative ranges, and exported files contain no raw session or prompt content.
