# TER v09 — Fine-Grained Span Segmentation

TER v09 adds opt-in, provenance-preserving segmentation for assistant reasoning and generated responses.

## Why

A single content block can mix useful analysis, repetition, corrections, plans, and summaries. Block-level classification forces one label onto all of that content. Fine segmentation creates smaller defensible units before embedding and classification.

## CLI

```bash
ter analyze tests/fixtures/sample_session.jsonl --fine-segmentation
```

Optional bounds:

```bash
ter analyze tests/fixtures/sample_session.jsonl \
  --fine-segmentation \
  --segment-min-tokens 12 \
  --segment-max-tokens 180
```

The feature is opt-in in v09 so existing reports and thresholds remain comparable.

## Boundaries

Candidate boundaries include:

- Paragraph breaks
- Markdown headings and horizontal rules
- Sentence groups for oversized units
- Discourse transitions such as `now`, `next`, `again`, `in summary`, `however`, and `finally`

Small adjacent fragments are merged up to the configured maximum. Tool calls and tool results remain atomic so their structured fingerprints are not weakened.

## Provenance

Every emitted `TokenSpan` now carries:

- `parent_block_id`
- `segment_index`
- `char_start`
- `char_end`
- Existing source message UUID, role, phase, and block type

The original source text can therefore be reconstructed or highlighted exactly.

## Compatibility

Calling `segment_spans(session)` retains legacy one-block-per-span behavior. Fine segmentation is enabled explicitly through `SegmentationConfig` or the CLI flag.

## Validation

Focused loader, classifier, and segmentation regressions pass. Production thresholds are unchanged. Real benchmark comparison between block-level and segment-level classification remains pending annotated data.
