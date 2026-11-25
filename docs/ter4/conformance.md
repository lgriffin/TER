# The adapter conformance kit

`python -m ter conformance` checks an adapter that someone outside TER wrote
against its port's contract. It is Stage U3 of ADR 0008 (decision 7, point
P211, TER-CNF-001 to TER-CNF-004).

The kit is not a copy of TER's contract suites. It **is** those suites.
`tests/contract/test_session_source.py` and `tests/contract/test_price_book.py`
run the kit's obligations against every built-in adapter and fake, so the
kit checks an outside adapter against exactly what TER holds its own
adapters to.

## Run it

Name the adapter in one of two ways: by an entry-point value
(`module:attribute`), or by the name it registers as a TER capability. The
kit loads it and builds it. If the target is a class or factory, the kit
calls it with any `--arg` values, in order.

```bash
# A session source: give it sessions it should read.
python -m ter conformance SessionSource --adapter my_harness.ter:MySessionSource \
    --sample runs/one.log --sample runs/two.log

# A price book that reads a file: pass the file as --arg.
python -m ter conformance PriceBook --adapter focus-sku-price --arg sku_price.csv
```

The exit status tells you the outcome:

- **0**: every obligation is met.
- **1**: at least one obligation is not met. Each obligation that fails is
  listed with how it fails, and every obligation is still checked
  (TER-CNF-003).
- **2**: the adapter could not be loaded or built. The kit says why and
  checks nothing (TER-CNF-004).

```text
SessionSource otel-genai conformance
  PASS  protocol                     fits the SessionSource port
  FAIL  deterministic                the same input gives an identical trace
        raised SessionFormatError: sku_price.csv holds no resourceSpans
  ...
1 of 7 obligations met
```

## Obligations

### SessionSource (TER-CNF-001)

| Id | Obligation |
|---|---|
| `protocol` | has `format_name` (not empty) and `read(ref)` |
| `deterministic` | reading the same sample twice gives an identical trace (TER-SRC-004) |
| `ids` | event ids are unique, sequences run 0..n-1, and the trace names the adapter's `format_name` |
| `chain` | every event belongs to the trace's session, the first has no parent, and each later one follows the one before |
| `generated` | the generated events are the assistant's, never a prompt or a tool's output (TER-ANL-001) |
| `missing` | a missing file raises `FileNotFoundError` |
| `refuses` | an empty file, plain text and bytes that are not UTF-8, each with the samples' suffixes, raise `SessionFormatError` (TER-TST-007) |

### PriceBook (TER-CNF-002)

| Id | Obligation |
|---|---|
| `protocol` | has `name` (not empty), `models()` (not empty) and `rate(model, at)` |
| `valid` | every listed model's latest rates are `Rates` (each price finite and not negative) |
| `deterministic` | the same lookup gives the same rates |
| `dated` | `rate(model, today)` gives `Rates` or raises `UnknownModelError` (`--today` to choose the day) |
| `unknown` | an unknown model raises `UnknownModelError` |
| `before` | the earliest date there is (`date.min`) raises `UnknownModelError` for every model |

## What it found

Its first run found a bug in TER's own GARE reader. A `.jsonl` file of
bytes that are not UTF-8 raised `UnicodeDecodeError` instead of
`SessionFormatError`. The fix reads rows split at newlines only, and the
input is now pinned in `tests/resilience/test_regressions.py`.

## Not yet covered

The `OutcomeSource` obligations come next. Checking them needs runs whose
results are already known, so the kit will need a way for an author to
supply that expected evidence. The obligations listed here are the ones
that apply to any adapter. Each format's own refusals, such as a torn span
export or a folder that is not a GARE run, stay in that format's tests.
