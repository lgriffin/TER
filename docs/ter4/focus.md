# FOCUS: prices in, efficiency out

[FOCUS](https://focus.finops.org/) is the FinOps Foundation's open format for
cost and usage data. TER meets it at two points (Stage U2,
[ADR 0008](../decisions/0008-upstream-adapters.md)): a provider's FOCUS price
list can price a session, and TER Records can be joined to FOCUS billing rows.
TER reads and writes FOCUS files by a pinned version and never imports a FOCUS
library.

## Prices from a FOCUS SKU Price file

FOCUS 1.5 (ratification 3 Dec 2026) adds a SKU Price dataset, published as a
preview: one row per unit price, with the price's type, currency, pricing unit
and effective dates. `PriceBook.focus-sku-price`
(`ter.adapters.driven.pricing.focus`) reads such a file, as CSV, by the 1.5
column contract:

```bash
python -m ter record export session.jsonl --prices sku_price.csv --out r.json
python -m ter record verify r.json session.jsonl --prices sku_price.csv
```

A record names its price book (`cost.price_book`, here `focus-sku-price-1.5`),
so verify it with the same `--prices` file it was exported with.

### What TER reads

| Column | Used for |
|---|---|
| `UnitPriceType`, `ContractId`, `PricingCurrency` | only public list prices (`List`, no contract) in `USD` are read; TER's rates are USD list prices (ADR 0003) |
| `PricingUnit`, `UnitPrice` | the per-million-token rate: `Tokens`, `1000 Tokens`, `1K Tokens`, `1M Tokens`, `1,000,000 Tokens` and `Million Tokens` are understood |
| `SkuPriceEffectiveStart` | the date the rate takes effect; empty means from the start |
| `SkuPriceEffectiveEnd` | the first day the rate is no longer in effect; empty means until replaced |
| `SkuPriceId` | where the rate came from (each rate's source names its price ids) |
| `SkuPriceDetails` | what the row prices: `ModelId`, `TokenDirection`, `TokenCacheAction` |

Each rate stays in force until a later row of the same model and kind
replaces it, or its end date passes. A date that restates only some rates
(a new output price, say) keeps the others in force. A model is priced on a
date only while all four rates are in force: from a day when one has ended
with no successor, its sessions are unpriced, never priced from an expired
rate, and the gap is listed in `incomplete`. Two different list prices for
one model, kind and start date are refused, naming their SKU price ids, so
the order of rows never changes a rate. A session's dated model
(`claude-sonnet-5-5-20261001`) is priced as the model the file names
(`claude-sonnet-5-5`) when the file does not price the dated name itself.

A file missing any of `ServiceProviderName`, `SkuId`, `SkuPriceId`,
`ContractId`, `PricingCurrency`, `PricingUnit`, `UnitPrice`, `UnitPriceType`,
`SkuPriceEffectiveStart` or `SkuPriceEffectiveEnd` is refused, naming the
missing columns (TER-ANL-046).

### Token labels

FOCUS 1.5 recommends two SKU Price Details properties for token prices,
`TokenDirection` (`Input`, `Output`) and `TokenCacheAction` (`Uncached`,
`Read`, `Write`, `Other`), and `ModelId` for the model as billing names it.
They map to TER's four rates:

| SKU Price Details | TER rate |
|---|---|
| `Input`, `Uncached` | `input` |
| `Input`, `Read` | `cache_read` |
| `Input`, `Write` | `cache_write` |
| `Output` (reasoning tokens metered separately are also `Output`) | `output` |

The 1.5 SKU Price dataset defers its own SKU Price Details column to a later
release; in 1.5 the column is on Cost and Usage rows. So TER takes the labels
from the price row when it carries a `SkuPriceDetails` column, and otherwise
from the provider's FOCUS Cost and Usage file, matched by `SkuPriceId`:

```bash
python -m ter record export session.jsonl --prices sku_price.csv \
  --price-details cost_and_usage.csv
```

### What TER leaves unpriced

Nothing is guessed. A row that is not a public USD list price, has no labels,
bills several token kinds on one meter (no `TokenDirection`, or
`TokenCacheAction` `Other`), has a unit that is not tokens, or a price or date
that does not parse is skipped and counted. A model whose rows lack one of the
four rates on a date is not priced from that date (TER-ANL-047); an earlier
complete price stays in force, and a session turn with no price is counted
unpriced, as with the shipped book.

## Records joined to FOCUS billing rows

FOCUS 1.5 sets the session aside: a Cost and Usage row says which model was
used, by whom and in which charge period, never in which session. So TER
joins many to one. `python -m ter record focus` reads the provider's Cost and
Usage file and any number of Records, and writes the billing file back with
efficiency columns added:

```bash
python -m ter record export a.jsonl --out a.record.json
python -m ter record export b.jsonl --out b.record.json
python -m ter record focus a.record.json b.record.json \
  --billing cost_and_usage.csv --out joined.csv
```

It says on stderr how many Records joined how many rows, and why the rest did
not.

### The join

A Record joins every row of a model it used whose charge period overlaps
the Record's day (`join.charge_period`, the session's date, in UTC). Its usage of that
model (`usage.by_model`) gives its share of the row:

- A row whose SKU Price Details name a token kind (`TokenDirection` and
  `TokenCacheAction`, as for the price book) takes the Record's tokens of
  that kind. A row with no kind takes all of them. A row of `Other` tokens,
  or whose SKU Price Details cannot be read, is not joined.
- The row's model is its `ModelId`, else its `SkuId`. A dated model
  (`claude-sonnet-5-5-20261001`) joins rows billed under its own name when
  the day has any, else rows billed under its alias (`claude-sonnet-5-5`),
  never both, and nothing else matches.
- The row's tokens are `ConsumedQuantity` in its `ConsumedUnit`; a row whose
  unit is not tokens, or whose period does not parse, is written unchanged
  and not joined.
- Records carry no principal (ADR 0007) and no hour, so all the rows of one
  model and kind that overlap the day share a Record's tokens in proportion
  to their consumed tokens: rows that differ by `PrincipalId`, and the
  hourly rows of a file with sub-day charge periods. A token is never
  counted in two rows.
- A row with more cells than the header is refused (writing it back would
  drop the extra cell), and so is a file that cannot be read.

Every FOCUS column and every row is written as it was read. The exporter
only adds columns, and every one carries the `x_` prefix FOCUS reserves for
custom columns (TER-EXP-011). A billing file that already has them is
refused, so a joined file is never joined twice.

| Column | What it holds |
|---|---|
| `x_TerRecordCount` | Records joined to the row |
| `x_TerRecordTokens` | their tokens of the row's kind (shared by principal as above) |
| `x_TerTokenShare` | those tokens over the row's consumed tokens |
| `x_TerValueAddingCostShare`, `x_TerNecessaryCostShare`, `x_TerAvoidableCostShare`, `x_TerUncertainCostShare` | the joined Records' priced cost by activity class (whole-turn allocation, ADR 0007), as shares of their cost |
| `x_TerVerifiedOutcomes` | joined Records with an accepted outcome |

A row no Record joined keeps these columns empty. Uncertain cost stays in its
own column and is never added to avoidable (ADR 0006).

### What it does not claim

The cost shares are the Records' own, not the row's: they say what the
sessions behind a row did, and `x_TerTokenShare` says how much of the row
those sessions are. A row with a low token share is mostly sessions TER has
not seen. Whether a join on model and day is fine enough for a FinOps team is
the document's open question (ADR 0008).

## Requirements

TER-ANL-045 to TER-ANL-047 (the price book) and TER-EXP-011 to TER-EXP-013
(the exporter); vision points P208 and P209.
