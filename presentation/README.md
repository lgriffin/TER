# Presentation

[ter-overview.md](ter-overview.md) is a slide deck of about thirty slides
giving a high-level overview of TER: its intent, the Lean framing, the
engineering practices behind TER 4, the per-run A3, waste detection, L3
Grounded (repository evidence, change surface, context bundles, advisory
routing), what real sessions taught us, the GARE integration and the road
to L4 and beyond. Speaker notes sit under
each slide as HTML comments.

It is a [Marp](https://marp.app/) deck: plain Markdown with Marp front
matter. The theme is Marp's `default` with a `style:` block in the front
matter, the same convention `ter present` uses for generated decks, so it
needs no theme file or HTML switch.

## Render it

```bash
npx @marp-team/marp-cli@4.1.2 presentation/ter-overview.md --allow-local-files -o ter-overview.html
npx @marp-team/marp-cli@4.1.2 presentation/ter-overview.md --allow-local-files --pdf -o ter-overview.pdf
npx @marp-team/marp-cli@4.1.2 presentation/ter-overview.md --allow-local-files --pptx -o ter-overview.pptx
npx @marp-team/marp-cli@4.1.2 -s presentation                # live preview server
```

PDF, PPTX and image output need Chrome, Chromium or Edge; set `CHROME_PATH`
if Marp cannot find one. `--allow-local-files` lets the browser read the
images in `img/`. In VS Code, the
[Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)
extension previews and exports the deck with no extra settings.

## Images

| File | What | Source |
|---|---|---|
| `img/value-stream.svg` | The agentic value stream | Hand-drawn from [the Lean guide](../docs/guides/lean.md) |
| `img/hexagon.svg` | The TER 4 hexagon | Hand-drawn from [the architecture](../docs/ter4/architecture.md) |
| `img/maturity.svg` | Maturity levels L0 to L6 and their status | Hand-drawn from the README's level table |

The L3 terminal excerpts are real output of `python -m ter explain` and
`python -m ter route` with `--repo` on the synthetic session and shop
repository in `tests/unit/test_ter4_grounded_cli.py`.
| `img/a3-*.png` | Crops of a real A3 page | `ter a3 tests/golden/sessions/lean_mix.jsonl --html a3.html`, screenshotted at 1600 px wide |

Colours follow the report palette in
`src/ter/adapters/driving/reports/palette.py`.

## Keeping it accurate

The deck only claims what is on `main`, and labels everything else planned.
When the numbers it quotes move (requirements verified per level, points
done, the detector count), update the slides with them: the README's level
table, [docs/ter4/strategy.md](../docs/ter4/strategy.md) and
[docs/ter4/points.md](../docs/ter4/points.md) are the sources.
`tests/docs` checks the deck's relative links and that its shell examples
name real `ter` commands and options.
