# TER guides

Practical guides for using, testing and extending TER. They show real commands
and real code from this repository, and `tests/docs` keeps them honest: every
relative link must resolve and every `ter`, `ter-req` and `python -m ter`
command shown in a shell block must exist with the options it uses.

| Guide | Read it when you want to… |
|---|---|
| [Architecture](architecture.md) | understand the hexagon, the strangler fig over TER 3, the ports and the L0 to L6 maturity levels (L0 to L4 met) |
| [Testing](testing.md) | run the suite, understand each test layer and gate, or write a new test end to end |
| [Lean](lean.md) | learn how TER models an agent session as a value stream and what each waste detector looks for |
| [The per-run report and the A3](a3-report.md) | read or produce the HTML report and the A3, and apply its countermeasures |
| [Hooks](hooks.md) | capture live sessions with Claude Code hooks, turn on live advice (L4), or install the hooks the A3 recommends |
| [Claude Code plugin](claude-plugin.md) (exploratory) | let Claude run TER for you: analyse a session, apply the A3's countermeasures, turn on hooks |
| [EARS requirements](ears.md) | write a requirement, tag a test with it, and pass the trace gates |
| [Vision points and definition of done](definition-of-done.md) | find a point, decide whether it is done, and record it honestly |
| [Contributing](contributing.md) | set up, work through a change and push it past every gate |

## Where the reference material lives

The guides explain and link; they do not duplicate the reference pages.

| Reference | Holds |
|---|---|
| [docs/ter4/strategy.md](../ter4/strategy.md) | The TER 4 strategy: maturity model, where TER is (L0 to L4 met), what real data showed, and the L5 and L6 roadmap |
| [docs/ter4/architecture.md](../ter4/architecture.md) | The TER 4 hexagon, ports and adapters, import contracts, event contract and the gate tables per level |
| [docs/ter4/l1-observed.md](../ter4/l1-observed.md) | The L1 event stream, hook-to-event mapping and `StreamReport` fields |
| [docs/ter4/l2-explained.md](../ter4/l2-explained.md) | The L2 Lean model, detector catalogue, scorecard, evidence graph and A3 |
| [docs/ter4/l3-grounded.md](../ter4/l3-grounded.md) | L3 Grounded: the `RepositoryEvidence` port and its lexical, Git, Python and multi-language syntax engines, change surface, architecture contracts, context bundles, evidence usage, routing, languages and stack |
| [docs/ter4/l6-stack-comparison.md](../ter4/l6-stack-comparison.md) | Token use and waste compared across languages and stacks, and the first real run |
| [docs/ter4/reports.md](../ter4/reports.md) | The visual report layer and `ter report --html` |
| [docs/ter4/requirements.md](../ter4/requirements.md) | The EARS catalogue schema and the `ter-req` controls |
| [docs/ter4/points.md](../ter4/points.md) | The generated index of the 200 vision points |
| [docs/decisions/](../decisions/) | Architecture decision records (ADRs 0001 to 0005) |
| [docs/user-guide.md](../user-guide.md), [docs/architecture.md](../architecture.md), [docs/context-orchestrator.md](../context-orchestrator.md) | TER 3 reference: every `ter` command, the TER 3 pipeline, the context orchestrator |
| [tests/golden/README.md](../../tests/golden/README.md) | The golden corpus and what each synthetic session exercises |
