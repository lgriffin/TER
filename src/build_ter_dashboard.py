#!/usr/bin/env python3

import argparse
import html
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if isinstance(value, dict):
                results.append(value)

    if not results:
        raise ValueError(f"No JSON objects found in {path}")

    return results


def number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default

    if isinstance(value, (int, float)):
        return float(value)

    return default


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered))
    weight = position - lower

    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_figure_html(
    figure: go.Figure,
    div_id: str,
) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


def make_dashboard(results: list[dict[str, Any]]) -> str:
    sessions = len(results)

    total_tokens = sum(number(item.get("total_tokens")) for item in results)
    aligned_tokens = sum(number(item.get("aligned_tokens")) for item in results)
    waste_tokens = sum(number(item.get("waste_tokens")) for item in results)

    ter_values = [
        number(item.get("aggregate_ter"))
        for item in results
        if isinstance(item.get("aggregate_ter"), (int, float))
    ]

    average_ter = statistics.mean(ter_values) if ter_values else 0
    median_ter = statistics.median(ter_values) if ter_values else 0

    weighted_ter = (
        aligned_tokens / total_tokens
        if total_tokens
        else 0
    )

    waste_sessions = sum(
        1 for item in results
        if number(item.get("waste_tokens")) > 0
    )

    # TER distribution
    ter_bins = [
        ("< 0.50", 0.0, 0.5),
        ("0.50–0.74", 0.5, 0.75),
        ("0.75–0.89", 0.75, 0.9),
        ("0.90–0.99", 0.9, 1.0),
        ("1.00", 1.0, 1.0000001),
    ]

#    ter_bins = []
#
#    step = 0.05
#
#    for index in range(20):
#        minimum = index * step
#        maximum = minimum + step
#
#        label = f"{minimum:.2f}–{maximum:.2f}"
#
#        ter_bins.append((label, minimum, maximum))
#
#    # Keep a distinct bucket for perfect scores
#    ter_bins.append(("1.00", 1.0, 1.0000001))

    ter_distribution = []
    for label, minimum, maximum in ter_bins:
        count = sum(
            minimum <= value < maximum
            for value in ter_values
        )
        ter_distribution.append((label, count))

    ter_figure = go.Figure(
        data=[
            go.Bar(
                x=[label for label, _ in ter_distribution],
                y=[count for _, count in ter_distribution],
                hovertemplate="%{x}: %{y} sessions<extra></extra>",
            )
        ]
    )
    ter_figure.update_layout(
        title="TER score distribution",
        xaxis_title="Aggregate TER",
        yaxis_title="Sessions",
        margin=dict(l=60, r=30, t=60, b=60),
        height=420,
    )

    # Waste category aggregation
    category_tokens: Counter[str] = Counter()
    category_sessions: Counter[str] = Counter()

    for item in results:
        waste_summary = item.get("waste_summary") or {}
        categories = waste_summary.get("waste_by_category") or {}

        if not isinstance(categories, dict):
            continue

        for category, value in categories.items():
            numeric_value = number(value)
            category_tokens[str(category)] += numeric_value

            if numeric_value > 0:
                category_sessions[str(category)] += 1

    category_items = category_tokens.most_common(15)

    if category_items:
        category_figure = go.Figure(
            data=[
                go.Bar(
                    x=[value for _, value in reversed(category_items)],
                    y=[name for name, _ in reversed(category_items)],
                    orientation="h",
                    customdata=[
                        category_sessions[name]
                        for name, _ in reversed(category_items)
                    ],
                    hovertemplate=(
                        "%{y}<br>"
                        "Waste tokens: %{x:,.0f}<br>"
                        "Affected sessions: %{customdata}"
                        "<extra></extra>"
                    ),
                )
            ]
        )
        category_figure.update_layout(
            title="Top waste categories",
            xaxis_title="Waste tokens",
            yaxis_title="",
            margin=dict(l=180, r=30, t=60, b=60),
            height=max(420, 40 * len(category_items)),
        )
        category_html = build_figure_html(
            category_figure,
            "waste-category-chart",
        )
    else:
        category_html = """
        <div class="empty-state">
            No waste categories were detected in the analyzed sessions.
        </div>
        """

    # Waste by phase
    phase_waste: Counter[str] = Counter()

    for item in results:
        waste_summary = item.get("waste_summary") or {}
        phases = waste_summary.get("waste_by_phase") or {}

        if isinstance(phases, dict):
            for phase, value in phases.items():
                phase_waste[str(phase)] += number(value)

    if phase_waste:
        phase_waste_figure = go.Figure(
            data=[
                go.Pie(
                    labels=list(phase_waste.keys()),
                    values=list(phase_waste.values()),
                    hole=0.45,
                    hovertemplate=(
                        "%{label}<br>"
                        "%{value:,.0f} tokens<br>"
                        "%{percent}<extra></extra>"
                    ),
                )
            ]
        )
        phase_waste_figure.update_layout(
            title="Waste by phase",
            margin=dict(l=30, r=30, t=60, b=30),
            height=420,
        )
        phase_waste_html = build_figure_html(
            phase_waste_figure,
            "waste-phase-chart",
        )
    else:
        phase_waste_html = """
        <div class="empty-state">
            No phase-level waste was detected.
        </div>
        """

    # Average phase scores
    phase_names = ["reasoning", "tool_use", "generation"]
    phase_averages: dict[str, float] = {}

    for phase_name in phase_names:
        values = []

        for item in results:
            phase_scores = item.get("phase_scores") or {}
            value = phase_scores.get(phase_name)

            if isinstance(value, (int, float)):
                values.append(float(value))

        phase_averages[phase_name] = (
            statistics.mean(values)
            if values
            else 0
        )

    phase_score_figure = go.Figure(
        data=[
            go.Bar(
                x=[
                    "Reasoning",
                    "Tool use",
                    "Generation",
                ],
                y=[
                    phase_averages["reasoning"],
                    phase_averages["tool_use"],
                    phase_averages["generation"],
                ],
                text=[
                    f"{phase_averages['reasoning']:.3f}",
                    f"{phase_averages['tool_use']:.3f}",
                    f"{phase_averages['generation']:.3f}",
                ],
                textposition="auto",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        ]
    )
    phase_score_figure.update_layout(
        title="Average phase scores",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        margin=dict(l=60, r=30, t=60, b=60),
        height=420,
    )

    # Token volume vs waste scatter plot
    scatter_x = []
    scatter_y = []
    scatter_text = []
    scatter_ter = []

    for item in results:
        session_id = str(item.get("session_id", "unknown"))
        tokens = number(item.get("total_tokens"))
        waste = number(item.get("waste_tokens"))
        ter = number(item.get("aggregate_ter"))

        scatter_x.append(tokens)
        scatter_y.append(waste)
        scatter_ter.append(ter)
        scatter_text.append(
            f"Session: {html.escape(session_id)}"
        )

    scatter_figure = go.Figure(
        data=[
            go.Scatter(
                x=scatter_x,
                y=scatter_y,
                mode="markers",
                text=scatter_text,
                customdata=scatter_ter,
                hovertemplate=(
                    "%{text}<br>"
                    "Total tokens: %{x:,.0f}<br>"
                    "Waste tokens: %{y:,.0f}<br>"
                    "TER: %{customdata:.4f}"
                    "<extra></extra>"
                ),
            )
        ]
    )
    scatter_figure.update_layout(
        title="Session size versus detected waste",
        xaxis_title="Total tokens",
        yaxis_title="Waste tokens",
        margin=dict(l=70, r=30, t=60, b=60),
        height=500,
    )

    # Worst sessions table
    ranked_results = sorted(
        results,
        key=lambda item: (
            number(item.get("waste_tokens")),
            -number(item.get("aggregate_ter"), 1),
        ),
        reverse=True,
    )

    worst_sessions = ranked_results[:100]

    rows = []
    for item in worst_sessions:
        session_id = html.escape(
            str(item.get("session_id", "unknown"))
        )
        ter = number(item.get("aggregate_ter"))
        total = int(number(item.get("total_tokens")))
        waste = int(number(item.get("waste_tokens")))

        phase_scores = item.get("phase_scores") or {}

        explanation = html.escape(
            str(
                (item.get("waste_summary") or {}).get(
                    "explanation",
                    "",
                )
            )
        )

        rows.append(
            f"""
            <tr>
                <td><code>{session_id}</code></td>
                <td data-sort="{ter}">{ter:.4f}</td>
                <td data-sort="{total}">{total:,}</td>
                <td data-sort="{waste}">{waste:,}</td>
                <td>{number(phase_scores.get("reasoning")):.3f}</td>
                <td>{number(phase_scores.get("tool_use")):.3f}</td>
                <td>{number(phase_scores.get("generation")):.3f}</td>
                <td class="explanation">{explanation}</td>
            </tr>
            """
        )

    perfect_sessions = sum(value == 1.0 for value in ter_values)

    summary_text = (
        f"{perfect_sessions:,} of {sessions:,} sessions "
        f"received a perfect TER score."
    )

    plotly_js = get_plotlyjs()

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >
    <title>TER analysis dashboard</title>

    <script>{plotly_js}</script>

    <style>
        :root {{
            color-scheme: light dark;
            --background: #f4f6f8;
            --surface: #ffffff;
            --text: #17202a;
            --muted: #65717e;
            --border: #dfe4ea;
            --accent: #315efb;
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --background: #11151a;
                --surface: #1a2027;
                --text: #edf2f7;
                --muted: #a7b0ba;
                --border: #313a45;
                --accent: #7d9bff;
            }}
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background: var(--background);
            color: var(--text);
            font-family:
                Inter,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }}

        main {{
            width: min(1500px, calc(100% - 32px));
            margin: 32px auto 80px;
        }}

        header {{
            margin-bottom: 28px;
        }}

        h1 {{
            margin-bottom: 8px;
            font-size: clamp(2rem, 4vw, 3.2rem);
        }}

        .subtitle {{
            color: var(--muted);
            font-size: 1.05rem;
        }}

        .cards {{
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .card,
        .panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 4px 18px rgb(0 0 0 / 5%);
        }}

        .card {{
            padding: 20px;
        }}

        .card-label {{
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 8px;
        }}

        .card-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}

        .grid {{
            display: grid;
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
            gap: 20px;
        }}

        .panel {{
            padding: 12px;
            overflow: hidden;
        }}

        .panel.full {{
            grid-column: 1 / -1;
        }}

        .empty-state {{
            min-height: 420px;
            display: grid;
            place-items: center;
            padding: 32px;
            color: var(--muted);
            text-align: center;
        }}

        .table-panel {{
            margin-top: 20px;
            padding: 20px;
        }}

        .table-toolbar {{
            display: flex;
            gap: 12px;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}

        input {{
            width: min(420px, 100%);
            padding: 10px 12px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text);
        }}

        .table-wrapper {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th,
        td {{
            padding: 11px 10px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }}

        th {{
            position: sticky;
            top: 0;
            background: var(--surface);
            cursor: pointer;
            white-space: nowrap;
        }}

        td.explanation {{
            min-width: 320px;
            max-width: 650px;
        }}

        code {{
            font-size: 0.82rem;
        }}

        footer {{
            margin-top: 28px;
            color: var(--muted);
            font-size: 0.9rem;
        }}

        @media (max-width: 900px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}

            .panel.full {{
                grid-column: auto;
            }}
        }}
    </style>
</head>

<body>
<main>
    <header>
        <h1>TER analysis dashboard</h1>
        <div class="subtitle">
            {html.escape(summary_text)}
        </div>
    </header>

    <section class="cards">
        <div class="card">
            <div class="card-label">Sessions analyzed</div>
            <div class="card-value">{sessions:,}</div>
        </div>

        <div class="card">
            <div class="card-label">Total tokens</div>
            <div class="card-value">{total_tokens:,.0f}</div>
        </div>

        <div class="card">
            <div class="card-label">Waste tokens</div>
            <div class="card-value">{waste_tokens:,.0f}</div>
        </div>

        <div class="card">
            <div class="card-label">Weighted TER</div>
            <div class="card-value">{weighted_ter:.4f}</div>
        </div>

        <div class="card">
            <div class="card-label">Average session TER</div>
            <div class="card-value">{average_ter:.4f}</div>
        </div>

        <div class="card">
            <div class="card-label">Median TER</div>
            <div class="card-value">{median_ter:.4f}</div>
        </div>

        <div class="card">
            <div class="card-label">Sessions with waste</div>
            <div class="card-value">{waste_sessions:,}</div>
        </div>

        <div class="card">
            <div class="card-label">90th percentile TER</div>
            <div class="card-value">
                {percentile(ter_values, 0.90):.4f}
            </div>
        </div>
    </section>

    <section class="grid">
        <div class="panel">
            {build_figure_html(ter_figure, "ter-distribution-chart")}
        </div>

        <div class="panel">
            {build_figure_html(phase_score_figure, "phase-score-chart")}
        </div>

        <div class="panel">
            {category_html}
        </div>

        <div class="panel">
            {phase_waste_html}
        </div>

        <div class="panel full">
            {build_figure_html(scatter_figure, "token-waste-scatter")}
        </div>
    </section>

    <section class="panel table-panel">
        <div class="table-toolbar">
            <div>
                <h2>Top sessions by detected waste</h2>
                <div class="subtitle">
                    Showing up to 100 sessions.
                </div>
            </div>

            <input
                id="session-filter"
                type="search"
                placeholder="Filter by session ID or explanation"
            >
        </div>

        <div class="table-wrapper">
            <table id="results-table">
                <thead>
                    <tr>
                        <th>Session</th>
                        <th>TER</th>
                        <th>Total tokens</th>
                        <th>Waste tokens</th>
                        <th>Reasoning</th>
                        <th>Tool use</th>
                        <th>Generation</th>
                        <th>Explanation</th>
                    </tr>
                </thead>

                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </section>

    <footer>
        Generated locally from the TER JSONL output.
        The report contains session identifiers and analysis details,
        so treat the HTML file as confidential.
    </footer>
</main>

<script>
    const filter = document.getElementById("session-filter");
    const rows = Array.from(
        document.querySelectorAll("#results-table tbody tr")
    );

    filter.addEventListener("input", () => {{
        const query = filter.value.toLowerCase().trim();

        for (const row of rows) {{
            row.hidden = !row.textContent.toLowerCase().includes(query);
        }}
    }});

    const headers = document.querySelectorAll(
        "#results-table thead th"
    );

    headers.forEach((header, columnIndex) => {{
        let ascending = true;

        header.addEventListener("click", () => {{
            const tbody = document.querySelector(
                "#results-table tbody"
            );

            const sortedRows = [...tbody.rows].sort((a, b) => {{
                const aCell = a.cells[columnIndex];
                const bCell = b.cells[columnIndex];

                const aValue =
                    aCell.dataset.sort ?? aCell.textContent.trim();
                const bValue =
                    bCell.dataset.sort ?? bCell.textContent.trim();

                const aNumber = Number(aValue.replaceAll(",", ""));
                const bNumber = Number(bValue.replaceAll(",", ""));

                let comparison;

                if (
                    Number.isFinite(aNumber) &&
                    Number.isFinite(bNumber)
                ) {{
                    comparison = aNumber - bNumber;
                }} else {{
                    comparison = aValue.localeCompare(bValue);
                }}

                return ascending ? comparison : -comparison;
            }});

            sortedRows.forEach(row => tbody.appendChild(row));
            ascending = !ascending;
        }});
    }});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an HTML dashboard from TER JSONL results."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to all-results.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ter-dashboard.html"),
        help="Output HTML path",
    )

    args = parser.parse_args()

    results = load_jsonl(args.input)
    dashboard = make_dashboard(results)

    args.output.write_text(dashboard, encoding="utf-8")

    print(f"Dashboard written to: {args.output.resolve()}")
    print(f"Sessions included: {len(results)}")


if __name__ == "__main__":
    main()
