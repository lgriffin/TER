"""Standalone HTML reporting for TER analysis results."""

from __future__ import annotations

import html
import json
from collections import Counter

from .formatter_json import ter_result_to_dict
from .models import ALIGNED_LABELS, TERResult


def _diagnostics(result: TERResult) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    uncertainty = result.uncertainty
    if uncertainty is not None and uncertainty.low_confidence_share >= 0.10:
        items.append(
            {
                "severity": "high"
                if uncertainty.low_confidence_share >= 0.25
                else "medium",
                "code": "TER-H014",
                "message": (
                    f"{uncertainty.low_confidence_share:.1%} of scored tokens have "
                    "low-confidence classifications. Review the highlighted spans."
                ),
            }
        )
    if (
        uncertainty is not None
        and uncertainty.interval_lower == uncertainty.interval_upper
    ):
        if uncertainty.low_confidence_tokens > 0:
            items.append(
                {
                    "severity": "medium",
                    "code": "TER-H021",
                    "message": "The TER interval has zero width despite low-confidence tokens.",
                }
            )
    low_alignment = [
        item
        for item in result.classified_spans
        if item.span.phase.value == "generation" and item.cosine_similarity < 0.20
    ]
    if low_alignment:
        items.append(
            {
                "severity": "high",
                "code": "TER-H033",
                "message": (
                    f"{len(low_alignment)} generation span(s) have prompt alignment below 0.20."
                ),
            }
        )
    user_spans = [
        item for item in result.classified_spans if item.span.source_role != "assistant"
    ]
    if user_spans:
        items.append(
            {
                "severity": "high",
                "code": "TER-H001",
                "message": "User-origin content appears in scored output spans.",
            }
        )
    if not items:
        items.append(
            {
                "severity": "info",
                "code": "TER-H000",
                "message": "No high-priority consistency warnings were detected.",
            }
        )
    return items


def _report_payload(result: TERResult) -> dict:
    data = ter_result_to_dict(result)
    spans: list[dict[str, object]] = []
    for item in result.classified_spans:
        spans.append(
            {
                "position": item.span.position,
                "phase": item.span.phase.value,
                "label": item.label.value,
                "aligned": item.label in ALIGNED_LABELS,
                "confidence": round(item.confidence, 4),
                "alignment": round(item.cosine_similarity, 4),
                "tokens": item.span.token_count,
                "text": item.span.text,
                "source_role": item.span.source_role,
                "reason": item.explanation.summary
                if item.explanation
                else "No explanation available.",
                "signals": item.explanation.signals if item.explanation else {},
            }
        )
    label_tokens: Counter[str] = Counter()
    phase_tokens: Counter[str] = Counter()
    for item in result.classified_spans:
        label_tokens[item.label.value] += item.span.token_count
        phase_tokens[item.span.phase.value] += item.span.token_count
    data["report"] = {
        "spans": spans,
        "label_tokens": dict(label_tokens),
        "phase_tokens": dict(phase_tokens),
        "diagnostics": _diagnostics(result),
    }
    return data


def format_html(result: TERResult) -> str:
    """Render a portable, dependency-free HTML analysis report."""
    payload = _report_payload(result)
    encoded = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    title = html.escape(f"TER Report — {result.session_id}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--text:#182033;--muted:#667085;--line:#dfe3ec;--good:#19734a;--bad:#b42318;--warn:#b54708;--accent:#3157d5}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10131a;--card:#181d27;--text:#f3f5f8;--muted:#a6adbb;--line:#303744;--good:#58c48d;--bad:#ff7b72;--warn:#f0b35a;--accent:#8ca6ff}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1280px;margin:auto;padding:28px}} header{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:20px}} h1{{margin:0;font-size:28px}} h2{{font-size:17px;margin:0 0 14px}} .muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 2px 8px #0000000a}} .metric{{grid-column:span 2}} .metric b{{display:block;font-size:25px;margin-top:5px}} .wide{{grid-column:span 8}} .side{{grid-column:span 4}} .full{{grid-column:1/-1}}
.chart{{min-height:220px}} svg{{width:100%;height:auto;overflow:visible}} .legend{{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}} .legend span::before{{content:'';display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--swatch);margin-right:5px}}
.timeline{{display:flex;gap:3px;height:54px;align-items:stretch;overflow:auto;padding-bottom:6px}} .segment{{min-width:9px;border:0;border-radius:5px;cursor:pointer;opacity:.9}} .segment:hover,.segment.active{{outline:3px solid var(--accent);opacity:1}}
.table-wrap{{overflow:auto;max-height:520px}} table{{border-collapse:collapse;width:100%}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:var(--card);z-index:1}} tr{{cursor:pointer}} tr:hover{{background:color-mix(in srgb,var(--accent) 8%,transparent)}}
.badge{{display:inline-block;padding:2px 8px;border-radius:999px;background:var(--line);font-size:12px}} .diag{{border-left:4px solid var(--accent);padding:9px 12px;margin:8px 0;background:color-mix(in srgb,var(--accent) 5%,transparent)}} .diag.high{{border-color:var(--bad)}} .diag.medium{{border-color:var(--warn)}}
.inspector pre{{white-space:pre-wrap;max-height:300px;overflow:auto;background:var(--bg);padding:12px;border-radius:8px}} button.download{{padding:9px 13px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--text);cursor:pointer}}
@media(max-width:900px){{.metric{{grid-column:span 4}}.wide,.side{{grid-column:1/-1}}}} @media(max-width:560px){{main{{padding:15px}}.metric{{grid-column:span 6}}header{{display:block}}}}
</style>
</head>
<body><main>
<header><div><h1>Token Efficiency Report</h1><div class="muted" id="session"></div></div><button class="download" id="download">Download analysis JSON</button></header>
<section class="grid">
<div class="card metric"><span class="muted">TER</span><b id="ter"></b></div>
<div class="card metric"><span class="muted">Scored tokens</span><b id="total"></b></div>
<div class="card metric"><span class="muted">Aligned</span><b id="aligned"></b></div>
<div class="card metric"><span class="muted">Waste</span><b id="waste"></b></div>
<div class="card metric"><span class="muted">Cost</span><b id="cost"></b></div>
<div class="card metric"><span class="muted">Reliability</span><b id="reliability"></b></div>
<div class="card wide"><h2>Token composition</h2><div id="composition" class="chart"></div></div>
<div class="card side"><h2>Phase distribution</h2><div id="phases" class="chart"></div></div>
<div class="card full"><h2>Span timeline</h2><div class="muted">Width represents token count. Select a span to inspect it.</div><div id="timeline" class="timeline"></div></div>
<div class="card wide"><h2>Alignment vs confidence</h2><div id="scatter" class="chart"></div></div>
<div class="card side"><h2>Diagnostics</h2><div id="diagnostics"></div></div>
<div class="card full inspector"><h2>Span inspector</h2><div id="inspector" class="muted">Select a timeline segment or table row.</div></div>
<div class="card full"><h2>All classified spans</h2><div class="table-wrap"><table><thead><tr><th>#</th><th>Phase</th><th>Label</th><th>Tokens</th><th>Confidence</th><th>Alignment</th><th>Excerpt</th></tr></thead><tbody id="rows"></tbody></table></div></div>
</section>
</main>
<script id="ter-data" type="application/json">{encoded}</script>
<script>
const d=JSON.parse(document.getElementById('ter-data').textContent), r=d.report, spans=r.spans;
const $=id=>document.getElementById(id), fmt=n=>Number(n||0).toLocaleString();
$('session').textContent='Session '+d.session_id; $('ter').textContent=Number(d.aggregate_ter).toFixed(2); $('total').textContent=fmt(d.total_tokens); $('aligned').textContent=fmt(d.aligned_tokens); $('waste').textContent=fmt(d.waste_tokens);
$('cost').textContent=d.economics?'$'+Number(d.economics.estimated_cost_usd).toFixed(4):'n/a'; $('reliability').textContent=d.uncertainty?d.uncertainty.reliability:'n/a';
const colors=['#3157d5','#31a06b','#d89b2b','#c64e45','#7c58c2','#478da8','#8a96a8'];
function barChart(target,obj){{const entries=Object.entries(obj),total=entries.reduce((a,[,v])=>a+v,0)||1;let x=0;const rects=entries.map(([k,v],i)=>{{const w=v/total*100,s=`<rect x="${{x}}%" y="24" width="${{w}}%" height="42" rx="5" fill="${{colors[i%colors.length]}}"><title>${{k}}: ${{v}} tokens</title></rect>`;x+=w;return s}}).join('');target.innerHTML=`<svg viewBox="0 0 800 95" role="img">${{rects}}</svg><div class="legend">${{entries.map(([k,v],i)=>`<span style="--swatch:${{colors[i%colors.length]}}">${{k.replaceAll('_',' ')}} (${{fmt(v)}})</span>`).join('')}}</div>`}}
barChart($('composition'),r.label_tokens); barChart($('phases'),r.phase_tokens);
function selectSpan(i){{document.querySelectorAll('.segment').forEach(x=>x.classList.remove('active'));const b=document.querySelector(`.segment[data-i="${{i}}"]`);if(b)b.classList.add('active');const s=spans[i];$('inspector').innerHTML=`<div><span class="badge">${{s.phase}}</span> <span class="badge">${{s.label}}</span> <b>${{fmt(s.tokens)}} tokens</b></div><p><b>Confidence:</b> ${{s.confidence.toFixed(3)}} &nbsp; <b>Alignment:</b> ${{s.alignment.toFixed(3)}} &nbsp; <b>Source:</b> ${{s.source_role}}</p><p>${{escapeHtml(s.reason)}}</p><pre>${{escapeHtml(s.text)}}</pre>`}}
function escapeHtml(v){{return String(v).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[c]))}}
const max=Math.max(...spans.map(s=>s.tokens),1);$('timeline').innerHTML=spans.map((s,i)=>`<button class="segment" data-i="${{i}}" title="#${{s.position}} ${{s.label}}, ${{s.tokens}} tokens" style="flex:${{Math.max(s.tokens,8)}};background:${{s.aligned?'var(--good)':'var(--bad)'}}"></button>`).join('');document.querySelectorAll('.segment').forEach(x=>x.onclick=()=>selectSpan(Number(x.dataset.i)));
$('rows').innerHTML=spans.map((s,i)=>`<tr data-i="${{i}}"><td>${{s.position}}</td><td>${{s.phase}}</td><td>${{s.label}}</td><td>${{fmt(s.tokens)}}</td><td>${{s.confidence.toFixed(3)}}</td><td>${{s.alignment.toFixed(3)}}</td><td>${{escapeHtml(s.text.slice(0,120))}}</td></tr>`).join('');document.querySelectorAll('#rows tr').forEach(x=>x.onclick=()=>selectSpan(Number(x.dataset.i)));
function scatter(){{const W=760,H=250,p=38;const pts=spans.map((s,i)=>`<circle cx="${{p+s.alignment*(W-2*p)}}" cy="${{H-p-s.confidence*(H-2*p)}}" r="${{Math.min(16,4+Math.sqrt(s.tokens)/3)}}" fill="${{s.aligned?'var(--good)':'var(--bad)'}}" opacity=".75"><title>Span ${{s.position}}: alignment ${{s.alignment}}, confidence ${{s.confidence}}, ${{s.tokens}} tokens</title></circle>`).join('');$('scatter').innerHTML=`<svg viewBox="0 0 ${{W}} ${{H}}"><line x1="${{p}}" y1="${{H-p}}" x2="${{W-p}}" y2="${{H-p}}" stroke="var(--line)"/><line x1="${{p}}" y1="${{p}}" x2="${{p}}" y2="${{H-p}}" stroke="var(--line)"/>${{pts}}<text x="${{W/2}}" y="${{H-5}}" text-anchor="middle" fill="var(--muted)">Prompt alignment</text><text x="12" y="${{H/2}}" transform="rotate(-90 12 ${{H/2}})" text-anchor="middle" fill="var(--muted)">Confidence</text></svg>`}}scatter();
$('diagnostics').innerHTML=r.diagnostics.map(x=>`<div class="diag ${{x.severity}}"><b>${{x.code}}</b><br>${{escapeHtml(x.message)}}</div>`).join('');
$('download').onclick=()=>{{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{{type:'application/json'}}));a.download=d.session_id+'.ter.json';a.click();URL.revokeObjectURL(a.href)}};
if(spans.length)selectSpan(0);
</script></body></html>"""
