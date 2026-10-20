"""Streamlit dashboard.

Run: streamlit run ui/app.py
Set API_URL (e.g. http://localhost:8000) to go through FastAPI; otherwise the
agent runs in-process and progress is streamed live.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

import config
import style

API_URL = os.getenv("API_URL", "").rstrip("/")
METRIC_LABELS = {"revenue": "Revenue", "operating_income": "Operating income", "net_income": "Net income"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
EVAL_RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"
EXAMPLES = {
    "Profitability": ["Is Infosys becoming more profitable? Explain the change using its annual report.",
                      "How did Asian Paints' operating margin change over the last three years?"],
    "Comparison": ["Compare Infosys and Asian Paints on margins over aligned financial years.",
                   "Compare revenue growth for Infosys, HDFC Bank and Asian Paints."],
    "Risks": ["What risks does HDFC Bank highlight in its latest annual report?",
              "What does Infosys say about client concentration risk?"],
    "Annual report": ["Summarise Asian Paints' strategy from its latest annual report.",
                      "What does HDFC Bank report about deposit growth?"],
}

st.set_page_config(page_title="LedgerMind", page_icon=":material/monitoring:", layout="wide")
st.html(style.CSS)


# ---------------------------------------------------------------- formatting
def indian_group(n: float) -> str:
    """Group digits the Indian way: 1,62,990."""
    s = f"{abs(n):.0f}"
    head, tail = s[:-3], s[-3:]
    if head:
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        s = f"{head},{tail}"
    return ("-" if n < 0 else "") + s


def fmt_money(value, currency="INR") -> str:
    if value is None:
        return "n/a"
    if currency == "INR":
        return f"₹{indian_group(value / 1e7)} Cr"  # 1 crore = 10 million
    return f"{value / 1e9:,.2f} B {currency}"


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def detect_companies(text: str) -> list[str]:
    lc = (text or "").lower()
    return [name for name, meta in config.COMPANIES.items()
            if any(re.search(rf"\b{re.escape(a)}\b", lc) for a in meta["aliases"])]


def style_fig(fig, independent_y: bool = False):
    fig.update_layout(**style.PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False, linecolor=style.BORDER, type="category")
    fig.update_yaxes(gridcolor="#EBEAE6", zeroline=False)
    if independent_y:
        fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


# ---------------------------------------------------------------- runs
def save_run(result: dict) -> None:
    if result.get("run_id") and RUN_ID_RE.match(result["run_id"]):
        try:
            (config.RUNS_DIR / f"{result['run_id']}.json").write_text(json.dumps(result, default=str, indent=1))
        except OSError:
            pass


def load_run(run_id: str) -> dict | None:
    if not RUN_ID_RE.match(run_id or ""):
        return None
    path = config.RUNS_DIR / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def recent_runs(limit: int = 6) -> list[dict]:
    runs = []
    for p in sorted(config.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            r = json.loads(p.read_text())
            runs.append({"run_id": r.get("run_id", p.stem), "question": r.get("question", ""),
                         "status": r.get("status", "")})
        except (OSError, ValueError):
            continue
    return runs


def step_html(t: dict) -> str:
    if t["type"] == "tool_call":
        ok = t["status"] == "success"
        mark = ('<span class="mk" style="background:#DFF0E7;color:#106B4F">✓</span>' if ok else
                '<span class="mk" style="background:#FFF1D6;color:#8A5300">!</span>')
        dur = f'<span class="dur">{esc(t.get("duration_s", ""))}s</span>' if t.get("duration_s") is not None else ""
        return f'<div class="lm-step">{mark}<code>{esc(t["tool"])}</code><span>{esc(t.get("summary", ""))}</span>{dur}</div>'
    ok = t.get("status") == "success"
    label = "model turn" if ok else f"model call failed ({esc(t.get('error', ''))})"
    return (f'<div class="lm-step"><span class="mk" style="background:#FBEFD5;color:#121D36">·</span>'
            f'<span style="color:{style.MUTED}">{label}</span></div>')


def run_research(question: str, companies: list[str], include_news: bool) -> dict:
    if API_URL:
        import httpx

        with st.status("Waiting for the API…", expanded=False):
            r = httpx.post(f"{API_URL}/research", timeout=180,
                           json={"question": question, "companies": companies, "include_news": include_news})
        if r.status_code != 200:
            return {"status": "error", "answer": f"API error {r.status_code}: {r.json().get('detail')}",
                    "citations": [], "trace": [], "warnings": [], "artifacts": {}, "disclaimer": ""}
        return r.json()

    from agent.loop import run_agent

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Report")
        st.html('<div class="lm-sk" style="width:30%"></div><div class="lm-sk"></div><div class="lm-sk" style="width:94%">'
                '</div><div class="lm-sk" style="width:70%"></div><div class="lm-sk" style="width:36%;margin-top:24px">'
                '</div><div class="lm-sk" style="width:88%"></div><div class="lm-sk" style="width:60%"></div>')
    with right:
        status = st.status("Researching…", expanded=True)
    steps: list[str] = []

    def on_event(e):
        if e["type"] == "tool_call" or (e["type"] == "llm_call" and e["status"] != "success"):
            steps.append(step_html(e))
            status.html("".join(steps[-1:]))

    result = run_agent(question, companies=companies or None, include_news=include_news, on_event=on_event)
    status.update(label=f"Done · {result['status']} · {result['duration_s']}s", state="complete", expanded=False)
    return result


# ---------------------------------------------------------------- result sections
def trust_bar(result: dict) -> None:
    cites = result.get("citations", [])
    claims = result.get("claims", [])
    supported = sum(1 for c in claims if c.get("source_ids"))
    removed = sorted({sid for c in claims for sid in c.get("rejected_source_ids", [])})
    errors = result.get("tool_errors", sum(1 for t in result.get("trace", [])
                                           if t["type"] == "tool_call" and t["status"] != "success"))

    def cell(kind, mark, value, label):
        return f'<div><span class="ic {kind}">{mark}</span><span><b>{value}</b><small>{label}</small></span></div>'

    st.html('<div class="lm-trust">'
            + cell("ok", "✓", f"{len(cites)}/{len(cites)}", "citations valid")
            + cell("ok" if supported == len(claims) else "warn", "✓", f"{supported}/{len(claims)}", "claims with a verified source")
            + cell("bad" if removed else "ok", "−" if removed else "✓", len(removed), "citations removed")
            + cell("warn" if errors else "ok", "!" if errors else "✓", errors, "tool errors / retries")
            + "</div>")


def kpi_cards(financials: dict, prices: dict) -> None:
    for ticker, fin in financials.items():
        periods = fin.get("periods", [])
        if not periods:
            continue
        latest, prev = periods[0], periods[1] if len(periods) > 1 else None
        cur, src = fin.get("currency", "INR"), fin.get("source") or "provider"
        cards = []
        for m in ("revenue", "operating_income", "net_income"):
            delta = ""
            if prev and latest.get(m) is not None and prev.get(m):
                pct = (latest[m] - prev[m]) / abs(prev[m]) * 100
                delta = f'<span class="{"up" if pct >= 0 else "down"}">{"▲" if pct >= 0 else "▼"} {abs(pct):.1f}% vs {esc(prev["fiscal_year"])}</span>'
            cards.append(f'<div class="lm-kpi"><span class="lbl">{METRIC_LABELS[m]} · {esc(latest["fiscal_year"])}</span>'
                         f'<span class="val">{fmt_money(latest.get(m), cur)}</span>{delta}'
                         f'<span class="src">{esc(src)} · {esc(latest["fiscal_year"])}</span></div>')
        if latest.get("revenue") and latest.get("net_income") is not None:
            nm = latest["net_income"] / latest["revenue"] * 100
            delta = ""
            if prev and prev.get("revenue") and prev.get("net_income") is not None:
                pp = nm - prev["net_income"] / prev["revenue"] * 100
                delta = f'<span class="{"up" if pp >= 0 else "down"}">{"▲" if pp >= 0 else "▼"} {abs(pp):.1f} pp</span>'
            cards.append(f'<div class="lm-kpi"><span class="lbl">Net margin</span><span class="val">{nm:.1f}%</span>'
                         f'{delta}<span class="src">calculator · net ÷ revenue</span></div>')
        if ticker in prices:
            p = prices[ticker]
            cards.append(f'<div class="lm-kpi"><span class="lbl">Close · {esc(p["date"])}</span>'
                         f'<span class="val">{p["close"]:,.2f}</span><span class="src">{esc(p.get("source", ""))} · '
                         f'{esc(p["currency"])}{" · cached" if p.get("cached") else ""}</span></div>')
        st.markdown(f"**{esc(fin.get('company') or ticker)}** <span class='lm-mono' style='color:{style.MUTED};font-size:12px'>"
                    f"{esc(ticker)} · period ending {esc(latest['period_end'])}</span>", unsafe_allow_html=True)
        st.html(f'<div class="lm-kpis">{"".join(cards)}</div>')
        for w in fin.get("warnings", []):
            st.caption(f":material/warning: {w}")


def source_panel(result: dict) -> None:
    cites = result.get("citations", [])
    if not cites:
        st.caption("No citations in this report.")
        return
    numbers = [c["number"] for c in cites]
    sel = st.session_state.get("sel_cite")
    if sel not in numbers:
        sel = numbers[0]
    picked = st.segmented_control("Citations", numbers, default=sel, format_func=lambda n: f"[{n}]",
                                  key=f"cite_{result.get('run_id')}")
    c = next(x for x in cites if x["number"] == (picked or sel))
    st.session_state["sel_cite"] = c["number"]

    claim = next((cl["claim"] for cl in result.get("claims", []) if c["source_id"] in cl.get("source_ids", [])), None)
    body = f"<mark>{esc(c['excerpt'])}</mark>" if c.get("excerpt") else f"<span style='color:{style.MUTED}'>Structured tool output, no passage text.</span>"
    link = f"<div style='margin-top:8px'><a href='{esc(c['link'])}' target='_blank'>Open source ↗</a></div>" if c.get("link") else ""
    claim_html = (f"<div style='font-size:13px;color:{style.MUTED};margin-bottom:8px'>Supports: “{esc(claim[:220])}”</div>"
                  if claim else "")
    st.html(f"<div class='lm-source'><div style='font-weight:600;margin-bottom:6px'>[{c['number']}] {esc(c['label'])}</div>"
            f"<span class='lm-chip cmp'>Valid source</span><span class='lm-chip muted lm-mono'>{esc(c['source_id'])}</span>"
            f"{claim_html}<div style='font-size:14px;line-height:1.65'>{body}</div>{link}</div>")

    removed = sorted({sid for cl in result.get("claims", []) for sid in cl.get("rejected_source_ids", [])})
    if removed:
        st.html(f"<div class='lm-removed'><b>Removed citations ({len(removed)})</b>The model cited "
                f"{', '.join(f'<code>{esc(s)}</code>' for s in removed)}, which did not match any retrieved source. "
                "They were removed from the report.</div>")


def research_note_html(result: dict) -> str:
    """A print-ready research note; open it and print to PDF."""
    fins = (result.get("artifacts") or {}).get("financials") or {}
    rows = ""
    for ticker, fin in fins.items():
        for p in fin.get("periods", [])[:3]:
            rows += (f"<tr><td>{esc(fin.get('company') or ticker)}</td><td>{esc(p['fiscal_year'])}</td>"
                     + "".join(f"<td>{fmt_money(p.get(m), fin.get('currency', 'INR'))}</td>" for m in METRIC_LABELS) + "</tr>")
    table = (f"<h2>Figures</h2><table><tr><th>Company</th><th>FY</th>"
             + "".join(f"<th>{v}</th>" for v in METRIC_LABELS.values()) + f"</tr>{rows}</table>") if rows else ""
    answer = "".join(f"<p>{esc(par)}</p>" for par in (result.get("answer") or "").split("\n\n") if par.strip())
    sources = "".join(f"<li>[{c['number']}] {esc(c['label'])}</li>" for c in result.get("citations", []))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>LedgerMind note {esc(result.get('run_id'))}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>@page{{size:A4;margin:18mm}}body{{font:12pt/1.55 'IBM Plex Sans',sans-serif;color:#16181D;max-width:720px;margin:auto}}
h1{{font-family:'IBM Plex Serif',Georgia,serif;font-weight:600;font-size:24pt;line-height:1.15;margin:.2em 0}}h2{{font-size:13pt;margin:1.4em 0 .4em}}
.top{{border-bottom:2px solid #16181D;padding-bottom:8px;font-size:10pt;color:#4F535C}}table{{border-collapse:collapse;width:100%;font-size:10pt}}
td,th{{border-bottom:1px solid #E0DFDB;padding:5px 6px;text-align:left}}footer{{margin-top:2em;border-top:1px solid #E0DFDB;padding-top:8px;font-size:9pt;color:#4F535C}}</style>
</head><body><div class="top"><b>LedgerMind</b> · Research note · run {esc(result.get('run_id'))}</div>
<h1>{esc(result.get('question'))}</h1>{answer}{table}<h2>Sources</h2><ol style="list-style:none;padding:0">{sources}</ol>
<footer>{esc(result.get('disclaimer'))}</footer></body></html>"""


def render_report_markdown(result: dict) -> str:
    lines = [f"# Research report — run {result.get('run_id', '')}", "", f"**Question:** {result.get('question', '')}",
             "", result.get("answer", ""), "", "## Citations"]
    for c in result.get("citations", []):
        link = f" — {c['link']}" if c.get("link") else ""
        lines.append(f"[{c['number']}] {c['label']}{link}")
    if result.get("warnings"):
        lines += ["", "## Warnings"] + [f"- {w}" for w in result["warnings"]]
    lines += ["", "---", result.get("disclaimer", "")]
    return "\n".join(lines)


def comparison_section(comp: dict) -> None:
    st.markdown(f"#### Comparison · aligned periods {', '.join(comp['aligned_periods'])}")
    for c in comp.get("comparability_caveats", []):
        st.warning(c, icon=":material/warning:")
    cdf = pd.DataFrame(comp["chart_data"])
    if cdf.empty:
        return
    latest = max(comp["aligned_periods"])

    def fmt(row):
        if row["value"] is None or pd.isna(row["value"]):
            return f"n/a · {row['note']}" if row.get("note") else "n/a"
        return f"{row['value']:.1f}%" if row["unit"] == "%" else fmt_money(row["value"], row["unit"])

    table = cdf[cdf["fiscal_year"] == latest].assign(shown=lambda d: d.apply(fmt, axis=1))
    pivot = table.pivot(index="metric", columns="company", values="shown")
    pivot.index = [f"{m.replace('_', ' ').capitalize()} · {latest}" for m in pivot.index]
    st.dataframe(pivot, width="stretch")

    pct = cdf[(cdf["unit"] == "%") & cdf["value"].notna()]
    if not pct.empty:
        metric = st.segmented_control("Metric", sorted(pct["metric"].unique()), default=sorted(pct["metric"].unique())[0],
                                      format_func=lambda m: m.replace("_", " ").capitalize(), key="cmp_metric")
        sub = pct[pct["metric"] == (metric or pct["metric"].iloc[0])].sort_values("fiscal_year")
        fig = px.bar(sub, x="fiscal_year", y="value", facet_col="company", color_discrete_sequence=[style.NAVY],
                     labels={"value": "%", "fiscal_year": "FY"}, title="Each company on its own scale")
        st.plotly_chart(style_fig(fig, independent_y=True), width="stretch")


def render_result(result: dict) -> None:
    art = result.get("artifacts") or {}
    head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
    with head_l:
        st.caption(f"RUN {result.get('run_id', '')} · {result.get('status')} · {result.get('duration_s', '')}s")
        st.markdown(f"### {result.get('question', '')}")
    with head_r:
        b1, b2, b3 = st.columns(3)
        if b1.button("Share link", icon=":material/link:", width="stretch"):
            st.query_params["run"] = result.get("run_id", "")
            st.toast("Link updated: copy it from the address bar.")
        b2.download_button("Note", research_note_html(result), file_name=f"note-{result.get('run_id', 'run')}.html",
                           mime="text/html", icon=":material/description:", width="stretch",
                           help="Print-ready research note: open it and print to PDF")
        b3.download_button("Markdown", render_report_markdown(result), file_name=f"report-{result.get('run_id', 'run')}.md",
                           mime="text/markdown", icon=":material/download:", width="stretch")

    if result.get("status") != "completed":
        st.error(f"Run status: {result.get('status')}")
    trust_bar(result)
    for w in result.get("warnings", []):
        if not w.startswith("Removed "):
            st.warning(w, icon=":material/warning:")

    tab_report, tab_data, tab_evidence, tab_trace = st.tabs(
        [":material/article: Report", ":material/bar_chart: Performance", ":material/menu_book: Evidence",
         ":material/timeline: Trace"])

    with tab_report:
        fins, prices = art.get("financials") or {}, art.get("prices") or {}
        if fins:
            kpi_cards(fins, prices)
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown(result.get("answer", ""))
        with right:
            source_panel(result)
        st.html(f"<div class='lm-foot'>{esc(result.get('disclaimer', ''))}</div>")

    with tab_data:
        fins = art.get("financials") or {}
        if fins:
            rows = [{"company": fin.get("company") or t, "fiscal_year": p["fiscal_year"], "metric": label,
                     "value_cr": p[m] / 1e7}
                    for t, fin in fins.items() for p in fin.get("periods", [])
                    for m, label in METRIC_LABELS.items() if p.get(m) is not None]
            df = pd.DataFrame(rows)
            if not df.empty:
                fig = px.bar(df.sort_values("fiscal_year"), x="fiscal_year", y="value_cr", color="metric",
                             facet_col="company", barmode="group",
                             labels={"value_cr": "INR crore", "fiscal_year": "Financial year"},
                             title="Annual results · INR crore, as reported by provider")
                st.plotly_chart(style_fig(fig, independent_y=len(fins) > 1), width="stretch")
        if art.get("comparison"):
            comparison_section(art["comparison"])
        if art.get("metrics"):
            st.markdown("#### Calculated metrics")
            st.caption("Deterministic Python. The model does no arithmetic.")
            st.dataframe(pd.DataFrame(art["metrics"])[["metric", "value", "unit", "inputs"]], width="stretch")
        if art.get("news"):
            st.markdown("#### Recent headlines")
            st.caption("Titles only. Full articles are not analysed.")
            for h in art["news"]:
                st.markdown(f"- [{h['title']}]({h['link']}) · {h['published']}")
        if not (fins or art.get("comparison") or art.get("metrics") or art.get("news")):
            st.caption("No structured data was retrieved in this run.")

    with tab_evidence:
        if not art.get("evidence"):
            st.caption("No annual-report passages were retrieved.")
        cited = {c["source_id"] for c in result.get("citations", [])}
        for ev in art.get("evidence", []):
            tag = "cited" if ev["source_id"] in cited else "retrieved"
            with st.expander(f"{ev.get('report_title')} · p. {ev.get('page_label')} · {tag} · "
                             f"distance {ev.get('distance')}", icon=":material/check_circle:" if tag == "cited" else None):
                st.caption(f"source_id `{ev['source_id']}` · file {ev.get('source_file')} · "
                           f"PDF page index {ev.get('page_index')}")
                st.write(ev["text"])

    with tab_trace:
        st.html("".join(step_html(t) for t in result.get("trace", [])))
        st.caption(f"Run {result.get('run_id')} · {result.get('duration_s')}s total")


# ---------------------------------------------------------------- pages
def header() -> None:
    mode = f"API · {API_URL}" if API_URL else f"In-process agent · {config.MODEL or 'model not set'}"
    st.html(f"""<div class="lm-bar"><span class="lm-logo"><svg width="18" height="18" viewBox="0 0 28 28" aria-hidden="true">
<path d="M6 21V7M11.5 21v-8M17 21v-11M22.5 21v-5" stroke="{style.NAVY}" stroke-width="2.6" stroke-linecap="round"/></svg></span>
<span class="lm-name">LedgerMind</span><span style="color:#C9D0E0;font-size:13px">Financial research agent · NSE</span>
<span class="lm-pill"><span class="lm-dot"></span>{esc(mode)}</span></div>""")


def research_page() -> None:
    header()
    if "run" in st.query_params and "result" not in st.session_state:
        loaded = load_run(st.query_params["run"])
        if loaded:
            st.session_state["result"] = loaded

    with st.sidebar:
        if st.button("New research", icon=":material/add:", type="primary", width="stretch"):
            st.session_state.pop("result", None)
            st.query_params.clear()
            st.rerun()
        include_news = st.toggle("Include recent headlines", value=True)
        if not API_URL and not (config.GROQ_API_KEY and config.MODEL):
            st.warning("GROQ_API_KEY / MODEL not set in .env. The agent cannot run.")
        runs = recent_runs()
        if runs:
            st.markdown("**Saved runs**")
            for r in runs:
                if st.button(r["question"][:48] or r["run_id"], key=f"run_{r['run_id']}", width="stretch",
                             help=f"{r['run_id']} · {r['status']}"):
                    st.session_state["result"] = load_run(r["run_id"])
                    st.query_params["run"] = r["run_id"]
                    st.rerun()

    result = st.session_state.get("result")
    if not result:
        st.html('<div class="lm-hero"><h1>Research Indian companies with verified sources</h1>'
                '<p>Figures come from data tools, explanations from annual reports, and every citation is checked.</p></div>')
    question = st.text_area("Research question", key="question", max_chars=config.MAX_QUESTION_CHARS, height=90,
                            placeholder="Ask about profitability, growth, risks or a comparison…",
                            label_visibility="collapsed" if not result else "visible")
    found = detect_companies(question)
    chips = "".join(f'<span class="lm-chip co">{esc(c)}</span>' for c in found) or \
        '<span class="lm-chip muted">No company detected yet. Name one in your question.</span>'
    if len(found) >= 2:
        chips += '<span class="lm-chip cmp">Comparison · aligned years</span>'
    c1, c2 = st.columns([5, 1], vertical_alignment="center")
    c1.html(f"<div><span style='font-size:12px;color:{style.MUTED};margin-right:8px'>Detected:</span>{chips}</div>")
    run = c2.button("Run research", type="primary", icon=":material/arrow_forward:", width="stretch",
                    disabled=not question.strip())

    if not result and not run:
        st.markdown("##### Try an example")
        cols = st.columns(2, gap="large")
        for i, (group, items) in enumerate(EXAMPLES.items()):
            with cols[i % 2]:
                st.caption(group.upper())
                for q in items:
                    st.button(q, key=f"ex_{q}", width="stretch",
                              on_click=lambda q=q: st.session_state.update(question=q))

    if run:
        q = question
        if len(found) >= 2 and "compar" not in q.lower():
            q = f"{q}\nAlso compare {', '.join(found)} on aligned financial years."
        result = run_research(q, found, include_news)
        save_run(result)
        st.session_state["result"] = result
        st.query_params["run"] = result.get("run_id", "")
        st.rerun()

    if result:
        st.divider()
        render_result(result)


def evaluation_page() -> None:
    header()
    st.markdown("### Evaluation results")
    files = sorted(EVAL_RESULTS.glob("eval_*.json"))
    if not files:
        st.info("No evaluation results yet. Run `python eval/run_eval.py` to create one.", icon=":material/info:")
        return
    report = json.loads(files[-1].read_text())
    st.caption(f"{files[-1].name} · eval/run_eval.py")
    t, r = report.get("tasks") or {}, report.get("retrieval") or {}

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    cards = [("Task success", pct(t.get("task_success_rate")), f"{t.get('passed', 0)}/{t.get('executed', 0)} passed"),
             ("Numerical correctness", pct(t.get("numerical_correctness")), "figures match tool output"),
             ("Citation validity", pct(t.get("citation_validity")), "source exists"),
             ("Citation support", pct(t.get("citation_support_heuristic")), "lexical heuristic"),
             ("Retrieval hit@4", "n/a" if r.get("hit_rate_at_4") is None else f"{r['hit_rate_at_4']:.2f}",
              f"{r.get('verified_pairs', 0)}/{r.get('total_pairs', 0)} pairs verified"),
             ("Tool error rate", pct(t.get("tool_error_rate")),
              f"latency p50 {t.get('latency_median_s')} s · p95 {t.get('latency_p95_s')} s")]
    st.html('<div class="lm-kpis">' + "".join(
        f'<div class="lm-kpi"><span class="lbl">{a}</span><span class="val">{esc(b)}</span><span class="src">{esc(c)}</span></div>'
        for a, b, c in cards) + "</div>")
    by_cat = t.get("by_category") or {}
    if by_cat:
        df = pd.DataFrame([{"category": k, **v} for k, v in by_cat.items()])
        df["pass_rate"] = df["pass"] / (df["pass"] + df["fail"]).where(lambda s: s > 0)
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"pass_rate": st.column_config.ProgressColumn("Pass rate", min_value=0, max_value=1,
                                                                                 format="percent")})


st.navigation([st.Page(research_page, title="Research", icon=":material/search:", default=True),
               st.Page(evaluation_page, title="Evaluation", icon=":material/fact_check:")],
              position="top").run()
