"""Streamlit dashboard.

Run: streamlit run ui/app.py
Set API_URL (e.g. http://localhost:8000) to go through FastAPI; otherwise the
agent runs in-process and progress is streamed live.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

import config

API_URL = os.getenv("API_URL", "").rstrip("/")
METRIC_LABELS = {"revenue": "Revenue", "operating_income": "Operating income", "net_income": "Net income"}

st.set_page_config(page_title="Financial Research Agent", page_icon="📊", layout="wide")


def fmt_money(value, currency="INR") -> str:
    if value is None:
        return "n/a"
    if currency == "INR":
        return f"₹{value / 1e7:,.0f} Cr"  # 1 crore = 10 million
    return f"{value / 1e9:,.2f} B {currency}"


def run_research(question: str, companies: list[str], include_news: bool, progress) -> dict:
    if API_URL:
        import httpx

        with progress.status("Waiting for the API...", expanded=False):
            r = httpx.post(f"{API_URL}/research", timeout=180,
                           json={"question": question, "companies": companies, "include_news": include_news})
        if r.status_code != 200:
            return {"status": "error", "answer": f"API error {r.status_code}: {r.json().get('detail')}",
                    "citations": [], "trace": [], "warnings": [], "artifacts": {}, "disclaimer": ""}
        return r.json()

    from agent.loop import run_agent

    status = progress.status("Researching...", expanded=True)

    def on_event(e):
        if e["type"] == "tool_call":
            icon = "✅" if e["status"] == "success" else "⚠️"
            status.write(f"{icon} `{e['tool']}` — {e['summary']}")
        elif e["type"] == "llm_call" and e["status"] != "success":
            status.write(f"⚠️ LLM call failed ({e.get('error')})")

    result = run_agent(question, companies=companies or None, include_news=include_news, on_event=on_event)
    status.update(label=f"Done ({result['status']}, {result['duration_s']}s)", state="complete", expanded=False)
    return result


def financial_frame(financials: dict) -> pd.DataFrame:
    rows = []
    for ticker, fin in financials.items():
        for p in fin.get("periods", []):
            for m, label in METRIC_LABELS.items():
                if p.get(m) is not None:
                    rows.append({"company": fin.get("company") or ticker, "fiscal_year": p["fiscal_year"],
                                 "metric": label, "value_cr": p[m] / 1e7})
    return pd.DataFrame(rows)


def render_cards(financials: dict, prices: dict) -> None:
    for ticker, fin in financials.items():
        periods = fin.get("periods", [])
        if not periods:
            continue
        latest, prev = periods[0], periods[1] if len(periods) > 1 else None
        cur = fin.get("currency", "INR")
        st.markdown(f"**{fin.get('company') or ticker}** · {latest['fiscal_year']} "
                    f"(period ending {latest['period_end']}) · source: {fin.get('source')}")
        cols = st.columns(5)
        for col, m in zip(cols, ("revenue", "operating_income", "net_income")):
            delta = None
            if prev and latest.get(m) is not None and prev.get(m):
                delta = f"{(latest[m] - prev[m]) / abs(prev[m]) * 100:+.1f}% vs {prev['fiscal_year']}"
            col.metric(METRIC_LABELS[m], fmt_money(latest.get(m), cur), delta)
        if latest.get("revenue") and latest.get("net_income") is not None:
            cols[3].metric("Net margin", f"{latest['net_income'] / latest['revenue'] * 100:.1f}%")
        if ticker in prices:
            p = prices[ticker]
            cols[4].metric(f"Close ({p['date']})", f"{p['close']:,.2f} {p['currency']}")
        for w in fin.get("warnings", []):
            st.caption(f"⚠️ {w}")


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


# ---------------------------------------------------------------- layout
st.title("📊 Financial Research Agent")
st.caption("NSE-listed companies · tool-calling agent · annual-report RAG · cited reports. "
           "Educational use only — not investment advice.")

with st.sidebar:
    st.header("Research settings")
    selected = st.multiselect("Companies", list(config.COMPANIES), default=["Infosys"], max_selections=3)
    include_news = st.toggle("Include recent headlines", value=True)
    compare_mode = st.toggle("Comparison mode", value=False, help="Ask the agent to compare selected companies")
    st.divider()
    mode = f"API: {API_URL}" if API_URL else "In-process agent"
    st.caption(f"Mode: {mode}")
    if not API_URL and not (config.GROQ_API_KEY and config.MODEL):
        st.warning("GROQ_API_KEY / MODEL not set in .env — the agent cannot run.")

default_q = "Is Infosys becoming more profitable? Explain the change using its annual report."
question = st.text_area("Research question", value=default_q, max_chars=config.MAX_QUESTION_CHARS, height=90)
if compare_mode and len(selected) >= 2 and "compar" not in question.lower():
    question = f"{question}\nAlso compare {', '.join(selected)} on aligned financial years."

if st.button("Run research", type="primary", disabled=not question.strip()):
    st.session_state["result"] = run_research(question, selected, include_news, st)

result = st.session_state.get("result")
if result:
    art = result.get("artifacts") or {}
    if result.get("status") not in ("completed",):
        st.error(f"Run status: {result.get('status')}")
    for w in result.get("warnings", []):
        st.warning(w)

    tab_report, tab_data, tab_evidence, tab_trace = st.tabs(
        ["Report", "Performance & charts", "Evidence & citations", "Execution trace"])

    with tab_report:
        st.markdown(result.get("answer", ""))
        if result.get("citations"):
            st.markdown("#### Citations")
            for c in result["citations"]:
                label = f"[{c['number']}] {c['label']}"
                st.markdown(f"{label} — [link]({c['link']})" if c.get("link") else label)
        st.info(result.get("disclaimer", ""))
        st.download_button("Download report (Markdown)", render_report_markdown(result),
                           file_name=f"report-{result.get('run_id', 'run')}.md", mime="text/markdown")

    with tab_data:
        fins, prices = art.get("financials") or {}, art.get("prices") or {}
        if fins:
            render_cards(fins, prices)
            df = financial_frame(fins)
            if not df.empty:
                fig = px.bar(df.sort_values("fiscal_year"), x="fiscal_year", y="value_cr", color="metric",
                             facet_col="company", barmode="group",
                             labels={"value_cr": "INR crore", "fiscal_year": "Financial year"},
                             title="Annual results (INR crore, as reported by provider)")
                st.plotly_chart(fig, width="stretch")
        comp = art.get("comparison")
        if comp:
            cdf = pd.DataFrame(comp["chart_data"])
            pct = cdf[(cdf["unit"] == "%") & cdf["value"].notna()]
            if not pct.empty:
                fig = px.line(pct.sort_values("fiscal_year"), x="fiscal_year", y="value", color="company",
                              facet_col="metric", markers=True, labels={"value": "%", "fiscal_year": "FY"},
                              title=f"Comparison on aligned periods: {', '.join(comp['aligned_periods'])}")
                st.plotly_chart(fig, width="stretch")
            for c in comp.get("comparability_caveats", []):
                st.warning(c)
        if art.get("metrics"):
            st.markdown("#### Calculated metrics (deterministic Python)")
            st.dataframe(pd.DataFrame(art["metrics"])[["metric", "value", "unit", "inputs"]],
                         width="stretch")
        if art.get("news"):
            st.markdown("#### Headlines (titles only — full articles not analysed)")
            for h in art["news"]:
                st.markdown(f"- [{h['title']}]({h['link']}) · {h['published']}")
        if not (fins or comp or art.get("metrics") or art.get("news")):
            st.caption("No structured data was retrieved in this run.")

    with tab_evidence:
        if not art.get("evidence"):
            st.caption("No annual-report passages were retrieved.")
        cited = {c["source_id"] for c in result.get("citations", [])}
        for ev in art.get("evidence", []):
            tag = "✅ cited" if ev["source_id"] in cited else "retrieved"
            with st.expander(f"{ev.get('report_title')} · p. {ev.get('page_label')} · {tag} · "
                             f"distance {ev.get('distance')}"):
                st.caption(f"source_id `{ev['source_id']}` · file {ev.get('source_file')} · "
                           f"PDF page index {ev.get('page_index')}")
                st.write(ev["text"])

    with tab_trace:
        for t in result.get("trace", []):
            if t["type"] == "tool_call":
                icon = "✅" if t["status"] == "success" else "⚠️"
                st.markdown(f"{icon} **step {t['step']}** `{t['tool']}`({t['arguments']}) → "
                            f"{t['summary']} · {t['duration_s']}s")
            else:
                st.caption(f"step {t['step']}: LLM call {t['status']} · {t['duration_s']}s")
        st.caption(f"Run {result.get('run_id')} · {result.get('duration_s')}s total")
