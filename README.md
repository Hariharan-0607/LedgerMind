# LedgerMind: Financial Research Agent

LedgerMind is a tool-calling AI research assistant for NSE-listed Indian companies. You ask a question in plain English. The agent picks which tools to call, fetches financial statements and prices, does every calculation in deterministic Python, searches annual reports with retrieval-augmented generation (RAG), optionally pulls recent headlines, and writes a structured report. Every citation in the report has been checked by the application.

> Educational research tool. **Not investment advice.** See the [disclaimer](#disclaimer).

## The problem

General-purpose LLMs make up financial figures, page references and explanations. This project keeps the LLM in a planning-and-writing role:

- **Numbers** come only from tools: yfinance for data, Python for arithmetic.
- **Explanations** come only from retrieved annual-report passages.
- **Citations** come only from the application. The model can reference internal `source_id`s, and the app maps each one to trusted metadata. Unknown IDs are removed and reported as warnings.

## Features

- A hand-written agent loop (no LangChain or LangGraph). It has a fixed tool registry, JSON-schema argument validation, handles several tool calls per model turn, retries a failed tool at most once, caps the number of steps, and records a full execution trace.
- Six tools: `get_stock_price`, `get_financials`, `calculate_named_metric`, `search_filings`, `get_news`, `compare_companies`.
- Annual-report RAG: PyMuPDF page-by-page extraction that keeps both the PDF page index and the printed page label, 1,200-character chunks with 200-character overlap within a page, persistent ChromaDB storage, and retrieval filtered by company.
- Citation validation that checks *validity* (the source exists) separately from *support* (the source backs the claim).
- Comparisons align financial years across companies, check currencies, show missing values as null (never zero), and warn about sector differences (bank vs. non-bank).
- A FastAPI backend with Pydantic validation, a company allow-list, run IDs, saved runs and controlled errors.
- A Streamlit dashboard with metric cards, Plotly charts, the report with clickable citations, an evidence panel, a live execution trace and a Markdown export.
- Reliability: bounded tenacity retries, a timestamped file cache with cached data labelled as such, logging without secrets, and caps on input size.
- A 50-task evaluation suite plus a 30-query retrieval set, reporting task success, numerical correctness, citation validity and support, retrieval hit@4, tool error rate and latency.
- 77 offline automated tests.

**Supported companies (v1):** Infosys (`INFY.NS`), HDFC Bank (`HDFCBANK.NS`), Asian Paints (`ASIANPAINT.NS`).

## Architecture

```
 Streamlit UI ──HTTP──▶ FastAPI ──▶ Agent loop ──▶ LLM (Groq, OpenAI-compatible)
 (ui/app.py)           (api/main.py)  (agent/loop.py)
                                         │  validated calls only
                                         ▼
                              Tool registry (agent/tools.py)
          ┌──────────────┬──────────────┼───────────────┬───────────────┐
          ▼              ▼              ▼               ▼               ▼
   financial_data   calculator     rag/retrieve      news.py       compare.py
   (yfinance)       (no eval)      (ChromaDB)       (RSS)        (aligned FYs)
          └───────────── cache.py ──────┘
                                         │
                       citations.py ◀── evidence registry (per run)
                                         │
                            validated report + trace + warnings

 Offline: annual-report PDFs ──▶ rag/ingest.py ──▶ data/chroma (persistent)
```

The agent loop is the only component that talks to the LLM. The model *requests* tool calls; the application validates and runs them. Every tool call gets exactly one tool result. Tool output is passed back to the model wrapped as `untrusted_data`, so documents and headlines can't override the system rules. If you set `API_URL`, the UI goes through FastAPI; otherwise it runs the agent in-process and streams progress live.

## Tech stack

Python 3.11+ (tested on 3.13) · Groq via the `openai` client · yfinance · PyMuPDF · ChromaDB · Sentence Transformers (optional; falls back to Chroma's ONNX MiniLM) · FastAPI + Uvicorn · Streamlit + Plotly · Pandas/NumPy · Tenacity · Pytest

## Setup

```bash
git clone https://github.com/Hariharan-0607/LedgerMind.git
cd LedgerMind
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install sentence-transformers  # optional, recommended for better retrieval
cp .env.example .env               # then edit .env
```

Environment variables (placeholders only, never commit `.env`):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM provider key |
| `MODEL` | A model your provider currently lists as supporting tool calling |
| `LLM_BASE_URL` | Optional; defaults to Groq's OpenAI-compatible endpoint |
| `EMBEDDING_MODEL` | Optional; `hash` gives offline lexical embeddings for tests |
| `API_URL` | Optional; makes the UI call FastAPI |

## Ingest annual reports

1. Download official, text-based annual-report PDFs from each company's investor-relations page into `data/reports/`.
2. Ingest them one at a time or from a manifest:

```bash
python -m rag.ingest --company Infosys --pdf data/reports/infosys-annual-report-2025.pdf \
    --year 2025 --fy FY2025 --title "Infosys Integrated Annual Report 2024-25"
# or
cp data/reports/manifest.example.json data/reports/manifest.json   # edit paths
python -m rag.ingest --manifest data/reports/manifest.json
```

Re-ingesting a report replaces its chunks (IDs are stable: `infosys-2025-p112-c0`). The index lives in `data/chroma/`. On hosts with ephemeral filesystems, either re-run ingestion at startup or mount persistent storage.

## Run

```bash
uvicorn api.main:app --reload            # API on :8000, docs at /docs
streamlit run ui/app.py                  # dashboard (in-process agent)
API_URL=http://localhost:8000 streamlit run ui/app.py   # dashboard via API
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check (also reports whether the LLM is configured) |
| GET | `/companies` | Supported companies and tickers |
| POST | `/research` | Run a research question |
| POST | `/compare` | Compare 2–3 companies on aligned periods |
| GET | `/research/{run_id}` | Retrieve a saved run |
| GET | `/research/{run_id}/trace` | Tool trace for a run |

```bash
curl -X POST localhost:8000/research -H 'content-type: application/json' -d '{
  "question": "How has Infosys revenue changed over its last three reported financial years?",
  "companies": ["Infosys"], "include_news": true}'
```

Response shape: `{run_id, status, answer, metrics[], citations[], trace[], warnings[], artifacts{}, disclaimer, duration_s}`.

Runs are saved to `logs/runs/<run_id>.json`. Each file holds the question, report, citations and trace, and nothing that identifies the user.

## Testing

```bash
EMBEDDING_MODEL=hash pytest
```

All tests run offline using fake yfinance, LLM, RSS and in-memory Chroma fixtures. They cover:

- Calculations, zero and non-finite inputs.
- Invalid tickers, missing statement fields, provider failures and cache labelling.
- One or several tool calls per turn, unknown tools, malformed arguments, the per-tool retry limit, the step limit, withholding the news tool, and LLM outages.
- The RAG company filter, expected-page retrieval, mapping citations to report pages, and rejecting unknown source IDs.
- API validation, saved runs, malicious run IDs, and making sure errors never leak details.

Current status: **77 passed**.

## Evaluation

```bash
python -m eval.run_eval                 # all runnable tasks
python -m eval.run_eval --mode direct   # tool-level tasks, no LLM needed
```

`eval/tasks.json` holds 50 fixed tasks across these categories: arithmetic (8), financial facts (9), tool selection (8), comparisons (5), citation support (5), robustness (11) and limitations (4). The runner reports:

| Metric | Definition |
|---|---|
| Task success rate | Tasks that meet their predefined expected outcome ÷ tasks executed |
| Citation validity | Displayed citations that map to real retrieved sources ÷ all citation attempts |
| Citation support | Heuristic token-overlap score per cited claim. Exported to `eval/results/citation_review_*.json` for human labelling, which is the reference measure |
| Retrieval hit rate @4 | Verified queries whose expected page is in the top 4 ÷ verified queries |
| Numerical correctness | Numeric tasks where the numbers match independent or tool values within 1% |
| Tool error rate | Failed tool executions ÷ total tool executions |
| Latency | Median and 95th-percentile task duration |

Tasks that can't run (no LLM key, reports not ingested) are listed as **SKIPPED** and are excluded from the rates, never counted as passes. Each run saves a timestamped JSON file in `eval/results/`.

`eval/retrieval_set.json` holds 30 retrieval questions (10 per company). **You have to verify the expected pages yourself** against the PDFs you ingested: fill in `expected_page_labels` and set `"verified": true`. The development target is ≥ 80% hit@4. That is a goal, not a guaranteed result.

### Results

_No published measurements yet._ When the evaluation was run while building this project, Yahoo Finance was blocked by that sandbox's network, so no live-data, LLM or retrieval numbers were collected there. Run the suite in your own environment and record the real figures here, with the date, model, test-set size and limitations. Don't publish example numbers.

## Known limitations

- **Provider reliability:** yfinance is unofficial. Data can be delayed, revised, missing or mapped to the wrong field. Prices are the latest session *close*, not real-time quotes.
- **Statement labels** vary. Banks usually have no "Operating Income" line, so operating margin isn't available for HDFC Bank.
- **Sector comparability:** bank revenue and margins aren't comparable to IT or consumer companies. Banking metrics such as net interest margin aren't implemented yet.
- **PDF tables** often extract poorly. Annual reports are used for explanations; structured data is used for numbers.
- **Report coverage:** RAG only knows the reports you ingested.
- **Headline-only news:** full articles aren't fetched or analysed.
- **Citation support** is scored by a heuristic. Human review is the reference.
- Free hosting tiers may not support local embeddings, PDF processing or persistent storage. Test memory use, cold starts and the filesystem before deploying.

## Deployment checklist

1. Check for secrets with `git grep -i "api_key\|gsk_"`. `.env` is gitignored.
2. Host FastAPI (Render, Railway, Fly.io, etc.) and Streamlit (Streamlit Community Cloud or the same host). Set `GROQ_API_KEY` and `MODEL` as platform secrets, and set `API_URL` for the UI.
3. Pre-index the reports or attach persistent storage for `data/chroma/`.
4. Verify `/health`, model availability, timeouts and rate limits. Then test from a clean browser session.
5. Record a 2-minute demo: a question, the tool trace, one citation and the evaluation output.

## Project structure

```
agent/      loop.py (agent loop) · tools.py (registry + schemas) · prompts.py
api/        main.py (FastAPI)
rag/        ingest.py · retrieve.py · store.py (Chroma + embedders)
services/   financial_data.py · calculator.py · news.py · citations.py · compare.py · cache.py
ui/         app.py (Streamlit)
eval/       tasks.json (50) · retrieval_set.json (30) · run_eval.py
tests/      offline pytest suite
data/       reports/ (PDFs, gitignored) · cache/ · chroma/ (gitignored)
```

## Roadmap

- An optional LangGraph version, compared against this loop on the same evaluation set.
- Banking metrics (NIM, GNPA, CASA) from a reliable source.
- Reranking for retrieval, and validated table extraction.

## Disclaimer

This project is an educational financial research tool. Market and financial data may be delayed, incomplete, revised, or incorrectly mapped by upstream providers. Annual-report text extraction and semantic retrieval can miss context, especially in tables. Check generated summaries against the cited sources. Nothing here is guaranteed accurate, certified financial advice, or a personalised recommendation to buy, sell, or hold securities.
