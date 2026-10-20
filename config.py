"""Central configuration: supported companies, paths, and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = ROOT / "logs"
RUNS_DIR = LOG_DIR / "runs"

for _d in (REPORTS_DIR, CACHE_DIR, LOG_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Version-one allow-list. Keys are canonical company names.
COMPANIES: dict[str, dict] = {
    "Infosys": {
        "ticker": "INFY.NS",
        "sector": "IT Services",
        "aliases": ["infosys", "infy", "infy.ns", "infosys ltd", "infosys limited"],
        "news_query": "Infosys",
    },
    "HDFC Bank": {
        "ticker": "HDFCBANK.NS",
        "sector": "Banking",
        "aliases": ["hdfc bank", "hdfcbank", "hdfcbank.ns", "hdfc"],
        "news_query": "HDFC Bank",
    },
    "Asian Paints": {
        "ticker": "ASIANPAINT.NS",
        "sector": "Paints / Consumer",
        "aliases": ["asian paints", "asianpaint", "asianpaint.ns", "asian paints ltd"],
        "news_query": "Asian Paints",
    },
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("MODEL", "")

MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

PRICE_CACHE_TTL_S = int(os.getenv("PRICE_CACHE_TTL_S", str(6 * 3600)))
FINANCIALS_CACHE_TTL_S = int(os.getenv("FINANCIALS_CACHE_TTL_S", str(7 * 24 * 3600)))
NEWS_CACHE_TTL_S = int(os.getenv("NEWS_CACHE_TTL_S", str(1800)))


def resolve_company(name: str) -> str | None:
    """Map a user-supplied company name or ticker to a canonical supported name."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    for canonical, info in COMPANIES.items():
        if key == canonical.lower() or key == info["ticker"].lower() or key in info["aliases"]:
            return canonical
    return None


def ticker_for(name: str) -> str | None:
    canonical = resolve_company(name)
    return COMPANIES[canonical]["ticker"] if canonical else None
