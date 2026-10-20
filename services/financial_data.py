"""Financial data retrieval via yfinance with structured results and errors.

Every successful result carries source, date, currency, unit, and reporting
period metadata plus a ``source_id`` that the citation registry can verify.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import COMPANIES, FINANCIALS_CACHE_TTL_S, PRICE_CACHE_TTL_S, resolve_company
from services.cache import cache

log = logging.getLogger(__name__)

# Statement labels vary across companies and providers; try in order.
LABEL_CANDIDATES: dict[str, list[str]] = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported", "EBIT"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    ],
}

ALLOWED_TICKERS = {info["ticker"] for info in COMPANIES.values()}


class TransientProviderError(Exception):
    """Network-ish failures worth a bounded retry."""


def _ticker_factory(symbol: str):
    import yfinance as yf  # imported lazily so tests don't require network setup

    return yf.Ticker(symbol)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _error(ticker: str, message: str, code: str = "provider_error") -> dict:
    return {"ticker": ticker, "status": "error", "error_code": code, "error": message, "source": "yfinance"}


def normalize_ticker(ticker: str) -> tuple[str | None, str | None]:
    """Return (ticker, error). Accepts supported tickers or company names."""
    if not isinstance(ticker, str) or not ticker.strip():
        return None, "Ticker must be a non-empty string"
    t = ticker.strip()
    if t.upper() in ALLOWED_TICKERS:
        return t.upper(), None
    canonical = resolve_company(t)
    if canonical:
        return COMPANIES[canonical]["ticker"], None
    return None, f"Unsupported ticker or company '{t}'. Supported: {sorted(ALLOWED_TICKERS)}"


def _currency(tk) -> str:
    for getter in (lambda: tk.fast_info.get("currency"), lambda: tk.info.get("financialCurrency")):
        try:
            cur = getter()
            if cur:
                return str(cur)
        except Exception:  # provider metadata is optional
            continue
    return "INR"


@retry(
    retry=retry_if_exception_type(TransientProviderError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
def _fetch_history(symbol: str):
    try:
        return _ticker_factory(symbol).history(period="7d", auto_adjust=False)
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise TransientProviderError(str(exc)) from exc


@retry(
    retry=retry_if_exception_type(TransientProviderError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
def _fetch_income_statement(symbol: str):
    try:
        tk = _ticker_factory(symbol)
        return tk.income_stmt, _currency(tk)
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise TransientProviderError(str(exc)) from exc


def get_stock_price(ticker: str) -> dict:
    """Latest available trading-session close (NOT a real-time quote)."""
    symbol, err = normalize_ticker(ticker)
    if err:
        return _error(str(ticker), err, "invalid_ticker")
    cached = cache.get("price", symbol, PRICE_CACHE_TTL_S)
    if cached:
        return cached
    try:
        hist = _fetch_history(symbol)
    except Exception as exc:
        log.warning("price fetch failed for %s: %s", symbol, type(exc).__name__)
        return _error(symbol, f"Price provider failed: {type(exc).__name__}")
    if hist is None or len(hist) == 0 or "Close" not in hist:
        return _error(symbol, "No price history returned by provider", "no_data")
    closes = hist["Close"].dropna()
    if closes.empty:
        return _error(symbol, "No valid closing prices in history", "no_data")
    close = float(closes.iloc[-1])
    if not math.isfinite(close):
        return _error(symbol, "Closing price is not finite", "bad_data")
    date = closes.index[-1]
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10]
    try:
        currency = _currency(_ticker_factory(symbol))
    except Exception:
        currency = "INR"
    result = {
        "ticker": symbol,
        "date": date_str,
        "close": round(close, 2),
        "price_type": "session_close",
        "currency": currency,
        "source": "yfinance",
        "source_id": f"price:{symbol}:{date_str}",
        "retrieved_at": _now_iso(),
        "note": "Latest available session close; may be delayed. Not a real-time quote.",
        "status": "success",
    }
    cache.set("price", symbol, result)
    return result


def _pick(statement, metric: str):
    for label in LABEL_CANDIDATES[metric]:
        if label in statement.index:
            return label
    return None


def get_financials(ticker: str, periods: int = 3) -> dict:
    """Last ``periods`` annual reporting periods of revenue, operating & net income."""
    symbol, err = normalize_ticker(ticker)
    if err:
        return _error(str(ticker), err, "invalid_ticker")
    periods = max(1, min(int(periods or 3), 5))
    cache_key = f"{symbol}:{periods}"
    cached = cache.get("financials", cache_key, FINANCIALS_CACHE_TTL_S)
    if cached:
        return cached
    try:
        statement, currency = _fetch_income_statement(symbol)
    except Exception as exc:
        log.warning("financials fetch failed for %s: %s", symbol, type(exc).__name__)
        return _error(symbol, f"Financials provider failed: {type(exc).__name__}")
    if statement is None or getattr(statement, "empty", True):
        return _error(symbol, "No income statement returned by provider", "no_data")

    labels = {m: _pick(statement, m) for m in LABEL_CANDIDATES}
    columns = sorted(statement.columns, reverse=True)[:periods]
    rows = []
    for col in columns:
        period_end = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
        row = {"period_end": period_end, "fiscal_year": _fiscal_year_label(col)}
        for metric, label in labels.items():
            value = None
            if label is not None:
                raw = statement.at[label, col]
                try:
                    raw = float(raw)
                    value = raw if math.isfinite(raw) else None
                except (TypeError, ValueError):
                    value = None
            row[metric] = value
        rows.append(row)

    missing = sorted({m for r in rows for m in LABEL_CANDIDATES if r[m] is None})
    warnings = []
    if missing:
        warnings.append(f"Missing values for: {missing}. They are reported as null, not zero.")
    if labels["operating_income"] is None:
        warnings.append("No operating income line found (common for banks); operating margin unavailable.")

    result = {
        "ticker": symbol,
        "company": resolve_company(symbol),
        "statement": "annual income statement",
        "currency": currency,
        "unit": "absolute (currency units, as reported by provider)",
        "labels_used": labels,
        "periods": rows,
        "source": "yfinance",
        "source_id": f"fin:{symbol}:income_stmt",
        "retrieved_at": _now_iso(),
        "warnings": warnings,
        "status": "success",
    }
    cache.set("financials", cache_key, result)
    return result


def _fiscal_year_label(col) -> str:
    """Indian companies typically report FY ending 31 March: period 2025-03-31 -> FY2025."""
    if hasattr(col, "year"):
        return f"FY{col.year}" if col.month <= 3 else f"FY{col.year + 1}"
    return str(col)[:4]
