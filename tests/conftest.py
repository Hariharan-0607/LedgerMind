"""Shared offline fixtures: fake yfinance, fake LLM client, isolated cache and vector store."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import cache as cache_mod  # noqa: E402
from services import financial_data  # noqa: E402

# Values in INR, ordered newest first like yfinance. Synthetic test data — not real figures.
FAKE_STATEMENTS = {
    "INFY.NS": {
        "Total Revenue": [1_600_000_000_000, 1_500_000_000_000, 1_400_000_000_000],
        "Operating Income": [336_000_000_000, 312_000_000_000, 294_000_000_000],
        "Net Income": [264_000_000_000, 240_000_000_000, 238_000_000_000],
    },
    "ASIANPAINT.NS": {
        "Total Revenue": [340_000_000_000, 350_000_000_000, 340_000_000_000],
        "Operating Income": [55_000_000_000, 70_000_000_000, 60_000_000_000],
        "Net Income": [38_000_000_000, 54_000_000_000, 41_000_000_000],
    },
    "HDFCBANK.NS": {  # bank: no operating income line
        "Total Revenue": [2_800_000_000_000, 2_400_000_000_000, 1_700_000_000_000],
        "Net Income": [700_000_000_000, 640_000_000_000, 460_000_000_000],
    },
}
PERIODS = pd.to_datetime(["2025-03-31", "2024-03-31", "2023-03-31"])


class FakeTicker:
    def __init__(self, symbol, fail=False, empty=False):
        self.symbol, self.fail, self.empty = symbol, fail, empty
        self.fast_info = {"currency": "INR"}
        self.info = {"financialCurrency": "INR"}

    def history(self, period="7d", auto_adjust=False):
        if self.fail:
            raise ConnectionError("network down")
        if self.empty:
            return pd.DataFrame()
        idx = pd.to_datetime(["2026-10-15", "2026-10-16", "2026-10-17"])
        return pd.DataFrame({"Close": [1500.0, 1510.5, 1523.25]}, index=idx)

    @property
    def income_stmt(self):
        if self.fail:
            raise ConnectionError("network down")
        if self.empty:
            return pd.DataFrame()
        data = FAKE_STATEMENTS[self.symbol]
        return pd.DataFrame({col: {k: v[i] for k, v in data.items()} for i, col in enumerate(PERIODS)})


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod.cache, "directory", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    yield


@pytest.fixture
def fake_yf(monkeypatch):
    state = {"fail": False, "empty": False}
    monkeypatch.setattr(financial_data, "_ticker_factory",
                        lambda s: FakeTicker(s, fail=state["fail"], empty=state["empty"]))
    return state


@pytest.fixture
def chroma_collection():
    import chromadb

    from rag.store import HashEmbeddingFunction, set_collection

    client = chromadb.EphemeralClient()
    name = f"test_{id(client)}"
    col = client.get_or_create_collection(name=name, embedding_function=HashEmbeddingFunction(),
                                          metadata={"hnsw:space": "cosine"})
    set_collection(col)
    yield col
    set_collection(None)
    client.delete_collection(name)


# ------------------------------------------------------------- fake LLM
def tool_call(name, args, call_id=None):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_id or f"call_{name}_{abs(hash(raw)) % 10000}", type="function",
                           function=SimpleNamespace(name=name, arguments=raw))


def llm_response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeLLM:
    """Replays scripted responses; records the messages it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        r = self.responses.pop(0)
        return r(kwargs) if callable(r) else r
