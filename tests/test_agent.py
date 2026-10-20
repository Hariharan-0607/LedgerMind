import json
from types import SimpleNamespace

import pytest

from agent.loop import run_agent
from agent.tools import execute_registered_tool, validate_arguments
from tests.conftest import FakeLLM, llm_response, tool_call


def test_one_tool_call_succeeds(fake_yf):
    llm = FakeLLM([
        llm_response(tool_calls=[tool_call("get_stock_price", {"ticker": "INFY.NS"}, "c1")]),
        llm_response("Infosys closed at 1523.25 INR on 2026-10-17 [[price:INFY.NS:2026-10-17]]."),
    ])
    r = run_agent("Latest Infosys close?", client=llm, model="m")
    assert r["status"] == "completed"
    assert r["answer"] == "Infosys closed at 1523.25 INR on 2026-10-17 [1]."
    assert r["citations"][0]["type"] == "price"
    assert r["tool_calls"] == 1 and r["tool_errors"] == 0
    tool_msgs = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert json.loads(tool_msgs[0]["content"])["untrusted_data"]["close"] == 1523.25


def test_multiple_tool_calls_in_one_response(fake_yf):
    llm = FakeLLM([
        llm_response(tool_calls=[tool_call("get_financials", {"ticker": "INFY.NS"}, "a"),
                                 tool_call("get_financials", {"ticker": "ASIANPAINT.NS"}, "b"),
                                 tool_call("calculate_named_metric",
                                           {"metric": "operating_margin",
                                            "inputs": {"revenue": 1000, "operating_income": 210}}, "c")]),
        llm_response("Done [[fin:INFY.NS:income_stmt]] [[fin:ASIANPAINT.NS:income_stmt]]."),
    ])
    r = run_agent("Compare profitability", client=llm, model="m")
    assert r["status"] == "completed"
    second = llm.calls[1]
    ids = [m["tool_call_id"] for m in second if m["role"] == "tool"]
    assert ids == ["a", "b", "c"]  # every tool call has exactly one result, in order
    assert set(r["artifacts"]["financials"]) == {"INFY.NS", "ASIANPAINT.NS"}
    assert r["artifacts"]["metrics"][0]["value"] == 21.0
    assert len(r["citations"]) == 2


def test_unknown_tools_are_rejected(fake_yf):
    llm = FakeLLM([
        llm_response(tool_calls=[tool_call("run_python", {"code": "import os"}, "x")]),
        llm_response("Data unavailable."),
    ])
    r = run_agent("Hack it", client=llm, model="m")
    assert r["status"] == "completed"
    call = r["trace"][1]
    assert call["tool"] == "run_python" and call["status"] == "error"
    tool_msg = next(m for m in llm.calls[1] if m["role"] == "tool")
    assert "Unknown tool" in tool_msg["content"]


def test_malformed_arguments_do_not_crash(fake_yf):
    llm = FakeLLM([
        llm_response(tool_calls=[tool_call("get_financials", "{not json", "m1"),
                                 tool_call("get_financials", {"ticker": 5}, "m2"),
                                 tool_call("get_financials", {"ticker": "INFY.NS", "x": 1}, "m3")]),
        llm_response("Data unavailable."),
    ])
    r = run_agent("Infosys revenue", client=llm, model="m")
    assert r["status"] == "completed"
    statuses = [t["status"] for t in r["trace"] if t["type"] == "tool_call"]
    assert statuses == ["error", "error", "error"]


def test_retry_limit_per_tool(fake_yf):
    bad = {"ticker": "NOPE.NS"}
    llm = FakeLLM([llm_response(tool_calls=[tool_call("get_financials", bad, f"r{i}")]) for i in range(3)]
                  + [llm_response("Unavailable.")])
    r = run_agent("x revenue", client=llm, model="m")
    third = [t for t in r["trace"] if t["type"] == "tool_call"][2]
    assert "retry_limit" in json.dumps(llm.calls[3][-1])
    assert third["status"] == "error"


def test_maximum_step_limit_stops_repeated_tool_calls(fake_yf):
    llm = FakeLLM([lambda kw: llm_response(tool_calls=[tool_call("get_stock_price", {"ticker": "INFY.NS"})])
                   for _ in range(10)])
    r = run_agent("loop forever", client=llm, model="m", max_steps=3)
    assert r["status"] == "step_limit"
    assert len(llm.calls) == 3
    assert any("maximum" in w for w in r["warnings"])


def test_invented_citations_are_removed(fake_yf):
    llm = FakeLLM([llm_response("Profit rose 40% [[infosys-2025-p12-c0]].")])
    r = run_agent("Infosys profit?", client=llm, model="m")
    assert r["citations"] == []
    assert "[[" not in r["answer"]
    assert any("did not match" in w for w in r["warnings"])
    assert any("no verified citations" in w for w in r["warnings"])


def test_news_tool_withheld_when_not_requested(fake_yf):
    seen = {}

    def first(kw):
        seen["tools"] = [t["function"]["name"] for t in kw["tools"]]
        return llm_response(tool_calls=[tool_call("get_news", {"company": "Infosys"}, "n")])

    llm = FakeLLM([first, llm_response("No news requested.")])
    r = run_agent("Infosys revenue", include_news=False, client=llm, model="m")
    assert "get_news" not in seen["tools"]
    assert r["trace"][1]["status"] == "error"


def test_input_controls():
    assert run_agent("", client=FakeLLM([]))["status"] == "rejected"
    assert run_agent("x" * 5000, client=FakeLLM([]))["status"] == "rejected"
    assert run_agent("Tesla?", companies=["Tesla"], client=FakeLLM([]))["status"] == "rejected"


def test_llm_failure_is_controlled(fake_yf, monkeypatch):
    class APIConnectionError(Exception):
        pass

    def boom(**kw):
        raise APIConnectionError("down")

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    r = run_agent("Infosys revenue", client=client, model="m")
    assert r["status"] == "llm_error"


def test_missing_configuration_is_reported(monkeypatch):
    monkeypatch.setattr("config.GROQ_API_KEY", "")
    r = run_agent("Infosys revenue")
    assert r["status"] == "config_error"


def test_prompt_contains_safety_rules():
    from agent.prompts import system_prompt

    p = system_prompt()
    for phrase in ("UNTRUSTED DATA", "Never invent a source_id", "No buy/sell/hold", "calculate_named_metric"):
        assert phrase in p


@pytest.mark.parametrize("name,args,ok", [
    ("get_news", {"company": "Infosys", "limit": 3}, True),
    ("get_news", {"company": "Infosys", "limit": 99}, False),
    ("compare_companies", {"companies": ["Infosys"]}, False),
    ("compare_companies", {"companies": ["Infosys", "HDFC Bank"], "metrics": ["revenue"]}, True),
    ("calculate_named_metric", {"metric": "cagr", "inputs": {"start": 1, "end": 2, "years": 3}}, True),
    ("calculate_named_metric", {"metric": "cagr", "inputs": {"start": "1"}}, False),
    ("search_filings", {"company": "Infosys", "query": "q" * 600}, False),
])
def test_schema_validation(name, args, ok):
    assert (validate_arguments(name, args) is None) is ok


def test_tool_exception_becomes_structured_error(monkeypatch):
    def broken(**kw):
        raise RuntimeError("bug")

    monkeypatch.setitem(__import__("agent.tools").tools.TOOL_FUNCTIONS, "get_financials", broken)
    r = execute_registered_tool("get_financials", {"ticker": "INFY.NS"})
    assert r == {"status": "error", "error_code": "tool_exception", "error": "get_financials failed: RuntimeError"}
