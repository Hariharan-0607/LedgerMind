"""Hand-written tool-calling agent loop (no LangChain / LangGraph).

The loop is the only component that talks to the LLM. The LLM *requests*
tools; this code validates and executes them from a fixed registry, records a
trace, and validates citations before returning the final report.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import config
from agent.prompts import DISCLAIMER, system_prompt, user_message
from agent.tools import TOOL_SCHEMAS, execute_registered_tool
from services.citations import EvidenceRegistry, validate_and_render

log = logging.getLogger("agent")

MAX_TOOL_RESULT_CHARS = 8000
MAX_FAILURES_PER_TOOL = 2  # first attempt + one corrected retry


class AgentConfigError(RuntimeError):
    pass


def make_client():
    if not config.GROQ_API_KEY:
        raise AgentConfigError("GROQ_API_KEY is not set. Copy .env.example to .env and add a key.")
    if not config.MODEL:
        raise AgentConfigError("MODEL is not set. Choose a current tool-calling model from your provider.")
    from openai import OpenAI

    return OpenAI(api_key=config.GROQ_API_KEY, base_url=config.LLM_BASE_URL,
                  timeout=config.LLM_TIMEOUT_S, max_retries=0)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
                    "TimeoutError", "ConnectionError"}


@retry(retry=retry_if_exception(_is_transient), stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, max=8), reraise=True)
def call_llm(client, model: str, messages: list[dict], tools: list[dict]):
    return client.chat.completions.create(model=model, messages=messages, tools=tools,
                                          tool_choice="auto", temperature=0)


def _assistant_message_dict(msg) -> dict:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in msg.tool_calls
        ]
    return out


def _summarize(result: dict) -> str:
    if result.get("status") != "success":
        return f"error: {result.get('error', 'unknown')}"[:200]
    for key, label in (("periods", "periods"), ("results", "passages"),
                       ("headlines", "headlines"), ("chart_data", "rows")):
        if key in result:
            return f"{len(result[key])} {label}"
    if "close" in result:
        return f"close {result['close']} {result.get('currency', '')} on {result.get('date')}"
    if "value" in result:
        return f"{result['metric']} = {result['value']}{result.get('unit', '')}"
    return "ok"


def _collect_artifacts(artifacts: dict, name: str, result: dict) -> None:
    """Keep structured tool outputs so the UI can draw cards/charts without parsing prose."""
    if result.get("status") != "success":
        return
    if name == "get_financials":
        artifacts["financials"][result["ticker"]] = result
    elif name == "get_stock_price":
        artifacts["prices"][result["ticker"]] = result
    elif name == "calculate_named_metric":
        artifacts["metrics"].append(result)
    elif name == "search_filings":
        artifacts["evidence"].extend({**r, "company": result["company"]} for r in result["results"])
    elif name == "get_news":
        artifacts["news"].extend({**h, "company": result["company"]} for h in result["headlines"])
    elif name == "compare_companies":
        artifacts["comparison"] = {k: result[k] for k in
                                   ("aligned_periods", "metrics", "chart_data", "comparability_caveats", "warnings")}
        for c in result["companies"]:
            if c["financials"].get("status") == "success":
                artifacts["financials"][c["financials"]["ticker"]] = c["financials"]


def run_agent(question: str, companies: list[str] | None = None, include_news: bool = True,
              client=None, model: str | None = None, max_steps: int | None = None,
              run_id: str | None = None, today: date | None = None, on_event=None) -> dict:
    """Run one research question end to end. Never raises for tool/model failures."""
    run_id = run_id or uuid.uuid4().hex[:12]
    max_steps = max_steps or config.MAX_STEPS
    started = time.perf_counter()
    trace: list[dict] = []
    warnings: list[str] = []
    registry = EvidenceRegistry()
    failures: dict[str, int] = {}
    artifacts = {"financials": {}, "prices": {}, "metrics": [], "evidence": [], "news": [], "comparison": None}

    def emit(event: dict) -> None:
        trace.append(event)
        if on_event:
            try:
                on_event(event)
            except Exception:
                pass

    def finish(status: str, answer: str, citations=None, claims=None) -> dict:
        result = {
            "run_id": run_id, "status": status, "question": question, "answer": answer,
            "citations": citations or [], "claims": claims or [], "trace": trace,
            "warnings": warnings, "artifacts": artifacts, "disclaimer": DISCLAIMER,
            "duration_s": round(time.perf_counter() - started, 3),
            "tool_calls": sum(1 for t in trace if t["type"] == "tool_call"),
            "tool_errors": sum(1 for t in trace if t["type"] == "tool_call" and t["status"] != "success"),
        }
        log.info("run=%s status=%s duration=%.2fs tool_calls=%d", run_id, status,
                 result["duration_s"], result["tool_calls"])
        return result

    if not isinstance(question, str) or not question.strip():
        return finish("rejected", "Question is empty.")
    if len(question) > config.MAX_QUESTION_CHARS:
        return finish("rejected", f"Question exceeds {config.MAX_QUESTION_CHARS} characters.")
    if companies:
        unknown = [c for c in companies if not config.resolve_company(c)]
        if unknown:
            return finish("rejected", f"Unsupported companies: {unknown}. Supported: {sorted(config.COMPANIES)}")
        companies = [config.resolve_company(c) for c in companies]

    try:
        client = client or make_client()
    except AgentConfigError as exc:
        warnings.append(str(exc))
        return finish("config_error", str(exc))
    model = model or config.MODEL or "unspecified"

    tools = TOOL_SCHEMAS if include_news else [t for t in TOOL_SCHEMAS if t["function"]["name"] != "get_news"]
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(today)},
        {"role": "user", "content": user_message(question.strip(), companies, include_news)},
    ]

    for step in range(max_steps):
        t0 = time.perf_counter()
        try:
            response = call_llm(client, model, messages, tools)
        except Exception as exc:
            emit({"type": "llm_call", "step": step, "status": "error", "error": type(exc).__name__,
                  "duration_s": round(time.perf_counter() - t0, 3)})
            warnings.append(f"LLM call failed: {type(exc).__name__}")
            return finish("llm_error", "The language model could not be reached. No report was generated.")
        emit({"type": "llm_call", "step": step, "status": "success",
              "duration_s": round(time.perf_counter() - t0, 3)})

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            validated = validate_and_render(msg.content or "", registry)
            warnings.extend(validated["warnings"])
            if not validated["citations"]:
                warnings.append("The report contains no verified citations.")
            return finish("completed", validated["answer"], validated["citations"], validated["claims"])

        messages.append(_assistant_message_dict(msg))
        for tc in tool_calls:  # every tool call gets exactly one tool result
            name = tc.function.name
            t1 = time.perf_counter()
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if args is None:
                result = {"status": "error", "error_code": "malformed_arguments",
                          "error": "Arguments were not valid JSON"}
            elif name == "get_news" and not include_news:
                result = {"status": "error", "error_code": "not_permitted",
                          "error": "News was not requested for this run"}
            elif failures.get(name, 0) >= MAX_FAILURES_PER_TOOL:
                result = {"status": "error", "error_code": "retry_limit",
                          "error": f"{name} already failed {MAX_FAILURES_PER_TOOL} times; "
                                   "report the data as unavailable"}
            else:
                result = execute_registered_tool(name, args)

            status = result.get("status", "error")
            if status != "success":
                failures[name] = failures.get(name, 0) + 1
            registry.register_tool_result(name, result)
            _collect_artifacts(artifacts, name, result)
            emit({"type": "tool_call", "step": step, "tool": name,
                  "arguments": args if isinstance(args, dict) else tc.function.arguments,
                  "status": status, "summary": _summarize(result),
                  "duration_s": round(time.perf_counter() - t1, 3)})

            content = json.dumps({"tool": name, "untrusted_data": result}, default=str)
            if len(content) > MAX_TOOL_RESULT_CHARS:
                content = content[:MAX_TOOL_RESULT_CHARS] + '..."[truncated]"'
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    warnings.append(f"Stopped after reaching the maximum of {max_steps} steps.")
    return finish("step_limit", "The agent reached its step limit before producing a final report. "
                                "Try a narrower question.")
