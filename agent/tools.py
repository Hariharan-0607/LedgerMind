"""Fixed tool registry: functions, JSON schemas, and argument validation.

The model can only request tools listed here. Arguments are validated against
the schema before execution; nothing model-generated is ever eval'd.
"""

from __future__ import annotations

from typing import Any, Callable

from config import COMPANIES
from rag.retrieve import search_filings
from services.calculator import METRICS, calculate_named_metric
from services.compare import SUPPORTED_METRICS, compare_companies
from services.financial_data import get_financials, get_stock_price
from services.news import get_news

COMPANY_ENUM = sorted(COMPANIES)
TICKER_ENUM = sorted(info["ticker"] for info in COMPANIES.values())

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Latest available trading-session close for a supported NSE ticker. "
                           "Not a real-time quote.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "enum": TICKER_ENUM}},
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": "Annual revenue, operating income and net income for the last reported "
                           "financial years, with currency, periods and source metadata.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "enum": TICKER_ENUM}},
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_named_metric",
            "description": "Deterministic calculation. Inputs per metric: "
                           "operating_margin{revenue, operating_income}; net_profit_margin{revenue, net_income}; "
                           "revenue_growth{current, previous}; growth_rate{current, previous}; "
                           "cagr{start, end, years}. Use raw numbers from get_financials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": sorted(METRICS)},
                    "inputs": {"type": "object", "additionalProperties": {"type": "number"}},
                },
                "required": ["metric", "inputs"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": "Search ingested annual reports of one company for explanations, strategy, "
                           "risks and management commentary. Returns passages with source_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "enum": COMPANY_ENUM},
                    "query": {"type": "string", "maxLength": 500},
                },
                "required": ["company", "query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Recent headlines (title, date, link only) for a supported company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "enum": COMPANY_ENUM},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["company"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": "Compare 2-3 supported companies on aligned financial years. Returns chart-ready "
                           "data and comparability caveats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "companies": {"type": "array", "items": {"type": "string", "enum": COMPANY_ENUM},
                                  "minItems": 2, "maxItems": 3},
                    "metrics": {"type": "array", "items": {"type": "string", "enum": sorted(SUPPORTED_METRICS)}},
                },
                "required": ["companies"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "get_stock_price": get_stock_price,
    "get_financials": get_financials,
    "calculate_named_metric": calculate_named_metric,
    "search_filings": search_filings,
    "get_news": get_news,
    "compare_companies": compare_companies,
}

_SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}
_PY_TYPES = {"string": str, "integer": int, "number": (int, float), "array": list, "object": dict}


def _check(value: Any, schema: dict, path: str) -> str | None:
    expected = schema.get("type")
    if expected:
        py = _PY_TYPES[expected]
        if isinstance(value, bool) or not isinstance(value, py):
            return f"'{path}' must be of type {expected}"
    if "enum" in schema and value not in schema["enum"]:
        return f"'{path}' must be one of {schema['enum']}"
    if expected == "string" and len(value) > schema.get("maxLength", 2000):
        return f"'{path}' is too long"
    if expected == "integer":
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            return f"'{path}' is out of range"
    if expected == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 50):
            return f"'{path}' has an invalid number of items"
        for i, item in enumerate(value):
            if (err := _check(item, schema.get("items", {}), f"{path}[{i}]")):
                return err
    if expected == "object":
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                return f"missing required argument '{req}'"
        for key, item in value.items():
            if key in props:
                if (err := _check(item, props[key], f"{key}" if path == "$" else f"{path}.{key}")):
                    return err
            elif schema.get("additionalProperties") is False:
                return f"unexpected argument '{key}'"
            elif isinstance(schema.get("additionalProperties"), dict):
                if (err := _check(item, schema["additionalProperties"], f"{path}.{key}")):
                    return err
    return None


def validate_arguments(tool_name: str, args: Any) -> str | None:
    """Return an error message, or None if the arguments are valid."""
    if tool_name not in _SCHEMA_BY_NAME:
        return f"Unknown tool '{tool_name}'"
    if not isinstance(args, dict):
        return "Arguments must be a JSON object"
    return _check(args, _SCHEMA_BY_NAME[tool_name], "$")


def execute_registered_tool(tool_name: str, args: dict) -> dict:
    """Validate then dispatch. Never raises; failures become structured errors."""
    err = validate_arguments(tool_name, args)
    if err:
        return {"status": "error", "error_code": "invalid_arguments", "error": err}
    try:
        return TOOL_FUNCTIONS[tool_name](**args)
    except Exception as exc:  # tool bugs must not crash the run
        return {"status": "error", "error_code": "tool_exception",
                "error": f"{tool_name} failed: {type(exc).__name__}"}
