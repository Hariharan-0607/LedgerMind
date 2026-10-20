"""Deterministic, named financial calculations.

The model never supplies expressions — only a metric name plus numeric inputs.
No eval(), no arbitrary syntax.
"""

from __future__ import annotations

import math
from numbers import Real

MAX_ABS_VALUE = 1e18  # guards against absurd / overflowing inputs


class CalculationError(ValueError):
    pass


def _num(name: str, value) -> float:
    if isinstance(value, bool) or value is None:
        raise CalculationError(f"'{name}' is missing or not numeric")
    if isinstance(value, str):
        try:
            value = float(value.replace(",", "").strip())
        except ValueError as exc:
            raise CalculationError(f"'{name}' is not a valid number") from exc
    if not isinstance(value, Real):
        raise CalculationError(f"'{name}' is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise CalculationError(f"'{name}' is not finite")
    if abs(value) > MAX_ABS_VALUE:
        raise CalculationError(f"'{name}' exceeds the supported magnitude")
    return value


def _ratio_pct(numerator: float, denominator: float, denom_name: str) -> float:
    if denominator == 0:
        raise CalculationError(f"'{denom_name}' is zero; division by zero")
    result = numerator / denominator * 100
    if not math.isfinite(result):
        raise CalculationError("result is not finite")
    return round(result, 4)


def operating_margin(revenue, operating_income) -> float:
    return _ratio_pct(_num("operating_income", operating_income), _num("revenue", revenue), "revenue")


def net_profit_margin(revenue, net_income) -> float:
    return _ratio_pct(_num("net_income", net_income), _num("revenue", revenue), "revenue")


def revenue_growth(current, previous) -> float:
    cur, prev = _num("current", current), _num("previous", previous)
    if prev < 0:
        raise CalculationError("'previous' is negative; growth rate is not meaningful")
    return _ratio_pct(cur - prev, prev, "previous")


def growth_rate(current, previous) -> float:
    """Generic period-over-period growth (e.g. net income growth)."""
    cur, prev = _num("current", current), _num("previous", previous)
    if prev == 0:
        raise CalculationError("'previous' is zero; division by zero")
    # Use |previous| so a move from -10 to -5 reads as an improvement.
    return round((cur - prev) / abs(prev) * 100, 4)


def cagr(start, end, years) -> float:
    s, e, n = _num("start", start), _num("end", end), _num("years", years)
    if s <= 0 or e <= 0:
        raise CalculationError("CAGR requires positive start and end values")
    if n <= 0 or n > 100:
        raise CalculationError("'years' must be between 0 and 100")
    return round(((e / s) ** (1 / n) - 1) * 100, 4)


# Registry: metric name -> (function, ordered required input names, unit)
METRICS: dict[str, tuple] = {
    "operating_margin": (operating_margin, ("revenue", "operating_income"), "%"),
    "net_profit_margin": (net_profit_margin, ("revenue", "net_income"), "%"),
    "revenue_growth": (revenue_growth, ("current", "previous"), "%"),
    "growth_rate": (growth_rate, ("current", "previous"), "%"),
    "cagr": (cagr, ("start", "end", "years"), "%"),
}


def calculate_named_metric(metric: str, inputs: dict) -> dict:
    """Tool entry point. Always returns a structured dict, never raises."""
    if metric not in METRICS:
        return {
            "status": "error",
            "error": f"Unsupported metric '{metric}'. Supported: {sorted(METRICS)}",
        }
    if not isinstance(inputs, dict):
        return {"status": "error", "error": "'inputs' must be an object"}
    func, required, unit = METRICS[metric]
    missing = [k for k in required if k not in inputs]
    if missing:
        return {"status": "error", "error": f"Missing inputs for {metric}: {missing}"}
    unexpected = [k for k in inputs if k not in required]
    if unexpected:
        return {"status": "error", "error": f"Unexpected inputs for {metric}: {unexpected}"}
    try:
        value = func(*(inputs[k] for k in required))
    except CalculationError as exc:
        return {"status": "error", "metric": metric, "error": str(exc)}
    return {
        "status": "success",
        "metric": metric,
        "value": value,
        "unit": unit,
        "inputs": {k: inputs[k] for k in required},
    }
