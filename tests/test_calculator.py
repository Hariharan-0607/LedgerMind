import math

import pytest

from services.calculator import (CalculationError, calculate_named_metric, cagr, growth_rate,
                                 net_profit_margin, operating_margin, revenue_growth)


def test_operating_margin_matches_expected():
    assert operating_margin(1000, 210) == 21.0


def test_net_profit_margin_matches_expected():
    assert net_profit_margin(1_600_000, 264_000) == pytest.approx(16.5)


def test_revenue_growth_matches_expected():
    assert revenue_growth(1600, 1500) == pytest.approx(6.6667, abs=1e-4)


def test_growth_rate_handles_negative_previous():
    assert growth_rate(-5, -10) == 50.0


def test_cagr():
    assert cagr(100, 121, 2) == pytest.approx(10.0)


@pytest.mark.parametrize("func,args", [
    (operating_margin, (0, 10)),
    (net_profit_margin, (0, 10)),
    (revenue_growth, (10, 0)),
])
def test_zero_denominator_rejected(func, args):
    with pytest.raises(CalculationError):
        func(*args)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, "abc", True, 1e30, [1]])
def test_non_finite_and_invalid_rejected(bad):
    with pytest.raises(CalculationError):
        operating_margin(bad, 10)


def test_numeric_strings_with_commas_accepted():
    assert operating_margin("1,000", "250") == 25.0


def test_tool_entry_success_shape():
    r = calculate_named_metric("operating_margin", {"revenue": 200, "operating_income": 50})
    assert r == {"status": "success", "metric": "operating_margin", "value": 25.0, "unit": "%",
                 "inputs": {"revenue": 200, "operating_income": 50}}


def test_tool_entry_rejects_unknown_metric_and_expressions():
    assert calculate_named_metric("__import__('os')", {})["status"] == "error"
    r = calculate_named_metric("operating_margin", {"revenue": "2**9999", "operating_income": 1})
    assert r["status"] == "error"


def test_tool_entry_missing_and_unexpected_inputs():
    assert "Missing" in calculate_named_metric("revenue_growth", {"current": 1})["error"]
    r = calculate_named_metric("revenue_growth", {"current": 2, "previous": 1, "x": 3})
    assert "Unexpected" in r["error"]


def test_tool_entry_zero_division_is_structured():
    r = calculate_named_metric("net_profit_margin", {"revenue": 0, "net_income": 5})
    assert r["status"] == "error" and "zero" in r["error"]
