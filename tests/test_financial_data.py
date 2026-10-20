from services.compare import compare_companies
from services.financial_data import get_financials, get_stock_price, normalize_ticker


def test_valid_ticker_returns_dated_price(fake_yf):
    r = get_stock_price("INFY.NS")
    assert r["status"] == "success"
    assert r["date"] == "2026-10-17"
    assert r["close"] == 1523.25
    assert r["currency"] == "INR"
    assert r["price_type"] == "session_close"
    assert r["source_id"] == "price:INFY.NS:2026-10-17"


def test_company_name_resolves_to_ticker(fake_yf):
    assert get_stock_price("Infosys")["ticker"] == "INFY.NS"
    assert normalize_ticker("hdfc bank") == ("HDFCBANK.NS", None)


def test_invalid_ticker_returns_controlled_error(fake_yf):
    r = get_stock_price("FAKE.NS")
    assert r["status"] == "error" and r["error_code"] == "invalid_ticker"
    assert get_financials("")["status"] == "error"


def test_empty_history_is_handled(fake_yf):
    fake_yf["empty"] = True
    assert get_stock_price("INFY.NS")["error_code"] == "no_data"
    assert get_financials("INFY.NS")["error_code"] == "no_data"


def test_provider_failure_is_structured_after_retries(fake_yf):
    fake_yf["fail"] = True
    r = get_stock_price("INFY.NS")
    assert r["status"] == "error" and "TransientProviderError" in r["error"]


def test_financials_shape_and_fiscal_years(fake_yf):
    r = get_financials("INFY.NS")
    assert r["status"] == "success"
    assert [p["fiscal_year"] for p in r["periods"]] == ["FY2025", "FY2024", "FY2023"]
    assert r["periods"][0]["revenue"] == 1_600_000_000_000
    assert r["labels_used"]["revenue"] == "Total Revenue"
    assert r["warnings"] == []


def test_missing_statement_fields_are_null_not_zero(fake_yf):
    r = get_financials("HDFCBANK.NS")
    assert r["status"] == "success"
    assert all(p["operating_income"] is None for p in r["periods"])
    assert any("operating income" in w.lower() for w in r["warnings"])


def test_cached_results_are_labelled(fake_yf):
    get_stock_price("INFY.NS")
    again = get_stock_price("INFY.NS")
    assert again["cached"] is True and "cached_at" in again


def test_compare_aligns_periods_and_flags_bank(fake_yf):
    r = compare_companies(["Infosys", "HDFC Bank"], ["revenue", "operating_margin", "revenue_growth"])
    assert r["status"] == "success"
    assert r["aligned_periods"] == ["FY2025", "FY2024", "FY2023"]
    bank_margin = [x for x in r["chart_data"] if x["company"] == "HDFC Bank" and x["metric"] == "operating_margin"]
    assert all(x["value"] is None for x in bank_margin)  # never substituted with zero
    assert any("bank" in c.lower() for c in r["comparability_caveats"])
    infy_growth = next(x for x in r["chart_data"] if x["company"] == "Infosys"
                       and x["metric"] == "revenue_growth" and x["fiscal_year"] == "FY2025")
    assert round(infy_growth["value"], 2) == 6.67
    # oldest period has no prior year -> null growth
    oldest = next(x for x in r["chart_data"] if x["company"] == "Infosys"
                  and x["metric"] == "revenue_growth" and x["fiscal_year"] == "FY2023")
    assert oldest["value"] is None


def test_compare_rejects_bad_input(fake_yf):
    assert compare_companies(["Infosys"])["status"] == "error"
    assert compare_companies(["Infosys", "Tesla"])["status"] == "error"
    assert compare_companies(["Infosys", "HDFC Bank"], ["ebitda"])["status"] == "error"
