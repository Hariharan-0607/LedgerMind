"""Multi-company comparison built on validated data and calculation functions."""

from __future__ import annotations

from config import COMPANIES, resolve_company
from services.calculator import calculate_named_metric
from services.financial_data import get_financials

RAW_METRICS = {"revenue", "operating_income", "net_income"}
DERIVED_METRICS = {"operating_margin", "net_profit_margin", "revenue_growth"}
SUPPORTED_METRICS = RAW_METRICS | DERIVED_METRICS


def _derive(metric: str, period: dict, previous: dict | None):
    if metric == "operating_margin":
        args = {"revenue": period.get("revenue"), "operating_income": period.get("operating_income")}
    elif metric == "net_profit_margin":
        args = {"revenue": period.get("revenue"), "net_income": period.get("net_income")}
    else:  # revenue_growth
        if previous is None:
            return None, "no prior period"
        args = {"current": period.get("revenue"), "previous": previous.get("revenue")}
    if any(v is None for v in args.values()):
        return None, "missing input"
    r = calculate_named_metric(metric, args)
    return (r["value"], None) if r["status"] == "success" else (None, r["error"])


def compare_companies(companies: list[str], metrics: list[str] | None = None) -> dict:
    if not isinstance(companies, list) or not 2 <= len(companies) <= 3:
        return {"status": "error", "error": "Provide between 2 and 3 companies to compare"}
    metrics = metrics or ["revenue", "net_income", "net_profit_margin", "revenue_growth"]
    bad = [m for m in metrics if m not in SUPPORTED_METRICS]
    if bad:
        return {"status": "error", "error": f"Unsupported metrics {bad}. Supported: {sorted(SUPPORTED_METRICS)}"}

    canon = []
    for c in companies:
        name = resolve_company(c)
        if not name:
            return {"status": "error", "error": f"Unsupported company '{c}'. Supported: {sorted(COMPANIES)}"}
        if name not in canon:
            canon.append(name)
    if len(canon) < 2:
        return {"status": "error", "error": "Provide at least two distinct companies"}

    per_company, warnings = [], []
    for name in canon:
        fin = get_financials(COMPANIES[name]["ticker"], periods=4)
        per_company.append({"company": name, "sector": COMPANIES[name]["sector"], "financials": fin})
        if fin.get("status") != "success":
            warnings.append(f"{name}: financial data unavailable ({fin.get('error')})")

    ok = [c for c in per_company if c["financials"].get("status") == "success"]
    if len(ok) < 2:
        return {"status": "error", "error": "Fewer than two companies have usable data",
                "warnings": warnings}

    currencies = {c["financials"]["currency"] for c in ok}
    if len(currencies) > 1:
        warnings.append(f"Currencies differ across companies: {sorted(currencies)}. Values are not directly comparable.")

    # Align on fiscal years present for every company.
    year_sets = [{p["fiscal_year"] for p in c["financials"]["periods"]} for c in ok]
    common = sorted(set.intersection(*year_sets), reverse=True)[:3]
    if not common:
        return {"status": "error", "error": "No common reporting periods across companies",
                "warnings": warnings}
    dropped = sorted(set.union(*year_sets) - set(common))
    if dropped:
        warnings.append(f"Periods not available for all companies were excluded: {dropped}")

    chart_rows = []  # long format: {company, fiscal_year, metric, value, unit}
    for c in ok:
        periods = {p["fiscal_year"]: p for p in c["financials"]["periods"]}
        ordered = sorted(periods)  # ascending FY
        for fy in common:
            prev_fy = ordered[ordered.index(fy) - 1] if ordered.index(fy) > 0 else None
            for m in metrics:
                note = None
                if m in RAW_METRICS:
                    value, unit = periods[fy].get(m), c["financials"]["currency"]
                    if value is None:
                        note = "missing in source"
                else:
                    value, note = _derive(m, periods[fy], periods.get(prev_fy) if prev_fy else None)
                    unit = "%"
                chart_rows.append({"company": c["company"], "fiscal_year": fy, "metric": m,
                                   "value": value, "unit": unit, "note": note})

    missing = [r for r in chart_rows if r["value"] is None]
    if missing:
        warnings.append(f"{len(missing)} metric value(s) missing; shown as null, never substituted with zero.")

    sectors = {c["sector"] for c in ok}
    caveats = []
    if len(sectors) > 1:
        caveats.append(f"Companies are in different sectors ({', '.join(sorted(sectors))}); "
                       "business models and statement structures differ.")
    if any(c["sector"] == "Banking" for c in ok):
        caveats.append("Bank revenue and operating income are not comparable to non-financial companies; "
                       "margin comparisons involving a bank can be misleading. Banking metrics such as "
                       "net interest margin are not yet supported.")

    return {
        "status": "success",
        "companies": per_company,
        "aligned_periods": common,
        "metrics": metrics,
        "chart_data": chart_rows,
        "comparability_caveats": caveats,
        "warnings": warnings,
    }
