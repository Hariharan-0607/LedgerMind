"""Run the fixed evaluation set and report real, reproducible metrics.

Usage:
    python -m eval.run_eval                      # everything that can run
    python -m eval.run_eval --mode direct        # tool-level tasks only (no LLM needed)
    python -m eval.run_eval --category arithmetic
    python -m eval.run_eval --retrieval-only

Tasks that cannot run (no LLM key, reports not ingested, unverified retrieval
pairs) are reported as SKIPPED — never as passes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from agent.loop import run_agent  # noqa: E402
from agent.tools import execute_registered_tool  # noqa: E402
from rag.retrieve import search_filings  # noqa: E402
from services.citations import citation_support_score  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
SUPPORT_THRESHOLD = 0.3
FY_RE = re.compile(r"^FY\d{4}$")
NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


# ------------------------------------------------------------ helpers
def _tool_numbers(run: dict) -> list[float]:
    """Every numeric value a tool produced in this run, in common display scales."""
    art = run.get("artifacts") or {}
    raw = []
    for fin in (art.get("financials") or {}).values():
        for p in fin.get("periods", []):
            raw += [p[m] for m in ("revenue", "operating_income", "net_income") if p.get(m) is not None]
    raw += [m["value"] for m in art.get("metrics", [])]
    raw += [p["close"] for p in (art.get("prices") or {}).values()]
    for row in (art.get("comparison") or {}).get("chart_data", []):
        if row.get("value") is not None:
            raw.append(row["value"])
    scales = (1, 1e5, 1e7, 1e9, 1e12, 1e6)  # units, lakh, crore, billion, lakh crore, million
    return [v / s for v in raw for s in scales]


def numbers_match_score(run: dict) -> float | None:
    """Fraction of material numbers in the answer that match a tool value within 1%.

    Ignores years, FY labels, citation numbers and small integers (< 10).
    """
    text = re.sub(r"\[\d+\]", " ", run.get("answer", ""))
    text = re.sub(r"\bFY\s?\d{2,4}\b|\b(19|20)\d{2}(-\d{2,4})?\b", " ", text)
    candidates = []
    for m in NUM_RE.findall(text):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        if abs(v) >= 10 or (("." in m) and abs(v) > 0):
            candidates.append(v)
    if not candidates:
        return None
    pool = _tool_numbers(run)
    ok = sum(1 for v in candidates if any(abs(v - t) <= max(0.01 * abs(t), 0.051) for t in pool))
    return round(ok / len(candidates), 3)


def _reports_available(company: str) -> bool:
    r = search_filings(company, "annual report", k=1)
    return r.get("status") == "success"


def check_direct(task: dict, result: dict) -> list[str]:
    e, fails = task["expect"], []
    if result.get("status") != e["status"]:
        fails.append(f"status {result.get('status')} != {e['status']} ({result.get('error', '')})")
        return fails
    if "error_code" in e and result.get("error_code") != e["error_code"]:
        fails.append(f"error_code {result.get('error_code')} != {e['error_code']}")
    if "value" in e and abs(result.get("value", float("inf")) - e["value"]) > e.get("tolerance", 0.01):
        fails.append(f"value {result.get('value')} != {e['value']}")
    for f in e.get("fields_present", []):
        if result.get(f) in (None, ""):
            fails.append(f"missing field {f}")
    if "periods_count" in e and len(result.get("periods", [])) != e["periods_count"]:
        fails.append(f"periods {len(result.get('periods', []))} != {e['periods_count']}")
    if e.get("fiscal_year_format") and not all(FY_RE.match(p["fiscal_year"]) for p in result.get("periods", [])):
        fails.append("fiscal_year labels malformed")
    if "price_type" in e and result.get("price_type") != e["price_type"]:
        fails.append("price_type mismatch")
    if "caveat_contains" in e:
        caveats = " ".join(result.get("comparability_caveats", [])).lower()
        if e["caveat_contains"] not in caveats:
            fails.append(f"no caveat mentioning '{e['caveat_contains']}'")
    if "aligned_periods_min" in e and len(result.get("aligned_periods", [])) < e["aligned_periods_min"]:
        fails.append("too few aligned periods")
    return fails


def check_agent(task: dict, run: dict) -> tuple[list[str], dict]:
    e, fails, info = task["expect"], [], {}
    answer = run.get("answer", "")
    called = [t["tool"] for t in run.get("trace", []) if t["type"] == "tool_call"]
    info["tools_called"] = called
    if "run_status" in e and run["status"] != e["run_status"]:
        fails.append(f"run status {run['status']}")
    if "run_status_in" in e and run["status"] not in e["run_status_in"]:
        fails.append(f"run status {run['status']}")
    for t in e.get("tools_called_include", []):
        if t not in called:
            fails.append(f"expected tool {t} not called")
    if e.get("tools_called_include_any") and not set(e["tools_called_include_any"]) & set(called):
        fails.append(f"none of {e['tools_called_include_any']} called")
    for t in e.get("tools_not_called", []):
        if t in called:
            fails.append(f"unnecessary tool {t} called")
    if "max_tool_calls" in e and len(called) > e["max_tool_calls"]:
        fails.append(f"{len(called)} tool calls > {e['max_tool_calls']}")
    if e.get("answer_contains_any") and not any(s.lower() in answer.lower() for s in e["answer_contains_any"]):
        fails.append(f"answer lacks any of {e['answer_contains_any']}")
    for s in e.get("answer_not_contains", []):
        if s.lower() in answer.lower():
            fails.append(f"answer contains forbidden '{s}'")
    cits = run.get("citations", [])
    if len(cits) < e.get("min_citations", 0):
        fails.append(f"{len(cits)} citations < {e['min_citations']}")
    if "max_citations" in e and len(cits) > e["max_citations"]:
        fails.append("unexpected citations")
    doc_cits = [c for c in cits if c.get("type") == "annual_report"]
    if len(doc_cits) < e.get("min_doc_citations", 0):
        fails.append(f"{len(doc_cits)} annual-report citations < {e['min_doc_citations']}")
    rendered = len(cits) + len(run.get("_invalid", []))
    info["citation_validity"] = (len(cits) / rendered) if rendered else None
    if "citation_validity" in e and info["citation_validity"] is not None and info["citation_validity"] < e["citation_validity"]:
        fails.append(f"citation validity {info['citation_validity']:.2f}")
    if e.get("numbers_match_tools"):
        score = numbers_match_score(run)
        info["numbers_match"] = score
        if score is None:
            fails.append("no numbers in answer")
        elif score < 0.8:
            fails.append(f"only {score:.0%} of numbers match tool outputs")
    return fails, info


def run_retrieval() -> dict:
    items = json.loads((EVAL_DIR / "retrieval_set.json").read_text())
    verified = [i for i in items if i.get("verified") and i.get("expected_page_labels")]
    rows, hits = [], 0
    for item in verified:
        r = search_filings(item["company"], item["query"], k=4)
        got = [str(x["page_label"]) for x in r.get("results", [])]
        hit = bool(set(got) & {str(p) for p in item["expected_page_labels"]})
        hits += hit
        rows.append({"id": item["id"], "hit": hit, "retrieved_pages": got,
                     "expected": item["expected_page_labels"], "status": r.get("status")})
    return {"total_pairs": len(items), "verified_pairs": len(verified),
            "hit_rate_at_4": round(hits / len(verified), 3) if verified else None, "rows": rows}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


# ------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "agent"])
    ap.add_argument("--category")
    ap.add_argument("--retrieval-only", action="store_true")
    a = ap.parse_args(argv)

    started = datetime.now(timezone.utc)
    RESULTS_DIR.mkdir(exist_ok=True)
    report: dict = {"started_at": started.isoformat(timespec="seconds"), "model": config.MODEL or None}

    if not a.retrieval_only:
        tasks = json.loads((EVAL_DIR / "tasks.json").read_text())
        tasks = [t for t in tasks if (not a.mode or t["mode"] == a.mode)
                 and (not a.category or t["category"] == a.category)]
        llm_ready = bool(config.GROQ_API_KEY and config.MODEL)
        results, latencies, tool_calls, tool_errors = [], [], 0, 0
        validity_num = validity_den = 0
        support_rows = []
        for t in tasks:
            row = {"id": t["id"], "category": t["category"], "mode": t["mode"]}
            if t["mode"] == "agent" and not llm_ready:
                row.update(outcome="SKIPPED", reason="LLM not configured")
            elif any(not _reports_available(c) for c in t.get("requires_reports", [])):
                row.update(outcome="SKIPPED", reason="annual report not ingested")
            else:
                t0 = time.perf_counter()
                if t["mode"] == "direct":
                    res = execute_registered_tool(t["tool"], t["args"])
                    fails = check_direct(t, res)
                    tool_calls += 1
                    tool_errors += res.get("status") != "success" and t["expect"]["status"] == "success"
                    info = {}
                else:
                    res = run_agent(t["question"], companies=t.get("companies"),
                                    include_news=t.get("include_news", True))
                    invalid = [sid for c in res.get("claims", []) for sid in c.get("rejected_source_ids", [])]
                    res["_invalid"] = invalid
                    fails, info = check_agent(t, res)
                    tool_calls += res.get("tool_calls", 0)
                    tool_errors += res.get("tool_errors", 0)
                    validity_num += len(res.get("citations", []))
                    validity_den += len(res.get("citations", [])) + len(invalid)
                    for claim in res.get("claims", []):
                        for sid in claim["source_ids"]:
                            c = next((c for c in res["citations"] if c["source_id"] == sid), None)
                            if c and c.get("excerpt"):
                                s = citation_support_score(claim["claim"], c["excerpt"])
                                support_rows.append({"task": t["id"], "claim": claim["claim"], "source_id": sid,
                                                     "heuristic_score": s, "heuristic_supported": s >= SUPPORT_THRESHOLD,
                                                     "human_label": None})
                    row["answer_excerpt"] = res.get("answer", "")[:400]
                dur = time.perf_counter() - t0
                latencies.append(dur)
                row.update(outcome="PASS" if not fails else "FAIL", failures=fails,
                           duration_s=round(dur, 3), **info)
            results.append(row)
            print(f"{row['outcome']:7} {t['id']:10} {t['category']:16} {'; '.join(row.get('failures', [])) or row.get('reason', '')}")

        executed = [r for r in results if r["outcome"] != "SKIPPED"]
        passed = [r for r in executed if r["outcome"] == "PASS"]
        num_tasks = [r for r in executed if r["category"] == "arithmetic" or "numbers_match" in r]
        num_ok = [r for r in num_tasks if r["outcome"] == "PASS"]
        by_cat: dict[str, dict] = {}
        for r in results:
            c = by_cat.setdefault(r["category"], {"pass": 0, "fail": 0, "skipped": 0})
            c[{"PASS": "pass", "FAIL": "fail", "SKIPPED": "skipped"}[r["outcome"]]] += 1
        report["tasks"] = {
            "total": len(results), "executed": len(executed), "skipped": len(results) - len(executed),
            "passed": len(passed),
            "task_success_rate": round(len(passed) / len(executed), 3) if executed else None,
            "numerical_correctness": round(len(num_ok) / len(num_tasks), 3) if num_tasks else None,
            "citation_validity": round(validity_num / validity_den, 3) if validity_den else None,
            "citation_support_heuristic": (round(sum(r["heuristic_supported"] for r in support_rows)
                                                 / len(support_rows), 3) if support_rows else None),
            "tool_error_rate": round(tool_errors / tool_calls, 3) if tool_calls else None,
            "latency_median_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": percentile(latencies, 0.95),
            "by_category": by_cat,
            "results": results,
        }
        if support_rows:
            review = RESULTS_DIR / f"citation_review_{started:%Y%m%dT%H%M%S}.json"
            review.write_text(json.dumps(support_rows, indent=1))
            report["tasks"]["citation_review_file"] = review.name

    report["retrieval"] = run_retrieval()
    out = RESULTS_DIR / f"eval_{started:%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, default=str))

    print("\n=== Summary ===")
    if "tasks" in report:
        s = report["tasks"]
        for k in ("total", "executed", "skipped", "passed", "task_success_rate", "numerical_correctness",
                  "citation_validity", "citation_support_heuristic", "tool_error_rate",
                  "latency_median_s", "latency_p95_s"):
            print(f"{k:28} {s[k]}")
    r = report["retrieval"]
    print(f"{'retrieval_hit_rate_at_4':28} {r['hit_rate_at_4']} ({r['verified_pairs']}/{r['total_pairs']} pairs verified)")
    print(f"\nSaved {out.relative_to(EVAL_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
