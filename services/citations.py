"""Citation registry and validation.

The LLM never writes source labels or page numbers. It may only reference
internal source IDs (e.g. ``[[infosys-2025-p112-c0]]``) that were returned by
tools during the run. The application maps those IDs to trusted metadata,
renders numbered citations, and drops anything it cannot verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MARKER_RE = re.compile(r"\[\[([A-Za-z0-9:._\-]{1,120})\]\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class EvidenceRegistry:
    """Run-scoped store of every source a tool actually returned."""

    sources: dict[str, dict] = field(default_factory=dict)

    def register(self, source_id: str, metadata: dict) -> None:
        if source_id:
            self.sources[source_id] = metadata

    def register_tool_result(self, tool_name: str, result: dict) -> None:
        """Extract verifiable sources from any tool's structured result."""
        if not isinstance(result, dict) or result.get("status") != "success":
            return
        if tool_name == "search_filings":
            for r in result.get("results", []):
                self.register(r["source_id"], {
                    "type": "annual_report",
                    "company": result.get("company"),
                    "report_title": r.get("report_title"),
                    "report_year": r.get("report_year"),
                    "financial_year": r.get("financial_year"),
                    "page_index": r.get("page_index"),
                    "page_label": r.get("page_label"),
                    "source_file": r.get("source_file"),
                    "text": r.get("text", ""),
                })
        elif tool_name == "get_news":
            for h in result.get("headlines", []):
                self.register(h["source_id"], {
                    "type": "news",
                    "company": result.get("company"),
                    "title": h.get("title"),
                    "published": h.get("published"),
                    "link": h.get("link"),
                    "publisher": h.get("publisher"),
                })
        elif tool_name in ("get_financials", "get_stock_price"):
            self._register_data(result)
        elif tool_name == "compare_companies":
            for item in result.get("companies", []):
                self._register_data(item.get("financials") or {})

    def _register_data(self, result: dict) -> None:
        if result.get("status") != "success" or "source_id" not in result:
            return
        sid = result["source_id"]
        if sid.startswith("price:"):
            meta = {"type": "price", "ticker": result["ticker"], "date": result.get("date"),
                    "source": result.get("source"), "retrieved_at": result.get("retrieved_at")}
        else:
            periods = [p.get("fiscal_year") for p in result.get("periods", [])]
            meta = {"type": "financial_statement", "ticker": result["ticker"],
                    "statement": result.get("statement"), "periods": periods,
                    "source": result.get("source"), "retrieved_at": result.get("retrieved_at")}
        self.register(sid, meta)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self.sources


def format_citation(meta: dict) -> str:
    t = meta.get("type")
    if t == "annual_report":
        page = meta.get("page_label") or (meta.get("page_index", 0) + 1)
        title = meta.get("report_title") or f"{meta.get('company')} Annual Report {meta.get('report_year')}"
        return f"{title}, p. {page}"
    if t == "news":
        pub = f" ({meta['publisher']})" if meta.get("publisher") else ""
        return f"\"{meta.get('title')}\"{pub}, {meta.get('published')}"
    if t == "price":
        return f"{meta.get('source')} session close for {meta.get('ticker')} on {meta.get('date')}"
    if t == "financial_statement":
        return (f"{meta.get('source')} {meta.get('statement')} for {meta.get('ticker')} "
                f"({', '.join(p for p in meta.get('periods', []) if p)}), retrieved {meta.get('retrieved_at')}")
    return str(meta)


def validate_and_render(answer: str, registry: EvidenceRegistry) -> dict:
    """Replace verified markers with [n], drop unknown ones, and build a claim map.

    Returns {answer, citations, claims, warnings, invalid_source_ids}.
    """
    order: dict[str, int] = {}
    invalid: list[str] = []

    def _sub(match: re.Match) -> str:
        sid = match.group(1)
        if sid not in registry:
            invalid.append(sid)
            return ""
        if sid not in order:
            order[sid] = len(order) + 1
        return f"[{order[sid]}]"

    # Build the claim map from the raw text first, so claims keep their IDs.
    claims = []
    for sentence in SENTENCE_RE.split(answer or ""):
        ids = MARKER_RE.findall(sentence)
        if ids:
            text = MARKER_RE.sub("", sentence).strip()
            claims.append({
                "claim": text,
                "source_ids": [i for i in ids if i in registry],
                "rejected_source_ids": [i for i in ids if i not in registry],
            })

    rendered = MARKER_RE.sub(_sub, answer or "")
    rendered = re.sub(r"[ \t]+([.,;:])", r"\1", rendered)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)

    citations = []
    for sid, n in order.items():
        meta = registry.sources[sid]
        citations.append({
            "number": n,
            "source_id": sid,
            "label": format_citation(meta),
            "type": meta.get("type"),
            "link": meta.get("link"),
            "page_index": meta.get("page_index"),
            "page_label": meta.get("page_label"),
            "excerpt": (meta.get("text") or "")[:600] or None,
        })

    warnings = []
    if invalid:
        warnings.append(
            f"Removed {len(invalid)} citation(s) that did not match any retrieved source: "
            f"{sorted(set(invalid))}"
        )
    for c in claims:
        if c["rejected_source_ids"] and not c["source_ids"]:
            warnings.append(f"Claim has no verified source: \"{c['claim'][:120]}\"")
    return {
        "answer": rendered.strip(),
        "citations": citations,
        "claims": claims,
        "warnings": warnings,
        "invalid_source_ids": sorted(set(invalid)),
    }


def citation_support_score(claim: str, evidence_text: str) -> float:
    """Cheap lexical support heuristic (token overlap) used in evaluation.

    This is NOT proof of support; human review remains the reference.
    """
    tok = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3}
    c, e = tok(claim), tok(evidence_text)
    return round(len(c & e) / len(c), 3) if c else 0.0
