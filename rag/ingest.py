"""Offline annual-report ingestion: PDF -> page text -> chunks -> ChromaDB.

Usage:
    python -m rag.ingest --company Infosys --pdf data/reports/infosys-ar-2025.pdf \
        --year 2025 --fy FY2025 --title "Infosys Integrated Annual Report 2024-25"
    python -m rag.ingest --manifest data/reports/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, resolve_company
from rag.store import get_collection

log = logging.getLogger(__name__)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalize_whitespace(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Character chunks within a single page, preferring whitespace boundaries."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = end - overlap
        # snap forward to a word boundary so chunks don't start mid-word
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return [{page_index, page_label, text}] one page at a time."""
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            try:
                label = page.get_label() or None
            except Exception:
                label = None
            pages.append({
                "page_index": i,
                "page_label": label or str(i + 1),
                "text": normalize_whitespace(page.get_text("text")),
            })
    return pages


def build_records(pages: list[dict], company: str, report_year: int, financial_year: str,
                  report_title: str, source_file: str) -> tuple[list[str], list[str], list[dict]]:
    ids, docs, metas = [], [], []
    cslug = slug(company)
    for page in pages:
        for ci, chunk in enumerate(chunk_text(page["text"])):
            ids.append(f"{cslug}-{report_year}-p{page['page_index']}-c{ci}")
            docs.append(chunk)
            metas.append({
                "company": company,
                "report_year": int(report_year),
                "financial_year": financial_year,
                "report_title": report_title,
                "page_index": page["page_index"],
                "page_label": page["page_label"],
                "source_file": source_file,
                "chunk_index": ci,
            })
    return ids, docs, metas


def ingest_pdf(pdf_path: str | Path, company: str, report_year: int, financial_year: str,
               report_title: str | None = None, collection=None, batch_size: int = 64) -> dict:
    canonical = resolve_company(company)
    if not canonical:
        raise ValueError(f"Unsupported company: {company}")
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    title = report_title or f"{canonical} Annual Report {report_year}"
    pages = extract_pages(pdf_path)
    empty_pages = sum(1 for p in pages if len(p["text"]) < 50)
    ids, docs, metas = build_records(pages, canonical, report_year, financial_year, title, pdf_path.name)
    col = collection or get_collection()
    # Re-ingesting the same report replaces its previous chunks (stable IDs).
    col.delete(where={"$and": [{"company": canonical}, {"report_year": int(report_year)}]})
    for i in range(0, len(ids), batch_size):
        col.upsert(ids=ids[i:i + batch_size], documents=docs[i:i + batch_size],
                   metadatas=metas[i:i + batch_size])
    summary = {"company": canonical, "report_year": report_year, "pages": len(pages),
               "chunks": len(ids), "near_empty_pages": empty_pages, "source_file": pdf_path.name}
    if pages and empty_pages / len(pages) > 0.5:
        summary["warning"] = "Most pages had little text; the PDF may be scanned and need OCR."
    return summary


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Ingest annual-report PDFs into ChromaDB")
    p.add_argument("--manifest", help="JSON list of {company, pdf, year, fy, title}")
    p.add_argument("--company")
    p.add_argument("--pdf")
    p.add_argument("--year", type=int)
    p.add_argument("--fy")
    p.add_argument("--title")
    a = p.parse_args(argv)

    jobs = []
    if a.manifest:
        jobs = json.loads(Path(a.manifest).read_text())
    elif a.company and a.pdf and a.year:
        jobs = [{"company": a.company, "pdf": a.pdf, "year": a.year,
                 "fy": a.fy or f"FY{a.year}", "title": a.title}]
    else:
        p.error("provide --manifest or --company/--pdf/--year")

    for job in jobs:
        summary = ingest_pdf(job["pdf"], job["company"], job["year"], job.get("fy") or f"FY{job['year']}",
                             job.get("title"))
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
