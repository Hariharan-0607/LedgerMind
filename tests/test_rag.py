from pathlib import Path

import pytest

from rag.ingest import build_records, chunk_text, ingest_pdf, normalize_whitespace
from rag.retrieve import search_filings
from services.citations import EvidenceRegistry, validate_and_render

PAGES = {
    "Infosys": [
        "Chairman's letter. Our digital transformation deals and large deal wins drove revenue growth.",
        "Operating margin improved due to Project Maximus cost optimisation and better utilisation.",
        "Risk factors include currency volatility, visa regulations and client concentration.",
    ],
    "Asian Paints": [
        "Decorative paints demand was weak due to subdued urban consumption and heavy monsoon.",
        "Raw material inflation in crude derivatives pressured gross margins during the year.",
    ],
}


@pytest.fixture
def populated(chroma_collection):
    for company, texts in PAGES.items():
        pages = [{"page_index": i, "page_label": str(i + 101), "text": t} for i, t in enumerate(texts)]
        ids, docs, metas = build_records(pages, company, 2025, "FY2025", f"{company} AR 2025", "ar.pdf")
        chroma_collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return chroma_collection


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # overlap carries context forward
    assert not chunks[1].startswith("ord")  # starts on a word boundary


def test_chunking_edge_cases():
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
    assert normalize_whitespace("a  b\n\nc trans-\nformation") == "a b c transformation"


def test_record_ids_and_metadata_are_stable():
    ids, _, metas = build_records([{"page_index": 111, "page_label": "112", "text": "hello world"}],
                                  "Infosys", 2025, "FY2025", "T", "f.pdf")
    assert ids == ["infosys-2025-p111-c0"]
    assert metas[0] | {} == {"company": "Infosys", "report_year": 2025, "financial_year": "FY2025",
                             "report_title": "T", "page_index": 111, "page_label": "112",
                             "source_file": "f.pdf", "chunk_index": 0}


def test_company_filter_is_applied(populated):
    r = search_filings("Infosys", "raw material inflation decorative paints demand")
    assert r["status"] == "success"
    assert r["results"] and all(x["source_id"].startswith("infosys-") for x in r["results"])


def test_known_question_retrieves_expected_page(populated):
    r = search_filings("Infosys", "Why did operating margin improve? cost optimisation", k=4)
    assert r["results"][0]["page_index"] == 1
    assert r["results"][0]["page_label"] == "102"
    r = search_filings("Asian Paints", "raw material inflation crude", k=4)
    assert r["results"][0]["page_index"] == 1


def test_unsupported_company_and_missing_reports(chroma_collection):
    assert search_filings("Tesla", "x")["error_code"] == "unsupported_company"
    assert search_filings("HDFC Bank", "net interest margin")["error_code"] == "no_reports"
    assert search_filings("Infosys", "   ")["error_code"] == "invalid_query"


def test_citation_maps_to_correct_report_page(populated):
    r = search_filings("Infosys", "operating margin cost optimisation")
    reg = EvidenceRegistry()
    reg.register_tool_result("search_filings", r)
    sid = r["results"][0]["source_id"]
    out = validate_and_render(f"Margins improved on cost programmes [[{sid}]].", reg)
    assert out["answer"] == "Margins improved on cost programmes [1]."
    c = out["citations"][0]
    assert c["source_id"] == sid and c["page_label"] == "102"
    assert c["label"] == "Infosys AR 2025, p. 102"


def test_unknown_source_ids_cannot_render_as_citations(populated):
    reg = EvidenceRegistry()
    reg.register_tool_result("search_filings", search_filings("Infosys", "risk factors"))
    out = validate_and_render("Revenue tripled [[infosys-2025-p999-c0]].", reg)
    assert out["citations"] == []
    assert "p999" not in out["answer"]
    assert out["invalid_source_ids"] == ["infosys-2025-p999-c0"]
    assert any("no verified source" in w for w in out["warnings"])


def test_ingest_pdf_end_to_end(tmp_path, chroma_collection):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "infy.pdf"
    doc = fitz.open()
    for text in ["Page one about attrition and hiring.", "Page two about large deal wins and growth."]:
        doc.new_page().insert_text((72, 72), text)
    doc.save(pdf)
    summary = ingest_pdf(pdf, "Infosys", 2025, "FY2025", collection=chroma_collection)
    assert summary["pages"] == 2 and summary["chunks"] == 2
    # re-ingest replaces, not duplicates
    ingest_pdf(pdf, "Infosys", 2025, "FY2025", collection=chroma_collection)
    assert len(chroma_collection.get(where={"company": "Infosys"})["ids"]) == 2
    r = search_filings("Infosys", "large deal wins")
    assert r["results"][0]["page_index"] == 1


def test_ingest_rejects_unknown_company(tmp_path):
    with pytest.raises(ValueError):
        ingest_pdf(Path("x.pdf"), "Tesla", 2025, "FY2025")
