import pytest
from fastapi.testclient import TestClient

import api.main as main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.config, "RUNS_DIR", tmp_path)
    main._runs.clear()
    return TestClient(main.app, raise_server_exceptions=False)


def fake_run(question, companies=None, include_news=True, run_id=None, **kw):
    return {"run_id": run_id, "status": "completed", "question": question, "answer": "ok [1]",
            "citations": [{"number": 1, "source_id": "fin:INFY.NS:income_stmt", "label": "x"}],
            "claims": [], "trace": [{"type": "tool_call", "tool": "get_financials", "status": "success"}],
            "warnings": [], "artifacts": {"metrics": [{"metric": "operating_margin", "value": 21.0}]},
            "disclaimer": "d", "duration_s": 0.1, "tool_calls": 1, "tool_errors": 0}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_companies(client):
    tickers = {c["ticker"] for c in client.get("/companies").json()}
    assert tickers == {"INFY.NS", "HDFCBANK.NS", "ASIANPAINT.NS"}


def test_research_roundtrip_and_trace(client, monkeypatch):
    monkeypatch.setattr(main, "run_agent", fake_run)
    r = client.post("/research", json={"question": "How has Infosys revenue changed?",
                                       "companies": ["infy"], "include_news": False})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["metrics"][0]["value"] == 21.0
    run_id = body["run_id"]
    assert client.get(f"/research/{run_id}").json()["answer"] == "ok [1]"
    main._runs.clear()  # forces load from disk
    assert client.get(f"/research/{run_id}/trace").json()["trace"][0]["tool"] == "get_financials"


@pytest.mark.parametrize("payload", [
    {"question": "hi", "companies": []},
    {"question": "x" * 5000},
    {"question": "Tell me about Tesla", "companies": ["Tesla"]},
    {"question": "valid question", "companies": ["Infosys", "HDFC Bank", "Asian Paints", "Infosys"]},
    {"companies": ["Infosys"]},
])
def test_invalid_requests_receive_clear_errors(client, payload):
    r = client.post("/research", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_compare_validation(client):
    assert client.post("/compare", json={"companies": ["Infosys"]}).status_code == 422
    assert client.post("/compare", json={"companies": ["Infosys", "HDFC Bank"],
                                         "metrics": ["ebitda"]}).status_code == 422


def test_compare_ok(client, fake_yf):
    r = client.post("/compare", json={"companies": ["Infosys", "Asian Paints"]})
    assert r.status_code == 200 and r.json()["status"] == "success"


def test_unknown_and_malicious_run_ids(client):
    assert client.get("/research/000000000000").status_code == 404
    assert client.get("/research/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/research/abc").status_code == 400


def test_internal_errors_do_not_leak(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret-key-123 stack detail")

    monkeypatch.setattr(main, "run_agent", boom)
    r = client.post("/research", json={"question": "Infosys revenue?"})
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["detail"].startswith("Internal error")
