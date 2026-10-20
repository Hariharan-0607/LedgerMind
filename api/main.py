"""FastAPI backend.

Run: uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import config
from agent.loop import run_agent
from services.compare import SUPPORTED_METRICS, compare_companies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(config.LOG_DIR / "app.log")],
)
log = logging.getLogger("api")

app = FastAPI(title="Financial Research Agent", version="1.0.0",
              description="Educational research assistant. Not investment advice.")

RUN_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_runs: dict[str, dict] = {}


def _validate_companies(value: list[str]) -> list[str]:
    out = []
    for c in value:
        name = config.resolve_company(c)
        if not name:
            raise ValueError(f"Unsupported company '{c}'. Supported: {sorted(config.COMPANIES)}")
        if name not in out:
            out.append(name)
    return out


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=config.MAX_QUESTION_CHARS)
    companies: list[str] = Field(default_factory=list, max_length=3)
    include_news: bool = True

    @field_validator("companies")
    @classmethod
    def check_companies(cls, v):
        return _validate_companies(v)


class CompareRequest(BaseModel):
    companies: list[str] = Field(..., min_length=2, max_length=3)
    metrics: list[str] = Field(default_factory=lambda: ["revenue", "net_income", "net_profit_margin",
                                                         "revenue_growth"], max_length=6)

    @field_validator("companies")
    @classmethod
    def check_companies(cls, v):
        return _validate_companies(v)

    @field_validator("metrics")
    @classmethod
    def check_metrics(cls, v):
        bad = [m for m in v if m not in SUPPORTED_METRICS]
        if bad:
            raise ValueError(f"Unsupported metrics {bad}. Supported: {sorted(SUPPORTED_METRICS)}")
        return v


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    answer: str
    metrics: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    warnings: list[str] = []
    artifacts: dict[str, Any] = {}
    disclaimer: str = ""
    duration_s: float = 0.0


def _save_run(run: dict) -> None:
    """Persist runs for debugging/evaluation. Stores the question and outputs only."""
    _runs[run["run_id"]] = run
    try:
        (config.RUNS_DIR / f"{run['run_id']}.json").write_text(json.dumps(run, default=str, indent=1))
    except OSError:
        log.warning("could not persist run %s", run["run_id"])


def _load_run(run_id: str) -> dict:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id")
    if run_id in _runs:
        return _runs[run_id]
    path = config.RUNS_DIR / f"{run_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    raise HTTPException(status_code=404, detail="Run not found")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    # Controlled error: no stack traces or secrets leak to clients.
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error. See server logs."})


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": bool(config.GROQ_API_KEY and config.MODEL)}


@app.get("/companies")
def companies():
    return [{"name": n, "ticker": i["ticker"], "sector": i["sector"]} for n, i in config.COMPANIES.items()]


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    run_id = uuid.uuid4().hex[:12]
    log.info("run=%s research request companies=%s news=%s", run_id, req.companies, req.include_news)
    result = run_agent(req.question, companies=req.companies or None,
                       include_news=req.include_news, run_id=run_id)
    result["metrics"] = result["artifacts"]["metrics"]
    _save_run(result)
    return result


@app.post("/compare")
def compare(req: CompareRequest):
    return compare_companies(req.companies, req.metrics)


@app.get("/research/{run_id}", response_model=ResearchResponse)
def get_run(run_id: str):
    return _load_run(run_id)


@app.get("/research/{run_id}/trace")
def get_trace(run_id: str):
    run = _load_run(run_id)
    return {"run_id": run_id, "status": run["status"], "trace": run["trace"]}
