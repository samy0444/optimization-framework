"""FastAPI service for running the SDL framework over HTTP.

Run:
    uvicorn sdl.api.app:app --reload --port 8000
Then open http://localhost:8000/docs for interactive API docs.

Endpoints
    GET  /health
    GET  /registry                      -> available domains/optimizers/executors
    GET  /domains/{name}                -> parameter space + objectives schema
    POST /domains/{name}/simulate       -> run the physics prior on one point
    POST /campaigns                     -> launch an optimization campaign (async)
    GET  /campaigns/{id}                -> status + summary
    GET  /campaigns                     -> list campaigns

Campaigns run in a background thread with an in-memory registry. For real
production you would back this with a task queue (Celery/RQ) and a database;
the orchestrator/store interfaces are already structured to allow that swap.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise SystemExit("FastAPI/pydantic not installed. `pip install fastapi uvicorn pydantic`.") from e

from sdl.core.objective import Direction
from sdl.core.orchestrator import CampaignConfig, Orchestrator
from sdl.core.registry import available, get_domain, get_executor, get_optimizer
from sdl.core.space import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
)

app = FastAPI(title="Self-Driving Lab", version="0.1.0",
              description="Domain-agnostic autonomous experimentation platform.")

_CAMPAIGNS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


class SimulateRequest(BaseModel):
    params: Dict[str, Any]


class CampaignRequest(BaseModel):
    domain: str
    optimizer: str = "hybrid"
    iterations: int = Field(15, ge=1, le=500)
    batch: int = Field(4, ge=1, le=64)
    initial: int = Field(8, ge=0, le=256)
    noise: float = Field(0.0, ge=0.0, le=1.0)
    seed: Optional[int] = None


def _param_schema(p) -> Dict[str, Any]:
    if isinstance(p, ContinuousParameter):
        return {"name": p.name, "type": "continuous", "low": p.low, "high": p.high, "log": p.log}
    if isinstance(p, IntegerParameter):
        return {"name": p.name, "type": "integer", "low": p.low, "high": p.high}
    if isinstance(p, CategoricalParameter):
        return {"name": p.name, "type": "categorical", "choices": list(p.choices)}
    return {"name": p.name, "type": "unknown"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/registry")
def registry() -> Dict[str, List[str]]:
    return available()


@app.get("/domains/{name}")
def describe_domain(name: str) -> Dict[str, Any]:
    try:
        domain = get_domain(name)()
    except KeyError as e:
        raise HTTPException(404, str(e))
    space = domain.parameter_space()
    objs = domain.objectives()
    return {
        "name": domain.name,
        "parameters": [_param_schema(p) for p in space.parameters],
        "constraints": [c.__class__.__name__ for c in space.constraints],
        "objectives": [
            {"name": o.name, "direction": o.direction.value, "weight": o.weight,
             "target": o.target, "bound": o.bound}
            for o in objs.objectives
        ],
    }


@app.post("/domains/{name}/simulate")
def simulate(name: str, req: SimulateRequest) -> Dict[str, Any]:
    try:
        domain = get_domain(name)()
    except KeyError as e:
        raise HTTPException(404, str(e))
    params = domain.parameter_space().repair(req.params)
    metrics = domain.simulate(params)
    if metrics is None:
        raise HTTPException(400, f"Domain '{name}' has no simulate(); needs a lab executor.")
    return {"params": params, "metrics": metrics}


def _run_campaign(cid: str, req: CampaignRequest) -> None:
    try:
        domain = get_domain(req.domain)()
        optimizer = get_optimizer(req.optimizer)(domain.parameter_space(), domain.objectives(), seed=req.seed)
        executor = get_executor("simulation")(domain, noise=req.noise, seed=req.seed)
        cfg = CampaignConfig(n_iterations=req.iterations, batch_size=req.batch,
                             n_initial=req.initial, seed=req.seed)
        campaign = Orchestrator(domain, optimizer, executor, cfg).run()
        with _LOCK:
            _CAMPAIGNS[cid].update(
                status="completed",
                summary=campaign.summary(),
                convergence=campaign.convergence_history,
            )
    except Exception as e:  # surface failures to the client
        with _LOCK:
            _CAMPAIGNS[cid].update(status="failed", error=str(e))


@app.post("/campaigns")
def create_campaign(req: CampaignRequest) -> Dict[str, str]:
    if req.domain not in available()["domains"]:
        raise HTTPException(404, f"Unknown domain '{req.domain}'")
    cid = uuid.uuid4().hex[:12]
    with _LOCK:
        _CAMPAIGNS[cid] = {"id": cid, "status": "running", "request": req.model_dump()}
    threading.Thread(target=_run_campaign, args=(cid, req), daemon=True).start()
    return {"campaign_id": cid, "status": "running"}


@app.get("/campaigns/{cid}")
def get_campaign(cid: str) -> Dict[str, Any]:
    with _LOCK:
        if cid not in _CAMPAIGNS:
            raise HTTPException(404, "Unknown campaign")
        return _CAMPAIGNS[cid]


@app.get("/campaigns")
def list_campaigns() -> List[Dict[str, Any]]:
    with _LOCK:
        return [{"id": c["id"], "status": c["status"]} for c in _CAMPAIGNS.values()]
