"""Generic experiment, result and campaign containers.

`ExperimentResult.metrics` is an arbitrary {name: value} dict, so the same
container works for solar cells, batteries, catalysts or a toy test function.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

Params = Dict[str, Any]
Metrics = Dict[str, float]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "exp") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Experiment:
    """A proposed point in the parameter space, awaiting execution."""

    params: Params
    experiment_id: str = field(default_factory=lambda: _new_id("exp"))
    proposed_by: str = "unknown"          # optimizer name
    iteration: int = -1
    predicted: Metrics = field(default_factory=dict)   # surrogate prediction
    predicted_std: Metrics = field(default_factory=dict)  # surrogate uncertainty
    created_at: str = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """The observed outcome of executing an experiment."""

    experiment_id: str
    params: Params
    metrics: Metrics
    success: bool = True
    source: str = "simulation"            # simulation | hybrid | lab | ...
    cost: float = 0.0                     # arbitrary cost units (time, money, material)
    duration_s: float = 0.0
    error: Optional[str] = None
    iteration: int = -1
    proposed_by: str = "unknown"
    observed_at: str = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Campaign:
    """Mutable state of an optimization campaign."""

    domain_name: str
    campaign_id: str = field(default_factory=lambda: _new_id("camp"))
    start_time: str = field(default_factory=_now)
    config: Dict[str, Any] = field(default_factory=dict)

    results: List[ExperimentResult] = field(default_factory=list)
    iterations_completed: int = 0
    total_cost: float = 0.0

    best_scalar: float = float("-inf")
    best_result: Optional[ExperimentResult] = None
    convergence_history: List[float] = field(default_factory=list)

    def record(self, result: ExperimentResult, scalar: float) -> None:
        self.results.append(result)
        self.total_cost += result.cost
        if result.success and scalar > self.best_scalar:
            self.best_scalar = scalar
            self.best_result = result
        self.convergence_history.append(self.best_scalar)

    @property
    def n_experiments(self) -> int:
        return len(self.results)

    def successful_results(self) -> List[ExperimentResult]:
        return [r for r in self.results if r.success]

    def summary(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "domain": self.domain_name,
            "iterations": self.iterations_completed,
            "experiments": self.n_experiments,
            "successful": len(self.successful_results()),
            "total_cost": self.total_cost,
            "best_scalar": self.best_scalar,
            "best_params": self.best_result.params if self.best_result else None,
            "best_metrics": self.best_result.metrics if self.best_result else None,
        }
