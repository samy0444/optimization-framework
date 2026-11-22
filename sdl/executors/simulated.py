"""Simulated executor - ground truth comes from the domain's forward model.

Optional Gaussian measurement noise makes it a realistic stand-in for a noisy
lab, which is what the optimizer and (later) validation phase must cope with.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from sdl.core.domain import Domain
from sdl.core.executor import Executor
from sdl.core.experiment import Experiment, ExperimentResult
from sdl.core.registry import register_executor


@register_executor("simulation")
class SimulatedExecutor(Executor):
    def __init__(self, domain: Domain, noise: float = 0.0, seed: Optional[int] = None, cost_per_run: float = 1.0):
        self.domain = domain
        self.noise = noise
        self.cost_per_run = cost_per_run
        self.rng = np.random.default_rng(seed)

    def run_one(self, experiment: Experiment) -> ExperimentResult:
        start = time.perf_counter()
        try:
            # Ground truth is what a real lab measures. When a domain supplies a
            # separate (more expensive) ground_truth(), the executor reports it
            # so the cheap simulate() prior stays an honest, imperfect prior.
            metrics = self.domain.ground_truth(experiment.params)
            if metrics is None:
                metrics = self.domain.simulate(experiment.params)
            if metrics is None:
                raise RuntimeError(f"Domain '{self.domain.name}' provides no ground_truth()/simulate(); use a lab executor.")
            if self.noise > 0:
                metrics = {
                    k: float(v + self.rng.normal(0.0, self.noise * (abs(v) if v else 1.0)))
                    for k, v in metrics.items()
                }
            return ExperimentResult(
                experiment_id=experiment.experiment_id,
                params=experiment.params,
                metrics={k: float(v) for k, v in metrics.items()},
                success=True,
                source="simulation",
                cost=self.cost_per_run,
                duration_s=time.perf_counter() - start,
                proposed_by=experiment.proposed_by,
                metadata=dict(experiment.metadata),
            )
        except Exception as e:  # a single bad point must not kill the campaign
            return ExperimentResult(
                experiment_id=experiment.experiment_id,
                params=experiment.params,
                metrics={},
                success=False,
                source="simulation",
                cost=self.cost_per_run,
                duration_s=time.perf_counter() - start,
                error=str(e),
                proposed_by=experiment.proposed_by,
            )
