"""Executor for dataset-backed (TabularDomain) campaigns.

It returns the REAL measured value for a proposed combination by looking it up
in the dataset. Crucially, if a combination was never measured it reports a
*failed* experiment - it never falls back to a simulated/prior value, so the
"measurements" the optimizer sees are always real lab numbers. (The generic
SimulatedExecutor would fall back to simulate(); for honest retrospective
validation we must not.)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from sdl.core.executor import Executor
from sdl.core.experiment import Experiment, ExperimentResult
from sdl.core.registry import register_executor
from sdl.domains.tabular import TabularDomain


@register_executor("tabular")
class TabularExecutor(Executor):
    def __init__(self, domain: TabularDomain, noise: float = 0.0,
                 seed: Optional[int] = None, cost_per_run: float = 1.0):
        self.domain = domain
        self.noise = noise
        self.cost_per_run = cost_per_run
        self.rng = np.random.default_rng(seed)

    def run_one(self, experiment: Experiment) -> ExperimentResult:
        start = time.perf_counter()
        value = self.domain.lookup(experiment.params)
        if value is None:
            return ExperimentResult(
                experiment_id=experiment.experiment_id, params=experiment.params,
                metrics={}, success=False, source="dataset", cost=self.cost_per_run,
                duration_s=time.perf_counter() - start,
                error="combination not present in dataset",
                proposed_by=experiment.proposed_by,
            )
        if self.noise > 0:
            value = float(value + self.rng.normal(0.0, self.noise))
        return ExperimentResult(
            experiment_id=experiment.experiment_id, params=experiment.params,
            metrics={self.domain.target: float(value)}, success=True, source="dataset",
            cost=self.cost_per_run, duration_s=time.perf_counter() - start,
            proposed_by=experiment.proposed_by, metadata=dict(experiment.metadata),
        )
