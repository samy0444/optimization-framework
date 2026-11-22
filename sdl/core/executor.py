"""Executor interface - turns a proposed Experiment into an observed result.

This is the boundary between the algorithm and "reality". Implementations:
  * SimulatedExecutor - calls domain.simulate() (+ optional noise),
  * (future) LabExecutor - drives real hardware / a lab API.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Sequence

from sdl.core.experiment import Experiment, ExperimentResult


class Executor(ABC):
    name: str = "executor"

    @abstractmethod
    def run_one(self, experiment: Experiment) -> ExperimentResult: ...

    def run_batch(self, experiments: Sequence[Experiment]) -> List[ExperimentResult]:
        return [self.run_one(e) for e in experiments]

    @staticmethod
    def _timed(fn, experiment: Experiment) -> ExperimentResult:
        start = time.perf_counter()
        result = fn(experiment)
        result.duration_s = time.perf_counter() - start
        return result
