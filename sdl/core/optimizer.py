"""Optimizer interface.

An optimizer proposes new experiments given the space, objectives and history.
It is domain-agnostic: it only sees parameter vectors and scalar/metric history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

from sdl.core.experiment import Experiment, ExperimentResult
from sdl.core.objective import ObjectiveSet
from sdl.core.space import ParameterSpace


class Optimizer(ABC):
    name: str = "optimizer"

    def __init__(self, space: ParameterSpace, objectives: ObjectiveSet, seed: Optional[int] = None):
        self.space = space
        self.objectives = objectives
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def propose(self, batch_size: int, history: Sequence[ExperimentResult]) -> List[Experiment]:
        """Return `batch_size` experiments to run next."""

    def observe(self, results: Sequence[ExperimentResult]) -> None:
        """Optional hook to update internal state after results arrive."""
        return None

    def bind_domain(self, domain) -> None:
        """Optional hook: physics-aware optimizers capture the domain's cheap
        prior here. The orchestrator calls this once before the campaign. Most
        optimizers ignore it.
        """
        return None
