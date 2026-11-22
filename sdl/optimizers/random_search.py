"""Random search - the baseline every other optimizer must beat."""

from __future__ import annotations

from typing import List, Sequence

from sdl.core.experiment import Experiment, ExperimentResult
from sdl.core.optimizer import Optimizer
from sdl.core.registry import register_optimizer


@register_optimizer("random")
class RandomSearch(Optimizer):
    def propose(self, batch_size: int, history: Sequence[ExperimentResult]) -> List[Experiment]:
        return [
            Experiment(params=self.space.sample(self.rng), proposed_by=self.name)
            for _ in range(batch_size)
        ]
