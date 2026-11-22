"""Hybrid BO+RL optimizer - splits each batch between BO and an RL policy.

Each proposed batch is split between two complementary proposers:
  * Bayesian optimization - calibrated uncertainty, strong global exploration
    and sample efficiency early in a campaign;
  * RL (SAC) policy        - learns a proposal *policy* over the surrogate
    landscape, strong at exploiting structure once data accumulates.

The split shifts from BO-heavy to RL-heavy as the campaign matures
(`rl_fraction` grows with the amount of history), aiming for the sample
efficiency of BO early and the policy-driven refinement of RL late.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from sdl.core.experiment import Experiment, ExperimentResult
from sdl.core.objective import ObjectiveSet
from sdl.core.optimizer import Optimizer
from sdl.core.registry import register_optimizer
from sdl.core.space import ParameterSpace
from sdl.optimizers.physics_bo import PhysicsPriorBO
from sdl.optimizers.rl import RLOptimizer


@register_optimizer("hybrid")
class HybridBORL(Optimizer):
    def __init__(
        self,
        space: ParameterSpace,
        objectives: ObjectiveSet,
        seed: Optional[int] = None,
        acquisition: str = "ei",
        rl_warmup: int = 16,        # experiments before RL contributes
        max_rl_fraction: float = 0.6,
    ):
        super().__init__(space, objectives, seed)
        # BO leg is physics-prior aware (delta-GP); RL leg exploits structure late.
        self.bo = PhysicsPriorBO(space, objectives, seed=seed, acquisition=acquisition)
        self.rl = RLOptimizer(space, objectives, seed=None if seed is None else seed + 1)
        self.rl_warmup = rl_warmup
        self.max_rl_fraction = max_rl_fraction

    def bind_domain(self, domain) -> None:
        """Forward the prior to both legs (the BO leg actually uses it)."""
        self.bo.bind_domain(domain)
        self.rl.bind_domain(domain)

    def _rl_count(self, batch_size: int, n_history: int) -> int:
        if n_history < self.rl_warmup:
            return 0
        # ramp RL share with experience, capped at max_rl_fraction
        frac = min(self.max_rl_fraction, (n_history - self.rl_warmup) / (4 * self.rl_warmup) * self.max_rl_fraction)
        return min(batch_size, int(round(batch_size * frac)))

    def propose(self, batch_size: int, history: Sequence[ExperimentResult]) -> List[Experiment]:
        n = len([r for r in history if r.success])
        n_rl = self._rl_count(batch_size, n)
        n_bo = batch_size - n_rl

        proposals: List[Experiment] = []
        if n_bo > 0:
            proposals.extend(self.bo.propose(n_bo, history))
        if n_rl > 0:
            proposals.extend(self.rl.propose(n_rl, history))
        for p in proposals:
            p.proposed_by = f"hybrid({p.proposed_by})"
        return proposals[:batch_size]
