"""Branin-Hoo: a classic optimization test function as a trivial SDL domain.

It shows that the framework is domain-agnostic: a
real field (perovskite) and a 6-line math function plug into the *same*
orchestrator, optimizers and executor with zero special-casing. The known
global minimum (~0.397887) gives benchmarks a ground-truth target.
"""

from __future__ import annotations

import math
from typing import Dict

from sdl.core.domain import Domain
from sdl.core.objective import Direction, Objective, ObjectiveSet
from sdl.core.registry import register_domain
from sdl.core.space import ContinuousParameter, ParameterSpace

GLOBAL_MIN = 0.397887        # global minimum of the cheap prior, simulate()
# ground_truth() deepens the well at (pi, 2.275) by 6.0, so reality's true
# minimum sits below the prior's. Benchmarks/tests compare against this floor.
GROUND_TRUTH_MIN = GLOBAL_MIN - 6.0


@register_domain("branin")
class BraninDomain(Domain):
    name = "branin"

    def parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters=[
                ContinuousParameter("x1", -5.0, 10.0),
                ContinuousParameter("x2", 0.0, 15.0),
            ]
        )

    def objectives(self) -> ObjectiveSet:
        return ObjectiveSet(
            [Objective("branin", direction=Direction.MINIMIZE, weight=1.0, target=0.5)]
        )

    def simulate(self, params: Dict[str, float]) -> Dict[str, float]:
        x1, x2 = float(params["x1"]), float(params["x2"])
        a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
        value = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
        return {"branin": value}

    def ground_truth(self, params: Dict[str, float]) -> Dict[str, float]:
        """Reality = the textbook prior minus a smooth systematic bias.

        The standard Branin has three tied global minima; reality deepens the
        well near (pi, 2.275) so its true optimum is unique and *misranked* by
        the prior. The delta-GP exists to learn exactly this gap.
        """
        x1, x2 = float(params["x1"]), float(params["x2"])
        base = self.simulate(params)["branin"]
        bias = 6.0 * math.exp(-((x1 - math.pi) ** 2 + (x2 - 2.275) ** 2) / 4.0)
        return {"branin": base - bias}
