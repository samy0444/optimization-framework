"""The Domain interface - the single seam every field plugs into.

To support a new field you implement one `Domain` subclass. The orchestrator,
optimizers and executors require nothing else. A domain declares:
  * a ParameterSpace (what can be varied),
  * an ObjectiveSet (what "good" means),
  * optionally simulate() - a cheap physics/analytic prior, and
  * optionally featurize() - domain features for the ML surrogate.

The cheap `simulate()` is the seam for the hybrid physics+ML ("delta") strategy:
the framework can learn the residual between simulate() and observed reality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from sdl.core.objective import ObjectiveSet
from sdl.core.space import ParameterSpace

Params = Dict[str, Any]
Metrics = Dict[str, float]


class Domain(ABC):
    """Base class for a pluggable scientific domain."""

    #: short unique name used by the plugin registry
    name: str = "unnamed"

    @abstractmethod
    def parameter_space(self) -> ParameterSpace: ...

    @abstractmethod
    def objectives(self) -> ObjectiveSet: ...

    def simulate(self, params: Params) -> Optional[Metrics]:
        """Cheap forward model (physics/analytic). Return None if unavailable.

        This is a *prior*, not ground truth. The executor produces ground truth;
        the surrogate may learn corrections on top of this.
        """
        return None

    def ground_truth(self, params: Params) -> Optional[Metrics]:
        """What the executor actually measures. Return None to fall back to simulate().

        In a real campaign this is the lab / a high-fidelity reference. In a
        simulated domain it is a more expensive forward model that DIFFERS from the
        cheap simulate() prior, so the delta-GP has a genuine reality gap to
        learn. Keeping it separate from simulate() keeps the physics-prior
        evaluation honest rather than circular.
        """
        return None

    def featurize(self, params: Params) -> np.ndarray:
        """Default featurization: the optimizer-facing unit vector.

        Override to inject domain descriptors (e.g. tolerance factor, bandgap).
        """
        return self.parameter_space().to_vector(self.parameter_space().repair(params))

    def feature_names(self) -> List[str]:
        return list(self.parameter_space().names)

    def validate(self, params: Params) -> bool:
        return self.parameter_space().is_feasible(params)
