"""sdl - a domain-agnostic Self-Driving Lab framework.

The core never knows what a "perovskite" (or any field) is. It speaks only in
ParameterSpace, Objective and ExperimentResult. A field is plugged in by
implementing the `Domain` interface (see `sdl.core.domain`).
"""

__version__ = "0.1.0"

from sdl.core.space import (
    ParameterSpace,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
    SimplexConstraint,
)
from sdl.core.objective import Objective, ObjectiveSet, Direction
from sdl.core.experiment import Experiment, ExperimentResult, Campaign
from sdl.core.domain import Domain
from sdl.core.optimizer import Optimizer
from sdl.core.executor import Executor

__all__ = [
    "ParameterSpace",
    "ContinuousParameter",
    "IntegerParameter",
    "CategoricalParameter",
    "SimplexConstraint",
    "Objective",
    "ObjectiveSet",
    "Direction",
    "Experiment",
    "ExperimentResult",
    "Campaign",
    "Domain",
    "Optimizer",
    "Executor",
]
