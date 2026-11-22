"""Parameter space definition and sampling.

A `ParameterSpace` is a list of named parameters plus optional constraints.
Optimizers consume the space through a uniform numeric-vector view
(`to_vector` / `from_vector`) so they never need domain knowledge.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

Params = Dict[str, Any]


class Parameter(ABC):
    """A single named tunable dimension."""

    name: str

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> Any: ...

    @abstractmethod
    def to_unit(self, value: Any) -> float:
        """Map a value to the unit interval [0, 1] for optimizer-facing vectors."""

    @abstractmethod
    def from_unit(self, u: float) -> Any:
        """Inverse of `to_unit`."""

    @abstractmethod
    def clamp(self, value: Any) -> Any: ...


@dataclass
class ContinuousParameter(Parameter):
    name: str
    low: float
    high: float
    log: bool = False  # sample on a log scale (for rates, concentrations, etc.)

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high must exceed low")
        if self.log and self.low <= 0:
            raise ValueError(f"{self.name}: log scale requires low > 0")

    def sample(self, rng: np.random.Generator) -> float:
        return self.from_unit(float(rng.random()))

    def to_unit(self, value: Any) -> float:
        v = self.clamp(float(value))
        if self.log:
            return (math.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (v - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        u = min(1.0, max(0.0, float(u)))
        if self.log:
            return float(math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low))))
        return float(self.low + u * (self.high - self.low))

    def clamp(self, value: Any) -> float:
        return float(min(self.high, max(self.low, float(value))))


@dataclass
class IntegerParameter(Parameter):
    name: str
    low: int
    high: int

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(f"{self.name}: high must be >= low")

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high + 1))

    def to_unit(self, value: Any) -> float:
        v = self.clamp(value)
        if self.high == self.low:
            return 0.0
        return (v - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> int:
        u = min(1.0, max(0.0, float(u)))
        return int(round(self.low + u * (self.high - self.low)))

    def clamp(self, value: Any) -> int:
        return int(min(self.high, max(self.low, int(round(float(value))))))


@dataclass
class CategoricalParameter(Parameter):
    name: str
    choices: Sequence[Any]

    def __post_init__(self):
        self.choices = list(self.choices)
        if len(self.choices) == 0:
            raise ValueError(f"{self.name}: needs at least one choice")

    def sample(self, rng: np.random.Generator) -> Any:
        return self.choices[int(rng.integers(0, len(self.choices)))]

    def to_unit(self, value: Any) -> float:
        idx = self.choices.index(value)
        if len(self.choices) == 1:
            return 0.0
        return idx / (len(self.choices) - 1)

    def from_unit(self, u: float) -> Any:
        u = min(1.0, max(0.0, float(u)))
        idx = int(round(u * (len(self.choices) - 1)))
        return self.choices[idx]

    def clamp(self, value: Any) -> Any:
        return value if value in self.choices else self.choices[0]


class Constraint(ABC):
    """A feasibility rule over a full parameter dict, with optional repair."""

    @abstractmethod
    def is_feasible(self, params: Params) -> bool: ...

    def repair(self, params: Params, rng: np.random.Generator) -> Params:
        """Project an infeasible point to a feasible one. Default: identity."""
        return params


@dataclass
class SimplexConstraint(Constraint):
    """A group of parameters that must sum to a constant (e.g. mole fractions).

    Generalizes the perovskite 'A-site cations sum to 1' rule to any field.
    """

    members: Sequence[str]
    total: float = 1.0
    tol: float = 1e-6

    def is_feasible(self, params: Params) -> bool:
        s = sum(float(params[m]) for m in self.members)
        return abs(s - self.total) <= self.tol

    def repair(self, params: Params, rng: np.random.Generator) -> Params:
        out = dict(params)
        vals = np.array([max(0.0, float(out[m])) for m in self.members], dtype=float)
        s = vals.sum()
        if s <= 0:
            vals = np.full(len(self.members), self.total / len(self.members))
        else:
            vals = vals / s * self.total
        for m, v in zip(self.members, vals):
            out[m] = float(v)
        return out


@dataclass
class ParameterSpace:
    parameters: List[Parameter]
    constraints: List[Constraint] = field(default_factory=list)

    def __post_init__(self):
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate parameter names in space")
        self._by_name = {p.name: p for p in self.parameters}

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def __getitem__(self, name: str) -> Parameter:
        return self._by_name[name]

    def is_feasible(self, params: Params) -> bool:
        return all(c.is_feasible(params) for c in self.constraints)

    def repair(self, params: Params, rng: Optional[np.random.Generator] = None) -> Params:
        rng = rng or np.random.default_rng()
        out = {p.name: p.clamp(params[p.name]) for p in self.parameters}
        for c in self.constraints:
            out = c.repair(out, rng)
        return out

    def sample(self, rng: Optional[np.random.Generator] = None) -> Params:
        rng = rng or np.random.default_rng()
        params = {p.name: p.sample(rng) for p in self.parameters}
        if self.constraints:
            params = self.repair(params, rng)
        return params

    def sample_batch(self, n: int, rng: Optional[np.random.Generator] = None) -> List[Params]:
        rng = rng or np.random.default_rng()
        return [self.sample(rng) for _ in range(n)]

    def enumerate_grid(self, max_points: int = 200_000) -> Optional[List[Params]]:
        """Every combination of a *finite* space (all categorical/integer), or
        None if the space is continuous or larger than ``max_points``.

        Lets optimizers score the full real option set instead of a random
        subset, so they never miss the best combination on a finite design.
        Constraint-infeasible combinations are dropped.
        """
        from itertools import product

        axes: List[List[Any]] = []
        for p in self.parameters:
            if isinstance(p, CategoricalParameter):
                axes.append(list(p.choices))
            elif isinstance(p, IntegerParameter):
                axes.append(list(range(p.low, p.high + 1)))
            else:
                return None  # a continuous dimension -> not enumerable
        count = 1
        for a in axes:
            count *= max(1, len(a))
            if count > max_points:
                return None
        combos: List[Params] = []
        for values in product(*axes):
            params = {p.name: v for p, v in zip(self.parameters, values)}
            if self.constraints and not self.is_feasible(params):
                continue
            combos.append(params)
        return combos

    # --- optimizer-facing uniform numeric view (unit hypercube) ---
    def to_vector(self, params: Params) -> np.ndarray:
        return np.array([p.to_unit(params[p.name]) for p in self.parameters], dtype=float)

    def from_vector(self, vec: Sequence[float], rng: Optional[np.random.Generator] = None) -> Params:
        params = {p.name: p.from_unit(float(v)) for p, v in zip(self.parameters, vec)}
        if self.constraints:
            params = self.repair(params, rng)
        return params
