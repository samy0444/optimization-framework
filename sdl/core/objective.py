"""Generic objectives and multi-objective utilities.

Replaces hardcoded perovskite metrics (PCE/Voc/Jsc/FF/T80) with a declarative
set of named objectives. Any field declares what it cares about and which way
is "better".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

Metrics = Dict[str, float]


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Objective:
    """One scalar quantity to optimize.

    weight: relative importance in scalarization.
    target: optional 'good enough' value; used for convergence and for
            saturating reward (no credit beyond target).
    bound:  optional hard feasibility bound; candidates violating it are penalized.
    """

    name: str
    direction: Direction = Direction.MAXIMIZE
    weight: float = 1.0
    target: Optional[float] = None
    bound: Optional[float] = None

    def signed(self, value: float) -> float:
        """Return value oriented so that larger is always better."""
        return value if self.direction == Direction.MAXIMIZE else -value

    def satisfies_target(self, value: float) -> bool:
        if self.target is None:
            return False
        return self.signed(value) >= self.signed(self.target)

    def satisfies_bound(self, value: float) -> bool:
        if self.bound is None:
            return True
        return self.signed(value) >= self.signed(self.bound)


@dataclass
class ObjectiveSet:
    objectives: List[Objective]
    # per-objective normalization scale, learned from observed data
    _scale: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        names = [o.name for o in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate objective names")
        self._by_name = {o.name: o for o in self.objectives}

    @property
    def names(self) -> List[str]:
        return [o.name for o in self.objectives]

    def __getitem__(self, name: str) -> Objective:
        return self._by_name[name]

    def update_scales(self, history: Sequence[Metrics]) -> None:
        """Estimate a robust per-objective scale from observed metrics."""
        for o in self.objectives:
            vals = np.array(
                [m[o.name] for m in history if o.name in m and np.isfinite(m[o.name])],
                dtype=float,
            )
            if vals.size >= 2:
                spread = float(np.std(vals))
                self._scale[o.name] = spread if spread > 1e-9 else 1.0
            elif vals.size == 1:
                self._scale[o.name] = max(abs(float(vals[0])), 1.0)

    def scalarize(self, metrics: Metrics) -> float:
        """Weighted scalarization where larger is better.

        Each objective is oriented (max), optionally saturated at its target,
        scaled, then weighted. Bound violations apply a steep penalty so the
        optimizer learns to avoid infeasible regions without hard rejection.
        """
        total = 0.0
        penalty = 0.0
        for o in self.objectives:
            if o.name not in metrics or not np.isfinite(metrics[o.name]):
                continue
            v = float(metrics[o.name])
            s = o.signed(v)
            if o.target is not None:
                s = min(s, o.signed(o.target))  # no credit past the target
            scale = self._scale.get(o.name, max(abs(s), 1.0))
            total += o.weight * (s / scale)
            if not o.satisfies_bound(v):
                deficit = o.signed(o.bound) - o.signed(v)
                penalty += o.weight * abs(deficit) / scale
        return total - 10.0 * penalty

    def all_targets_met(self, metrics: Metrics) -> bool:
        relevant = [o for o in self.objectives if o.target is not None]
        if not relevant:
            return False
        return all(o.satisfies_target(metrics.get(o.name, float("-inf"))) for o in relevant)


def dominates(a: Metrics, b: Metrics, objectives: Sequence[Objective]) -> bool:
    """True if a Pareto-dominates b (no worse on all, strictly better on one)."""
    not_worse = True
    strictly_better = False
    for o in objectives:
        av, bv = o.signed(a.get(o.name, float("-inf"))), o.signed(b.get(o.name, float("-inf")))
        if av < bv:
            not_worse = False
            break
        if av > bv:
            strictly_better = True
    return not_worse and strictly_better


def pareto_front(points: Sequence[Metrics], objectives: Sequence[Objective]) -> List[int]:
    """Return indices of the non-dominated points."""
    front: List[int] = []
    for i, pi in enumerate(points):
        if not any(dominates(pj, pi, objectives) for j, pj in enumerate(points) if j != i):
            front.append(i)
    return front
