"""Tabular (dataset-backed) domains - the bridge to REAL laboratory data.

A `TabularDomain` turns a table of *already-measured* experiments into an SDL
domain. Its `ground_truth()` is not a simulation - it is a lookup of the real
number a lab recorded. This enables retrospective validation (see
docs/BENCHMARKS.md): replay the optimizer over real published high-throughput data, let it pick which
experiments to "run", and reveal only the real measured outcome of its picks.
If it reaches the best real reaction in fewer picks than standard methods, that
is a real-data result - no lab required.

`BuchwaldHartwigDomain` wraps the Science 2018 (Ahneman/Dreher/Doyle) Buchwald-
Hartwig amination HTE dataset: 3,955 real reactions, 4 categorical factors
(ligand, additive, base, aryl halide), measured yield 0-100%.

Optional historical prior: `enable_prior(fraction, seed)` fits a quick model on a
random historical slice of the table and exposes it via `simulate()`. The
physics/physics_bo optimizer then warm-starts from that prior, showing on real
data how past reactions can speed up a new campaign. See docs/BENCHMARKS.md for
the caveats on this split.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from sdl.core.domain import Domain
from sdl.core.objective import Direction, Objective, ObjectiveSet
from sdl.core.registry import register_domain
from sdl.core.space import CategoricalParameter, ParameterSpace

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class TabularDomain(Domain):
    """Dataset-backed domain. `ground_truth` is a real-measurement lookup."""

    #: column names of the categorical factors (the parameter space)
    factors: List[str] = []
    #: column name of the measured objective
    target: str = "yield"
    #: target value that counts as "success" (e.g. reach >=95% yield)
    target_value: float = 95.0
    #: csv file name inside sdl/data/
    csv_name: str = ""

    def __init__(self) -> None:
        self._rows: List[Dict[str, str]] = []
        self._table: Dict[Tuple[str, ...], float] = {}
        self._levels: Dict[str, List[str]] = {}
        self._prior_model = None
        self._load()

    # --- data loading ---
    def _load(self) -> None:
        path = os.path.normpath(os.path.join(_DATA_DIR, self.csv_name))
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                self._rows.append(row)
                key = tuple(str(row[f]) for f in self.factors)
                self._table[key] = float(row[self.target])
        for f in self.factors:
            self._levels[f] = sorted({str(r[f]) for r in self._rows})

    # --- SDL interface ---
    def parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters=[CategoricalParameter(f, self._levels[f]) for f in self.factors]
        )

    def objectives(self) -> ObjectiveSet:
        return ObjectiveSet(
            [Objective(self.target, Direction.MAXIMIZE, weight=1.0,
                       target=self.target_value, bound=0.0)]
        )

    def lookup(self, params: Dict[str, str]) -> Optional[float]:
        """Return the real measured value, or None if that combination was not
        measured (the executor treats None as a failed/unavailable experiment)."""
        key = tuple(str(params[f]) for f in self.factors)
        return self._table.get(key)

    def ground_truth(self, params: Dict[str, str]) -> Optional[Dict[str, float]]:
        v = self.lookup(params)
        return None if v is None else {self.target: v}

    # --- optional historical prior (warm-start demo) ---
    def enable_prior(self, fraction: float = 0.25, seed: int = 0) -> int:
        """Fit a prior model on a random historical slice of the table.

        Returns the number of historical rows used. After this, `simulate()`
        returns the prior's prediction, so physics_bo warm-starts from it.
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception:
            return 0
        space = self.parameter_space()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self._rows))[: max(1, int(fraction * len(self._rows)))]
        X, y = [], []
        for i in idx:
            params = {f: self._rows[i][f] for f in self.factors}
            X.append(space.to_vector(params))
            y.append(float(self._rows[i][self.target]))
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
        model.fit(np.array(X), np.array(y))
        self._prior_model = model
        self._prior_space = space
        return len(idx)

    def simulate(self, params: Dict[str, str]) -> Optional[Dict[str, float]]:
        if self._prior_model is None:
            return None
        vec = self._prior_space.to_vector({f: str(params[f]) for f in self.factors})
        pred = float(self._prior_model.predict(vec.reshape(1, -1))[0])
        return {self.target: pred}


@register_domain("buchwald")
class BuchwaldHartwigDomain(TabularDomain):
    """Real Science 2018 Buchwald-Hartwig amination HTE dataset (3,955 reactions)."""

    name = "buchwald"
    factors = ["ligand", "additive", "base", "aryl_halide"]
    target = "yield"
    target_value = 95.0
    csv_name = "buchwald_hartwig.csv"
