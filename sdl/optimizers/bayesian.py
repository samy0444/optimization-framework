"""Bayesian optimization over the scalarized objective.

CPU-only via scikit-learn's GaussianProcessRegressor (no GPU/botorch needed).
Models the scalarized objective on the unit hypercube, then picks the batch by
maximizing an acquisition function over a large random candidate pool
(constraint-respecting, gradient-free, robust). Falls back to random search if
scikit-learn is unavailable.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from sdl.core.experiment import Experiment, ExperimentResult
from sdl.core.objective import ObjectiveSet
from sdl.core.optimizer import Optimizer
from sdl.core.registry import register_optimizer
from sdl.core.space import ParameterSpace
from sdl.optimizers._search import (
    build_grid,
    candidate_pool,
    greedy_diverse_batch,
    seen_keys,
    vector_key,
)

logger = logging.getLogger("sdl.optimizers.bayesian")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


@register_optimizer("bayesian")
class BayesianOptimizer(Optimizer):
    def __init__(
        self,
        space: ParameterSpace,
        objectives: ObjectiveSet,
        seed: Optional[int] = None,
        acquisition: str = "ei",      # "ei" | "ucb"
        kappa: float = 2.0,           # UCB exploration weight
        xi: float = 0.01,             # EI exploration margin
        pool_size: int = 4096,        # candidate pool per proposal (random spaces)
        min_train: int = 4,           # fall back to random below this many points
        batch_diversity: float = 0.2,  # spread batch picks (unit-cube scale; 0=off)
    ):
        super().__init__(space, objectives, seed)
        self.acquisition = acquisition
        self.kappa = kappa
        self.xi = xi
        self.pool_size = pool_size
        self.min_train = min_train
        self.batch_diversity = batch_diversity
        self._gp = None
        self._y_best = None
        self._grid = None
        self._grid_built = False
        self._proposed = set()

    def _fit(self, history: Sequence[ExperimentResult]) -> bool:
        if not _HAVE_SKLEARN:
            return False
        pts = [(self.space.to_vector(r.params), self.objectives.scalarize(r.metrics))
               for r in history if r.success]
        if len(pts) < self.min_train:
            return False
        X = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        # standardize targets for numerical stability
        self._y_mean, self._y_std = float(y.mean()), float(y.std() or 1.0)
        y_n = (y - self._y_mean) / self._y_std
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(1e-3, (1e-6, 1e1))
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False, n_restarts_optimizer=2,
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        )
        self._gp.fit(X, y_n)
        self._y_best = float(y_n.max())
        return True

    def _acq(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        sigma = np.maximum(sigma, 1e-9)
        if self.acquisition == "ucb":
            return mu + self.kappa * sigma
        if self.acquisition == "thompson":
            # one posterior draw per candidate; argmax selection = Thompson sampling
            return self.rng.normal(mu, sigma)
        # Expected improvement (maximization)
        from scipy.stats import norm
        imp = mu - self._y_best - self.xi
        z = imp / sigma
        return imp * norm.cdf(z) + sigma * norm.pdf(z)

    def _ensure_grid(self) -> None:
        if not self._grid_built:
            self._grid = build_grid(self.space)
            self._grid_built = True

    def _remember(self, params) -> None:
        self._proposed.add(vector_key(self.space.to_vector(params)))

    def propose(self, batch_size: int, history: Sequence[ExperimentResult]) -> List[Experiment]:
        self._ensure_grid()
        if not self._fit(history):
            out = []
            for _ in range(batch_size):
                p = self.space.sample(self.rng)
                self._remember(p)
                out.append(Experiment(params=p, proposed_by=self.name + ":random"))
            return out

        # full real option set (finite space) or deduped random pool, excluding
        # everything already measured or already proposed.
        exclude = self._proposed | seen_keys(self.space, history)
        cand_params = candidate_pool(self.space, self.pool_size, self.rng, exclude, self._grid)
        X = np.array([self.space.to_vector(p) for p in cand_params], dtype=float)
        mu, sigma = self._gp.predict(X, return_std=True)
        scores = self._acq(mu, sigma)

        chosen = []
        for idx in greedy_diverse_batch(X, scores, batch_size, self.batch_diversity):
            p = cand_params[idx]
            self._remember(p)
            exp = Experiment(params=p, proposed_by=self.name)
            exp.predicted = {"_scalar": float(mu[idx] * self._y_std + self._y_mean)}
            exp.predicted_std = {"_scalar": float(sigma[idx] * self._y_std)}
            chosen.append(exp)
        while len(chosen) < batch_size:
            p = self.space.sample(self.rng)
            self._remember(p)
            chosen.append(Experiment(params=p, proposed_by=self.name + ":fill"))
        return chosen
