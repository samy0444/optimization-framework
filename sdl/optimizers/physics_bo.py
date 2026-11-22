"""Physics-prior Bayesian optimization (delta / residual GP).

This is the framework's main method. Vanilla BO models the objective
from scratch; here the optimizer is *handed the domain's cheap physics prior*
(``Domain.simulate``) and only has to learn the GP on the residual

    delta(x) = observed(x) - scalarize(simulate(x)),

a Kennedy & O'Hagan style correction. The prediction it optimizes is then

    mu(x) = prior(x) + delta_GP(x),

so every proposal starts from physics and spends data only on what physics gets
wrong. When the prior is even loosely informative the residual surface is
flatter and lower-variance than the raw objective, which is exactly why this
reaches a target in fewer experiments than prior-free BO -- the sample-
efficiency claim a practitioner will trust.

Domain-agnostic: it never inspects what the prior *means*, only that it is a
scalar forward model. If a domain supplies no ``simulate`` the prior is treated
as zero and this degrades gracefully to ordinary BO. Falls back to random
search below ``min_train`` points or without scikit-learn.
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

logger = logging.getLogger("sdl.optimizers.physics_bo")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


@register_optimizer("physics_bo")
class PhysicsPriorBO(Optimizer):
    def __init__(
        self,
        space: ParameterSpace,
        objectives: ObjectiveSet,
        seed: Optional[int] = None,
        acquisition: str = "ei",      # "ei" | "ucb" | "thompson"
        kappa: float = 2.0,
        xi: float = 0.01,
        pool_size: int = 4096,
        min_train: int = 4,
        batch_diversity: float = 0.2,  # spread batch picks (unit-cube scale; 0=off)
    ):
        super().__init__(space, objectives, seed)
        self.acquisition = acquisition
        self.kappa = kappa
        self.xi = xi
        self.pool_size = pool_size
        self.min_train = min_train
        self.batch_diversity = batch_diversity
        self._domain = None
        self._gp = None
        self._y_best = None
        self._r_mean = 0.0
        self._r_std = 1.0
        self._grid = None              # cached finite grid [(params, key)] or None
        self._grid_built = False
        self._proposed = set()         # keys of every point ever proposed (no repeats)

    def bind_domain(self, domain) -> None:
        """Capture the cheap prior. Called once by the orchestrator."""
        self._domain = domain

    def _prior_scalar(self, params) -> float:
        """Scalarized objective predicted by the physics prior (0.0 if none)."""
        if self._domain is None:
            return 0.0
        try:
            metrics = self._domain.simulate(params)
        except Exception:
            metrics = None
        if not metrics:
            return 0.0
        return float(self.objectives.scalarize(metrics))

    def _fit(self, history: Sequence[ExperimentResult]) -> bool:
        if not _HAVE_SKLEARN:
            return False
        rows = [r for r in history if r.success]
        if len(rows) < self.min_train:
            return False
        X = np.array([self.space.to_vector(r.params) for r in rows], dtype=float)
        obs = np.array([self.objectives.scalarize(r.metrics) for r in rows], dtype=float)
        prior = np.array([self._prior_scalar(r.params) for r in rows], dtype=float)
        resid = obs - prior

        self._r_mean, self._r_std = float(resid.mean()), float(resid.std() or 1.0)
        r_n = (resid - self._r_mean) / self._r_std
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(1e-3, (1e-6, 1e1))
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False, n_restarts_optimizer=2,
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        )
        self._gp.fit(X, r_n)
        # best composite objective actually observed (== prior+resid at those x)
        self._y_best = float(obs.max())
        return True

    def _acq(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        sigma = np.maximum(sigma, 1e-9)
        if self.acquisition == "ucb":
            return mu + self.kappa * sigma
        if self.acquisition == "thompson":
            return self.rng.normal(mu, sigma)
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
            tag = ":random" if _HAVE_SKLEARN else ":random_nosk"
            out = []
            for _ in range(batch_size):
                p = self.space.sample(self.rng)
                self._remember(p)
                out.append(Experiment(params=p, proposed_by=self.name + tag))
            return out

        # candidates: full real option set (finite space) or deduped random pool,
        # always excluding anything already measured or already proposed.
        exclude = self._proposed | seen_keys(self.space, history)
        cand_params = candidate_pool(self.space, self.pool_size, self.rng, exclude, self._grid)
        X = np.array([self.space.to_vector(p) for p in cand_params], dtype=float)
        r_mu_n, r_sig_n = self._gp.predict(X, return_std=True)
        r_mu = r_mu_n * self._r_std + self._r_mean
        r_sigma = r_sig_n * self._r_std
        prior = np.array([self._prior_scalar(p) for p in cand_params], dtype=float)
        mu = prior + r_mu                       # physics prior + learned correction
        sigma = np.maximum(r_sigma, 1e-9)       # prior is deterministic; GP carries the variance
        scores = self._acq(mu, sigma)

        chosen = []
        for idx in greedy_diverse_batch(X, scores, batch_size, self.batch_diversity):
            p = cand_params[idx]
            self._remember(p)
            exp = Experiment(params=p, proposed_by=self.name)
            exp.predicted = {"_scalar": float(mu[idx]), "_prior": float(prior[idx])}
            exp.predicted_std = {"_scalar": float(sigma[idx])}
            chosen.append(exp)
        while len(chosen) < batch_size:                 # only if pool < batch
            p = self.space.sample(self.rng)
            self._remember(p)
            chosen.append(Experiment(params=p, proposed_by=self.name + ":fill"))
        return chosen
