"""Reinforcement-learning optimizer (SAC) for experiment proposal.

Formulation: a contextual bandit / one-step continuous-control problem.
  * action  = a point in the unit hypercube (decoded to real parameters),
  * reward  = scalarized objective predicted by a fast surrogate fit on the
              campaign history (so RL training spends *zero* real experiments),
  * episode = a single step.

SAC learns a policy over the surrogate landscape; we then propose the policy's
action plus exploratory perturbations as the next batch. This mirrors the
SAC / stable-baselines3 stack from the original project but is domain-agnostic.

Gracefully degrades to random search if stable-baselines3 / gymnasium are
unavailable, or if there is not yet enough history to fit a surrogate.
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

logger = logging.getLogger("sdl.optimizers.rl")

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC
    _HAVE_RL = True
except Exception:  # pragma: no cover
    _HAVE_RL = False

try:
    from sklearn.ensemble import RandomForestRegressor
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


if _HAVE_RL:

    class _SurrogateEnv(gym.Env):
        """One-step env: pick a point, get the surrogate's predicted scalar."""

        metadata = {"render_modes": []}

        def __init__(self, dim: int, surrogate, y_ref: float, y_scale: float):
            super().__init__()
            self.dim = dim
            self.surrogate = surrogate
            self.y_ref = y_ref
            self.y_scale = y_scale if y_scale > 1e-9 else 1.0
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            x = np.clip(np.asarray(action, dtype=float), 0.0, 1.0).reshape(1, -1)
            pred = float(self.surrogate.predict(x)[0])
            reward = (pred - self.y_ref) / self.y_scale
            return np.zeros(1, dtype=np.float32), float(reward), True, False, {}


@register_optimizer("rl")
class RLOptimizer(Optimizer):
    def __init__(
        self,
        space: ParameterSpace,
        objectives: ObjectiveSet,
        seed: Optional[int] = None,
        train_steps: int = 1500,
        min_train: int = 8,
        explore_sigma: float = 0.08,
    ):
        super().__init__(space, objectives, seed)
        self.train_steps = train_steps
        self.min_train = min_train
        self.explore_sigma = explore_sigma
        self._model = None

    def _fit_surrogate(self, history: Sequence[ExperimentResult]):
        if not _HAVE_SK:
            return None, None, None
        pts = [(self.space.to_vector(r.params), self.objectives.scalarize(r.metrics))
               for r in history if r.success]
        if len(pts) < self.min_train:
            return None, None, None
        X = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        rf = RandomForestRegressor(
            n_estimators=150, max_depth=None, n_jobs=-1,
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        )
        rf.fit(X, y)
        return rf, float(y.mean()), float(y.std())

    def propose(self, batch_size: int, history: Sequence[ExperimentResult]) -> List[Experiment]:
        surrogate, y_ref, y_scale = self._fit_surrogate(history)
        if not _HAVE_RL or surrogate is None:
            tag = self.name + (":random_nodeps" if not _HAVE_RL else ":random_warmup")
            return [Experiment(params=self.space.sample(self.rng), proposed_by=tag)
                    for _ in range(batch_size)]

        try:
            env = _SurrogateEnv(self.space.dim, surrogate, y_ref, y_scale)
            model = SAC("MlpPolicy", env, verbose=0, seed=int(self.rng.integers(0, 2**31 - 1)),
                        learning_starts=min(64, self.train_steps // 4 + 1),
                        train_freq=1, gradient_steps=1)
            model.learn(total_timesteps=self.train_steps, progress_bar=False)
            base_action, _ = model.predict(np.zeros(1, dtype=np.float32), deterministic=True)
            base = np.clip(np.asarray(base_action, dtype=float), 0.0, 1.0)
        except Exception as e:  # any RL runtime failure must not kill the campaign
            logger.warning("RL training failed (%s); falling back to random proposals.", e)
            return [Experiment(params=self.space.sample(self.rng), proposed_by=self.name + ":random_fallback")
                    for _ in range(batch_size)]

        proposals: List[Experiment] = []
        for i in range(batch_size):
            vec = base if i == 0 else np.clip(base + self.rng.normal(0, self.explore_sigma, size=base.shape), 0, 1)
            params = self.space.from_vector(vec, self.rng)
            proposals.append(Experiment(params=params, proposed_by=self.name))
        return proposals
