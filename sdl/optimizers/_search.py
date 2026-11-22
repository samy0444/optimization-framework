"""Shared candidate-selection helpers for the Bayesian optimizers.

Two reliability levers live here, used by both ``bayesian`` and ``physics_bo``:

1. **Full-grid candidates when the space is finite.** If every parameter is
   categorical/integer (e.g. "which ligand / base / additive"), we score *every*
   real combination each round instead of a random subset. Random subsets can
   simply miss the best reaction, which is what made the optimizer occasionally
   fail to reach the target. For continuous spaces we fall back to a deduplicated
   random pool.
2. **No repeats.** We remember every point ever proposed (and everything already
   measured) and exclude it from future candidate pools, so we never waste an
   experiment re-running something we've already seen.

A third lever (batch spread) is provided by ``greedy_diverse_batch``: pick the
batch greedily by acquisition score but penalize candidates near ones already
picked this round, so a batch of N explores N different regions instead of
clustering on one peak.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from sdl.core.experiment import ExperimentResult
from sdl.core.space import ParameterSpace, Params


def vector_key(vec: Sequence[float], dp: int = 6) -> Tuple:
    """Hashable rounded key for a unit-cube vector (for dedup)."""
    return tuple(np.round(np.asarray(vec, dtype=float), dp))


def build_grid(space: ParameterSpace, max_points: int = 200_000):
    """Precompute the finite grid once: list of (params, key) or None if the
    space is continuous / too large to enumerate."""
    grid = space.enumerate_grid(max_points=max_points)
    if grid is None:
        return None
    return [(p, vector_key(space.to_vector(p))) for p in grid]


def seen_keys(space: ParameterSpace, history: Sequence[ExperimentResult]) -> Set[Tuple]:
    """Keys of all successfully-measured points (so we don't re-propose them)."""
    out: Set[Tuple] = set()
    for r in history:
        if getattr(r, "success", False):
            out.add(vector_key(space.to_vector(r.params)))
    return out


def candidate_pool(
    space: ParameterSpace,
    pool_size: int,
    rng: np.random.Generator,
    exclude: Set[Tuple],
    grid: Optional[List[Tuple[Params, Tuple]]],
) -> List[Params]:
    """Candidate parameter dicts to score this round.

    Finite space -> the full grid minus already-seen points. Continuous space ->
    a deduplicated random sample of ~pool_size points. Never returns empty: if
    everything has been seen, it relaxes the exclusion rather than stalling.
    """
    if grid is not None:
        cands = [p for (p, k) in grid if k not in exclude]
        if not cands:                          # whole grid exhausted -> allow repeats
            cands = [p for (p, _) in grid]
        return cands

    out: List[Params] = []
    local: Set[Tuple] = set(exclude)
    tries, cap = 0, pool_size * 5
    while len(out) < pool_size and tries < cap:
        p = space.sample(rng)
        k = vector_key(space.to_vector(p))
        tries += 1
        if k in local:
            continue
        local.add(k)
        out.append(p)
    if not out:                                # degenerate: fall back to plain sample
        out = space.sample_batch(pool_size, rng)
    return out


def greedy_diverse_batch(
    X: np.ndarray,
    scores: np.ndarray,
    batch_size: int,
    diversity_len: float = 0.2,
) -> List[int]:
    """Indices chosen greedily by acquisition score, with local penalization so
    the batch spreads out. diversity_len is a distance scale in unit-cube units;
    set 0 to disable (pure top-k)."""
    s = np.asarray(scores, dtype=float).copy()
    scale = float(np.max(np.abs(scores))) or 1.0
    chosen: List[int] = []
    for _ in range(batch_size):
        if not np.isfinite(s).any():
            break
        i = int(np.argmax(s))
        chosen.append(i)
        s[i] = -np.inf
        if diversity_len and diversity_len > 0:
            d2 = np.sum((X - X[i]) ** 2, axis=1)
            s = s - scale * np.exp(-d2 / (2.0 * diversity_len ** 2))
            s[chosen] = -np.inf
    return chosen
