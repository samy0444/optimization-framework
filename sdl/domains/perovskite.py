"""Perovskite solar cell domain - the first real-field plugin.

This is a *physics prior* (cheap forward model), exactly the role `simulate()`
plays in the hybrid physics+ML strategy. It maps an A/B/X composition and a few
process knobs to device metrics (PCE, Voc, Jsc, FF) and a stability proxy (T80),
using standard perovskite relations:

  * bandgap   - Vegard-like mixing of endmember gaps + halide (I->Br) widening,
  * Jsc       - monotonic AM1.5 photocurrent envelope vs bandgap,
  * Voc       - radiative limit (Eg/q) minus an empirical non-radiative loss,
  * FF        - empirical baseline modulated by process quality,
  * T80       - composition/process stability proxy (MA hurts, Cs/Br help).

It is intentionally self-contained (no data files / API keys) so it runs
anywhere. With real data it is meant to be *corrected* by an ML residual model
trained on real data (the "delta" approach) - the framework supports that by
treating this as the prior and learning observed-minus-prior.

NOTE: coefficients are physically plausible but not calibrated to a specific
dataset; treat absolute numbers as illustrative, trends as meaningful.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from sdl.core.domain import Domain
from sdl.core.objective import Direction, Objective, ObjectiveSet
from sdl.core.registry import register_domain
from sdl.core.space import (
    ContinuousParameter,
    ParameterSpace,
    SimplexConstraint,
)

# Endmember optical bandgaps (eV) for A-site cation with PbI3.
_EG_AI = {"MA": 1.55, "FA": 1.48, "Cs": 1.73}
# Shannon-like effective ionic radii (A-site, X-site) for the tolerance factor.
_R_A = {"MA": 2.17, "FA": 2.53, "Cs": 1.88}      # angstrom (effective)
_R_X = {"I": 2.20, "Br": 1.96}
_R_B = 1.19  # Pb2+


@register_domain("perovskite")
class PerovskiteDomain(Domain):
    name = "perovskite"

    def parameter_space(self) -> ParameterSpace:
        params = [
            # A-site cation fractions (sum to 1)
            ContinuousParameter("MA", 0.0, 1.0),
            ContinuousParameter("FA", 0.0, 1.0),
            ContinuousParameter("Cs", 0.0, 1.0),
            # X-site halide fractions per formula unit of 3 (I + Br = 3)
            ContinuousParameter("I", 0.0, 3.0),
            ContinuousParameter("Br", 0.0, 3.0),
            # process knobs
            ContinuousParameter("anneal_temp", 90.0, 200.0),   # deg C
            ContinuousParameter("anneal_time", 5.0, 60.0),     # minutes
            ContinuousParameter("humidity", 0.0, 60.0),        # % RH during processing
        ]
        constraints = [
            SimplexConstraint(members=["MA", "FA", "Cs"], total=1.0),
            SimplexConstraint(members=["I", "Br"], total=3.0),
        ]
        return ParameterSpace(parameters=params, constraints=constraints)

    def objectives(self) -> ObjectiveSet:
        return ObjectiveSet(
            [
                Objective("pce", Direction.MAXIMIZE, weight=1.0, target=24.0, bound=0.0),
                Objective("t80", Direction.MAXIMIZE, weight=0.7, target=2000.0),
            ]
        )

    # --- physics prior ---
    def _bandgap(self, p: Dict[str, float]) -> float:
        a_tot = max(1e-9, p["MA"] + p["FA"] + p["Cs"])
        eg_i = sum(p[c] / a_tot * _EG_AI[c] for c in ("MA", "FA", "Cs"))
        br_frac = p["Br"] / max(1e-9, p["I"] + p["Br"])
        # halide widening with mild bowing (I->Br raises Eg by ~0.5 eV end to end)
        eg = eg_i + 0.50 * br_frac - 0.30 * br_frac * (1 - br_frac)
        return float(eg)

    def tolerance_factor(self, p: Dict[str, float]) -> float:
        a_tot = max(1e-9, p["MA"] + p["FA"] + p["Cs"])
        x_tot = max(1e-9, p["I"] + p["Br"])
        r_a = sum(p[c] / a_tot * _R_A[c] for c in ("MA", "FA", "Cs"))
        r_x = p["I"] / x_tot * _R_X["I"] + p["Br"] / x_tot * _R_X["Br"]
        return float((r_a + r_x) / (math.sqrt(2.0) * (_R_B + r_x)))

    def simulate(self, params: Dict[str, float]) -> Dict[str, float]:
        p = self.parameter_space().repair(params)
        eg = self._bandgap(p)

        # Jsc: smooth monotonic envelope of AM1.5 integrated current vs bandgap.
        jsc = float(np.clip(26.0 - 19.0 * (eg - 1.45), 8.0, 27.0))

        # Voc: radiative limit minus non-radiative loss; tolerance factor near 1 is best.
        tau = self.tolerance_factor(p)
        struct_penalty = 2.5 * (tau - 0.95) ** 2  # off-ideal structure -> extra loss
        v_loss = 0.40 + struct_penalty
        voc = float(np.clip(eg - v_loss, 0.4, eg - 0.30))

        # FF: process-dependent quality (annealing window + low humidity favored).
        t_opt = 150.0
        ff = 0.82 - 0.0010 * abs(p["anneal_temp"] - t_opt) - 0.0020 * p["humidity"]
        ff -= 0.05 * abs(min(p["anneal_time"], 40.0) - 20.0) / 20.0
        ff = float(np.clip(ff, 0.45, 0.85))

        pce = float(voc * jsc * ff)  # Pin = 100 mW/cm^2 => PCE(%) = Voc*Jsc*FF

        # T80 stability proxy (hours): MA hurts, Cs & moderate Br help; heat/humidity hurt.
        a_tot = max(1e-9, p["MA"] + p["FA"] + p["Cs"])
        ma_f, cs_f = p["MA"] / a_tot, p["Cs"] / a_tot
        br_f = p["Br"] / max(1e-9, p["I"] + p["Br"])
        base = 1500.0
        t80 = base * (1.0 - 0.55 * ma_f) * (1.0 + 0.6 * cs_f) * (1.0 + 0.4 * min(br_f, 0.3) / 0.3)
        t80 *= math.exp(-0.015 * max(0.0, p["humidity"] - 20.0))
        t80 *= math.exp(-0.004 * max(0.0, p["anneal_temp"] - 160.0))
        t80 = float(max(1.0, t80))

        return {"pce": pce, "voc": voc, "jsc": jsc, "ff": ff, "t80": t80, "bandgap": eg}

    def ground_truth(self, params: Dict[str, float]) -> Dict[str, float]:
        """Reality = the smooth prior plus structured effects it omits.

        The cheap simulate() is deliberately too smooth. Reality adds physics
        that practitioners know but a closed-form prior misses, concentrated in
        specific regions so the residual is *learnable* rather than noise:

          * Hoke-effect halide segregation: a Voc penalty peaked at intermediate
            Br fraction (the prior's halide term is monotonic and misses it),
          * a Br-dependent FF haircut, and
          * faster moisture-driven stability loss than the prior's exponential.

        This is the gap the delta-GP recovers; vanilla BO has to relearn the
        whole landscape instead.
        """
        p = self.parameter_space().repair(params)
        m = dict(self.simulate(p))
        br_f = p["Br"] / max(1e-9, p["I"] + p["Br"])
        seg = 0.12 * math.exp(-((br_f - 0.5) ** 2) / (2 * 0.15 ** 2))
        voc = max(0.2, m["voc"] - seg)
        ff = m["ff"] * (1.0 - 0.04 * br_f)
        m["voc"], m["ff"] = voc, ff
        m["pce"] = float(voc * m["jsc"] * ff)
        m["t80"] = float(max(1.0, m["t80"] * math.exp(-0.01 * max(0.0, p["humidity"] - 15.0))))
        return m

    def featurize(self, params: Dict[str, float]) -> np.ndarray:
        p = self.parameter_space().repair(params)
        base = self.parameter_space().to_vector(p)
        extra = np.array([self._bandgap(p), self.tolerance_factor(p)], dtype=float)
        return np.concatenate([base, extra])

    def feature_names(self):
        return list(self.parameter_space().names) + ["bandgap", "tolerance_factor"]
