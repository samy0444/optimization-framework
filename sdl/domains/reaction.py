"""Reaction / formulation optimization domain (simulated).

Optimizes a chemical reaction's yield and purity over process conditions
(temperature, time, catalyst loading, stoichiometry, concentration, solvent).
Each experiment in this setting is a short assay rather than a multi-month
device build, which makes it a practical testbed for "fewer experiments to
target" comparisons.

It carries a *cheap physics prior*, which is the setting the framework's
delta-GP is designed for: classical reaction engineering already gives a good
first guess.

  * conversion - first-order kinetics, Arrhenius rate k(T) modulated by
    catalyst loading (saturating), concentration and solvent,
  * selectivity - drops with temperature and excess reagent (side reactions),
  * yield      - 100 * conversion * selectivity,
  * purity     - desired-product fraction (selectivity expressed as %).

`simulate()` is intentionally smooth. `ground_truth()` adds effects a textbook
prior misses (catalyst deactivation at high T, a solvent-specific impurity, a
stoichiometry x concentration interaction) - the learnable residual the
physics-prior optimizer recovers while vanilla BO must relearn from scratch.

Coefficients are physically plausible, not calibrated to one reaction; trends
are meaningful, absolute numbers illustrative. With real experiments,
ground_truth() becomes real plate/HPLC data and the delta-GP corrects the prior
in place.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from sdl.core.domain import Domain
from sdl.core.objective import Direction, Objective, ObjectiveSet
from sdl.core.registry import register_domain
from sdl.core.space import (
    CategoricalParameter,
    ContinuousParameter,
    ParameterSpace,
)

_R = 8.314                       # J / mol / K
_A = 7.6e6                       # Arrhenius pre-exponential (1/h), tuned for sane range
_EA = 5.0e4                      # activation energy (J/mol)
# Relative rate multipliers per solvent (illustrative reaction-engineering priors).
_SOLVENT_RATE = {"toluene": 1.0, "THF": 1.2, "MeCN": 1.5, "DMF": 1.8, "water": 0.6}


@register_domain("reaction")
class ReactionDomain(Domain):
    name = "reaction"

    def parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            parameters=[
                ContinuousParameter("temperature", 20.0, 150.0),    # deg C
                ContinuousParameter("time", 1.0, 24.0),             # hours
                ContinuousParameter("catalyst", 0.1, 10.0),         # mol %
                ContinuousParameter("equiv", 1.0, 3.0),             # reagent equivalents
                ContinuousParameter("concentration", 0.05, 2.0),    # mol / L
                CategoricalParameter("solvent", ["toluene", "THF", "MeCN", "DMF", "water"]),
            ]
        )

    def objectives(self) -> ObjectiveSet:
        return ObjectiveSet(
            [
                Objective("yield", Direction.MAXIMIZE, weight=1.0, target=90.0, bound=0.0),
                Objective("purity", Direction.MAXIMIZE, weight=0.6, target=99.0),
            ]
        )

    # --- physics prior ---
    def _conversion(self, p: Dict[str, float]) -> float:
        t_k = float(p["temperature"]) + 273.15
        k = _A * math.exp(-_EA / (_R * t_k))                       # Arrhenius rate (1/h)
        cat_factor = p["catalyst"] / (p["catalyst"] + 1.0)         # saturating in loading
        solv = _SOLVENT_RATE.get(p["solvent"], 1.0)
        k_eff = k * cat_factor * solv * float(p["concentration"]) * min(float(p["equiv"]), 2.0)
        return float(1.0 - math.exp(-k_eff * float(p["time"])))    # first-order conversion

    def _selectivity(self, p: Dict[str, float]) -> float:
        t_c = float(p["temperature"])
        sel = 0.99 - 0.0020 * max(0.0, t_c - 60.0) - 0.05 * (float(p["equiv"]) - 1.0)
        return float(np.clip(sel, 0.5, 0.99))

    def simulate(self, params: Dict[str, float]) -> Dict[str, float]:
        p = self.parameter_space().repair(params)
        conv = self._conversion(p)
        sel = self._selectivity(p)
        yield_pct = float(100.0 * conv * sel)
        purity_pct = float(100.0 * sel)
        return {"yield": yield_pct, "purity": purity_pct,
                "conversion": float(100.0 * conv), "selectivity": purity_pct}

    def ground_truth(self, params: Dict[str, float]) -> Dict[str, float]:
        """Reality = the smooth prior plus effects a textbook model omits.

          * catalyst deactivation above ~110 C (the Arrhenius prior is monotonic
            in T and misses the roll-off),
          * a solvent-specific impurity (DMF/water) the prior doesn't model,
          * a stoichiometry x concentration interaction that breeds side product.

        These are localized, structured residuals -> learnable by the delta-GP,
        invisible to a prior-free model until it spends experiments there.
        """
        p = self.parameter_space().repair(params)
        conv = self._conversion(p)
        sel = self._selectivity(p)

        t_c = float(p["temperature"])
        deact = 1.0 / (1.0 + math.exp(-(t_c - 110.0) / 8.0))       # logistic onset ~110 C
        conv *= (1.0 - 0.45 * deact)

        impurity = {"DMF": 0.08, "water": 0.04}.get(p["solvent"], 0.0)
        interaction = 0.03 * max(0.0, (float(p["equiv"]) - 2.0) * (float(p["concentration"]) - 1.0))
        sel = float(np.clip(sel - impurity - interaction, 0.4, 0.99))

        yield_pct = float(100.0 * conv * sel)
        purity_pct = float(100.0 * sel)
        return {"yield": yield_pct, "purity": purity_pct,
                "conversion": float(100.0 * conv), "selectivity": purity_pct}

    def featurize(self, params: Dict[str, float]) -> np.ndarray:
        p = self.parameter_space().repair(params)
        base = self.parameter_space().to_vector(p)
        extra = np.array([self._conversion(p), self._selectivity(p)], dtype=float)
        return np.concatenate([base, extra])

    def feature_names(self):
        return list(self.parameter_space().names) + ["prior_conversion", "prior_selectivity"]
