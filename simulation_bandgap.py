# in some utils.py or simulation_bandgap.py

import numpy as np
from simulation import simulate_all_phy

# 1) Band‐gap estimator (Vegard’s law)
def estimate_band_gap(composition: dict) -> float:
    # reference bandgaps [eV] for pure-cation lead iodide perovskites
    Eg_FA = 1.48  
    Eg_MA = 1.60  
    Eg_Cs = 1.73  
    cs = composition.get("Cs", 0.0)
    ma = composition.get("MA", 0.0)
    fa = composition.get("FA", 0.0)
    # weighted average
    return cs * Eg_Cs + ma * Eg_MA + fa * Eg_FA

# 2) Simulation wrapper that applies bandgap‐based corrections  
def simulate_with_bandgap(params: dict) -> dict:
    # run surrogate + interface‐aware sim
    res = simulate_all_phy(params)
    base_eff = res["efficiency"]
    base_life = res["log_lifetime"]

    # compute composition band gap
    Eg = estimate_band_gap(params["composition"])

    # 3) Efficiency penalty for non-ideal bandgap
    ideal_Eg = 1.34
    # linear penalty: 20% efficiency loss per eV deviation  
    penalty = 1.0 - 0.2 * abs(Eg - ideal_Eg)
    eff = base_eff * max(penalty, 0.0)

    # 4) Lifetime boost for higher bandgap (assumes Eg_MA baseline)
    # replace Eg_MA by a constant or external reference
    Eg_MA = 1.60
    life = base_life + 0.1 * (Eg - Eg_MA)

    return {
        "efficiency": eff,
        "log_lifetime": life,
        **{k: v for k, v in res.items() if k not in ("efficiency","log_lifetime")}
    }
