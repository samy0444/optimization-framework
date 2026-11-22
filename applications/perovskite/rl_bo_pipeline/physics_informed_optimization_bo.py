# optimization_bo.py

import numpy as np
import pandas as pd
from rl_bo_pipeline.physics_informed_bo import PhysicsInformedBO
# from simulation import simulate_all_phy  # digital-twin surrogate + PINN wrapper
from simulation_bandgap import simulate_with_bandgap
from layers import default_stack
# app.py (Streamlit excerpt)
from interfaces import Interface

etl_if  = Interface(name="ETL/Perovskite", recomb_velocity=1e3, trap_density=1e10)
perh_if = Interface(name="Perovskite/HTL", recomb_velocity=5e2, trap_density=5e9)


default_layers = default_stack()
def run_bo(
    num_rounds: int = 5,
    material_batch: int = 10,
    data_root: str = "cluster-perovskite-data/Data",
):
    """
    Hierarchical material-level BO using the real MIT data.
    At each round:
      1) fit the GP (on measured MIT stability until enough results exist,
         then on this function's own results)
      2) suggest `material_batch` new (Cs, FA, MA) compositions summing to 1
      3) evaluate them with simulate_with_bandgap
      4) feed the results back into the BO for the next round
    Returns:
      comp_history: list of np.array(material_batch×3) for each round
      sim_history:  list of np.array(material_batch) predicted log-lifetimes
    """
    bo = PhysicsInformedBO(data_root=data_root, device="cpu")

    comp_history = []
    sim_history  = []

    for r in range(num_rounds):
        print(f"=== BO Round {r+1} ===")
        # 1) fit/update GP on current data
        bo.update_gp()

        # 2) suggest new candidates
        new_pts = bo.suggest(n_candidates=material_batch)  # shape (batch, 3)
        comp_history.append(new_pts)

        # 3) evaluate each suggestion via digital‐twin simulate_all_phy
        insts = []
        for cs, fa, ma in new_pts:               # BO columns are Cs, FA, MA
            params = {
                "layers": default_layers,   # standard five-layer list
                # quick‐and‐dirty map first fraction → Pb:
                "composition": {"Pb": cs, "Sn": 1.0 - cs},
                "uv_filter": 0.0,
                "spin_speed": 3500.0,
                "anneal_T": 125.0,
                "substrate": "Glass",
                "encapsulation": False,
                "interfaces": [etl_if, perh_if]
            }
            res = simulate_with_bandgap(params)
            insts.append(res["log_lifetime"])
        insts = np.array(insts)
        sim_history.append(insts)

        # 4) feed results back into the BO (higher log-lifetime is better)
        bo.add_observations(new_pts, insts)

        # print current best
        best_i = int(np.argmax(bo.y_obs.flatten()))
        best = bo.X_obs[best_i]
        print(f"  Best so far: Cs={best[0]:.3f}, FA={best[1]:.3f}, MA={best[2]:.3f}")

    return comp_history, sim_history



