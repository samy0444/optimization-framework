# optimization_bo.py

import numpy as np
import pandas as pd
from rl_bo_pipeline.physics_informed_bo import PhysicsInformedBO
# from simulation import simulate_all_phy  # your digital-twin surrogate + PINN wrapper
from simulation_bandgap import simulate_with_bandgap
from layers import Layer
# app.py (Streamlit excerpt)
from interfaces import Interface

etl_if  = Interface(name="ETL/Perovskite", recomb_velocity=1e3, trap_density=1e10)
perh_if = Interface(name="Perovskite/HTL", recomb_velocity=5e2, trap_density=5e9)


default_layers = [
    Layer('Substrate', 1000, 0, 0, 0, 0),
    Layer('ETL',       50, 1e-12, 0.5, 1e9, 1e-5, mobility=1e-3, band_offset=0.2),
    Layer('Perov',    400, 1e-12, 0.5, 1e9, 1e-5),
    Layer('HTL',      100,1e-12, 0.5, 1e9, 1e-5, mobility=5e-4, band_offset=0.3),
    Layer('Ag',        80,   0,   0,    0,    0),
]
def run_bo(
    num_rounds: int = 5,
    material_batch: int = 10,
    data_root: str = "cluster-perovskite-data/Data",
):
    """
    Hierarchical material-level BO using the real MIT data.
    At each round:
      1) fit GP on measured instability × purity
      2) suggest `material_batch` new Cs/MA/FA triples in [0,1]
      3) evaluate them with simulate_all_phy 
      4) append back into BO for the next round
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

        # 3) evaluate each suggestion via your digital‐twin simulate_all_phy
        insts = []
        for cs, ma, fa in new_pts:               # ← use new_pts, not new_CsMAFA
            params = {
                "layers": default_layers,   # your standard five-layer list
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

        # 4) append back into GP’s training data
        bo.X = np.vstack([bo.X, new_pts])
        bo.y = np.vstack([bo.y, insts.reshape(-1, 1)])

        # print current best
        best_i = np.argmin(bo.y.flatten())
        best = bo.X[best_i]
        print(f"  Best so far: Cs={best[0]:.3f}, MA={best[1]:.3f}, FA={best[2]:.3f}")

    return comp_history, sim_history





# # optimization_bo.py (excerpt)

# from physics_informed_bo import PhysicsInformedBO
# import numpy as np
# import pandas as pd

# # optimization_bo.py - Updated integration

# def run_bo(num_rounds=5, material_batch=10):
#     """
#     Run hierarchical BO using actual MIT data structure
#     """
#     # Use actual MIT data
#     bo = PhysicsInformedBO(
#         xrd_compositions_csv="cluster-perovskite-data/Data/XRD/xrd_data_compositions.csv",
#         beta_dft=0.025
#     )
    
#     comp_history, inst_history = [], []
    
#     for round_num in range(num_rounds):
#         print(f"BO Round {round_num + 1}")
#         comps, insts = bo.run_round(material_batch)
#         comp_history.append(comps)
#         inst_history.append(insts)
        
#         # Best composition so far
#         best_idx = np.argmin(bo.y)
#         best_comp = bo.X[best_idx]
#         print(f"Best composition: Cs{best_comp[0]:.1f}MA{best_comp[1]:.1f}FA{best_comp[2]:.1f}")
    
#     return comp_history, inst_history

# def run_bo(num_iters=20, init_size=20):
#     # --- Load DFT mix energies ---
#     dft_df = pd.read_csv('Data/Metadata/dft_data.csv')  # must include ΔG_mix
#     dft_data = { (r.Cs, r.MA, r.FA): r.delta_G_mix for _,r in dft_df.iterrows() }

#     # --- Instantiate physics‐informed BO ---
#     bo = PhysicsInformedBO(
#         composition_space={'Cs':(0,100),'MA':(0,100),'FA':(0,100)},
#         dft_data=dft_data,
#         beta_dft=0.025
#     )

#     # --- Seed with experimental dataset ---
#     exp_df = pd.read_csv('Data/Metadata/results.csv')
#     bo.experimental_data['compositions'] = exp_df[['Cs','MA','FA']].values.tolist()
#     bo.experimental_data['instability_indices'] = exp_df.instability_index.tolist()

#     # --- Run BO-driven composition optimization ---
#     results = bo.run_optimization(max_rounds=num_iters, batch_size=init_size)
#     return results
