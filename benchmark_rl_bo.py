# benchmark_enhanced.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import simulate_all_phy
from rl_bo_pipeline.physics_informed_bo import PhysicsInformedBO
from rl_bo_pipeline.evaluate_fixed_comp import RLEvaluator
from rl_bo_pipeline.tri_loop import TriLoop
from train_base_agent_rl_bo import train_base_agent

from layers import Layer
# default layer stack
default_layers = [
    Layer('Substrate', 1000, 0, 0, 0, 0),
    Layer('ETL',       50, 1e-12, 0.5, 1e9, 1e-5, mobility=1e-3, band_offset=0.2),
    Layer('Perov',    400, 1e-12, 0.5, 1e9, 1e-5),
    Layer('HTL',      100,1e-12, 0.5, 1e9, 1e-5, mobility=5e-4, band_offset=0.3),
    Layer('Ag',        80,   0,   0,    0,    0),
]


def benchmark_all(
    total_cycles: int,
    bo_batch: int,
    rl_steps: int,
    n_repeats: int = 5,
    seed: int = 0,
):
    np.random.seed(seed)
    strategies = ["random", "bo_only", "rl_only", "tri_loop"]
    # results[strat] will be shape (n_repeats, total_cycles)
    results = {s: np.zeros((n_repeats, total_cycles)) for s in strategies}

    for rep in range(n_repeats):
        print(f"=== Repeat {rep+1}/{n_repeats} ===")

        # 1) Random sampling
        for c in range(total_cycles):
            comps = np.random.rand(bo_batch, 3)
            best_score = -np.inf
            for cs, ma, fa in comps:
                out = simulate_all_phy({
                    "layers": default_layers,
                    "composition": {"Cs": cs, "MA": ma, "FA": fa},
                    "uv_filter": 0.5,
                    "spin_speed": 3500,
                    "anneal_T": 125,
                    "substrate": "Glass",
                    "encapsulation": False,
                })
                score = out["efficiency"] * out["log_lifetime"]
                best_score = max(best_score, score)
            results["random"][rep, c] = best_score

        # 2) BO only
        bo = PhysicsInformedBO(
            data_root="cluster-perovskite-data/Data",
            device="cpu"
        )
        for c in range(total_cycles):
            bo.update_gp()
            comps = bo.suggest(n_candidates=bo_batch)
            best_score = -np.inf
            lifetimes = []
            for cs, ma, fa in comps:
                out = simulate_all_phy({
                    "layers": default_layers,
                    "composition": {"Cs": cs, "MA": ma, "FA": fa},
                    "uv_filter": 0.5,
                    "spin_speed": 3500,
                    "anneal_T": 125,
                    "substrate": "Glass",
                    "encapsulation": False,
                })
                lifetimes.append(out["log_lifetime"])
                best_score = max(best_score, out["efficiency"] * out["log_lifetime"])
            results["bo_only"][rep, c] = best_score
            # feed back into BO
            bo.X = np.vstack([bo.X, comps])
            bo.y = np.vstack([bo.y, np.array(lifetimes).reshape(-1, 1)])

        # 3) RL only (fixed composition)
        base_comp = {"Cs": 0.33, "MA": 0.33, "FA": 0.34}
        for c in range(total_cycles):
            agent = train_base_agent(
                fixed_composition=base_comp,
                timesteps=rl_steps,
                n_envs=1
            )
            res, _ = RLEvaluator(base_comp).evaluate(agent, n_episodes=5)
            results["rl_only"][rep, c] = res["efficiency"] * res["log_lifetime"]

        # 4) Tri-loop (BO + RL)
        tri = TriLoop(
            base_agent_path="fab_agent.zip",
            data_root="cluster-perovskite-data/Data",
            bo_device="cpu",
            n_envs=1,
            vec_env="dummy",
        )
        history = []
        for c in range(total_cycles):
            hist = tri.run_cycle(cycle=c, bo_batch=bo_batch, rl_steps=rl_steps)
            # take only this cycle’s entries
            new_entries = [h for h in hist if h["cycle"] == c]
            best = max(new_entries, key=lambda h: h["eff"] * h["log_life"])
            results["tri_loop"][rep, c] = best["eff"] * best["log_life"]

    # compute mean & std for each strategy / cycle
    stats = {}
    for s in strategies:
        stats[s] = {
            "mean": results[s].mean(axis=0),
            "std":  results[s].std(axis=0),
        }
    df_mean = pd.DataFrame({s: stats[s]["mean"] for s in strategies})
    return results, stats, df_mean



