#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Toy efficiency factors
# ──────────────────────────────────────────────────────────────────────────────
def perovskite_efficiency(pb: np.ndarray) -> np.ndarray:
    return 0.9 - 0.1 * np.abs(pb - 0.5)

def uv_degradation(uv: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 5 * uv)

def spin_penalty(spin: np.ndarray) -> np.ndarray:
    return 1.0 - 0.00002 * np.abs(spin - 3000)

def anneal_penalty(ann: np.ndarray) -> np.ndarray:
    return 1.0 - 0.001 * np.abs(ann - 400)

# ──────────────────────────────────────────────────────────────────────────────
# Sample 5‐layer “stack” base features: thickness, D0, logD0, Ea, E, CTE
# produces shape (N, 30)
# ──────────────────────────────────────────────────────────────────────────────
def sample_stack_base(N, seed=0):
    rng = np.random.default_rng(seed)
    # each is (N,5)
    L_nm   = rng.uniform(50, 800, size=(N,5))
    D0     = 10 ** rng.uniform(-16, -6, size=(N,5))
    logD0  = np.log10(D0)
    Ea     = rng.uniform(2, 20, size=(N,5))
    E      = rng.uniform(30e6, 130e6, size=(N,5))
    CTE    = rng.uniform(8e-6, 20e-6, size=(N,5))
    # stack into (N,30)
    return np.stack([L_nm, D0, logD0, Ea, E, CTE], axis=2).reshape(N,30)

# ──────────────────────────────────────────────────────────────────────────────
# Build 32-dim dataset
# ──────────────────────────────────────────────────────────────────────────────
def build_efficiency_dataset_32d(N=50000, seed=42):
    rng = np.random.default_rng(seed)

    # 1) composition & UV
    Pb = rng.random(N)
    uv = rng.random(N)

    # 2) process knobs
    spin = rng.uniform(1000, 5000, size=N)
    ann  = rng.uniform(350,  450,  size=N)

    # 3) compute toy efficiency
    eff = perovskite_efficiency(Pb)
    eff *= uv_degradation(uv)
    eff *= spin_penalty(spin)
    eff *= anneal_penalty(ann)
    eff = np.clip(eff, 0.0, 1.0)

    # 4) sample stack features
    base30 = sample_stack_base(N, seed)

    # 5) drop two unneeded stack dims (e.g. indices 6 & 7)
    mask = np.ones(30, dtype=bool)
    mask[6:8] = False
    base28 = base30[:, mask]   # now shape (N,28)

    # 6) assemble final (N,32)
    X = np.concatenate([base28,
                        Pb.reshape(-1,1),
                        uv.reshape(-1,1),
                        spin.reshape(-1,1),
                        ann.reshape(-1,1)], axis=1)

    # 7) column names
    cols = []
    for i,name in enumerate(["L","D0","logD0","Ea","E","CTE"]):
        for j in range(5):
            cols.append(f"{name}_{j}")
    # drop the two that correspond to mask[6:8]
    del cols[6]; del cols[6]
    cols += ["Pb_fraction","UV_filter","spin_rpm","anneal_temp"]

    df = pd.DataFrame(X, columns=cols)
    df["efficiency"] = eff
    return df

if __name__=="__main__":
    df = build_efficiency_dataset_32d()
    print("Efficiency distribution:\n", df["efficiency"].describe())
    df.to_csv("efficiency_dataset_32d.csv", index=False)
    print("→ Saved efficiency_dataset_32d.csv")
