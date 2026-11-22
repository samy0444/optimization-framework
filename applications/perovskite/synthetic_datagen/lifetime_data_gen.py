

#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import deepxde as dde
from dataclasses import dataclass
from tqdm import trange

# ─── PINN SETUP ───────────────────────────────────────────────────────────────
os.environ["DDE_BACKEND"]   = "pytorch"
os.environ["DDE_DEVICE_ID"] = "0"
dde.config.default_backend   = "pytorch"
dde.config.default_device_id = 0

D, H, T_final = 1e-12, 1.0, 1.0
geom = dde.geometry.GeometryXTime(
    dde.geometry.Rectangle([0, 0], [1, H]),
    dde.geometry.TimeDomain(0, T_final),
)
def pde(x, C):
    Ct  = dde.grad.jacobian(C, x, i=0, j=2)
    Cxx = dde.grad.hessian(C, x, component=0, i=0, j=0)
    Cyy = dde.grad.hessian(C, x, component=0, i=1, j=1)
    return Ct - D*(Cxx + Cyy)

ic   = dde.IC(geom, lambda x: 0.0, lambda _, on_i: on_i)
bc_l = dde.DirichletBC(
    geom, lambda x: 1.0, lambda x, on_b: on_b and np.isclose(x[0],0.0)
)
data = dde.data.TimePDE(geom, pde, [ic, bc_l], num_domain=1, num_boundary=1, num_initial=1)
net  = dde.maps.FNN([3] + [64]*4 + [1], "tanh", "Glorot uniform")
pinn = dde.Model(data, net)
pinn.compile("L-BFGS")
pinn.restore("synthetic_datagen/pinn_deepxde_model-25000.pt")

def lifetime_L2_pinn(stack, Ccrit=0.1, samples=100):
    H_real = sum(ly.L_nm for ly in stack)*1e-9
    Ys = np.linspace(0, H_real, samples)
    X  = np.stack([np.zeros_like(Ys), Ys, np.ones_like(Ys)*T_final], axis=1)
    C  = pinn.predict(X)[:,0]
    idx = np.argmax(C >= Ccrit)
    return (idx/(samples-1))*T_final

# ─── STACK DEFINITION ────────────────────────────────────────────────────────
@dataclass
class Layer:
    name: str; L_nm: float; D0: float; Ea: float; E: float; CTE: float

def sample_stack(rng):
    logD_min, logD_max = -16, -6
    specs = [
      ("FTO",   (500,700),(0.4,0.5),(60,80),(8e-6,10e-6)),
      ("HTL",   (50,250),(0.4,0.6),(2,4),(50e-6,70e-6)),
      ("Perov", (400,700),(0.3,0.5),(10,20),(30e-6,40e-6)),
      ("ETL",   (40,120),(0.5,0.7),(100,130),(8e-6,12e-6)),
      ("Ag",    (70,130),(0.6,0.8),(75,90),(18e-6,20e-6)),
    ]
    out = []
    for name, Lr, Ear, Er, Cr in specs:
        out.append(Layer(
            name,
            rng.uniform(*Lr),
            10**rng.uniform(logD_min, logD_max),
            rng.uniform(*Ear),
            rng.uniform(*Er),
            rng.uniform(*Cr),
        ))
    return out

def stack_to_features(stack):
    # flatten each layer into 6 features → total 30
    return [val for ly in stack for val in (
        ly.L_nm,
        ly.D0,
        np.log10(ly.D0),
        ly.Ea,
        ly.E,
        ly.CTE,
    )]

# ─── GPU‐BATCHED 1D SOLVER ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_batch_torch(stacks, dx_nm=20):
    B = len(stacks)
    # compute maximum grid length
    Nmax = max(sum(max(1, int(round(ly.L_nm/dx_nm))) for ly in s) for s in stacks)
    D_list = []
    T_K = torch.empty(1).uniform_(298,358).item()
    kB  = 8.617e-5
    for s in stacks:
        vals = []
        for ly in s:
            n = max(1, int(round(ly.L_nm/dx_nm)))
            σ = ly.E*1e9*(ly.CTE - 12e-6)*(T_K-298)
            D_T = ly.D0 * np.exp(-ly.Ea/(kB*T_K))
            vals += [D_T*(1+1e-6*σ)]*n
        vals += [vals[-1]]*(Nmax-len(vals))
        D_list.append(vals)
    D_batch = torch.tensor(D_list, device=device)
    dx = dx_nm*1e-7
    return D_batch.float(), dx


def lifetime_batch_1d(D_batch, dx, dt=0.1, Ccrit=0.1, max_steps=5000):
    
    B, N = D_batch.shape
    r   = D_batch * (dt/dx**2)
    a0  = 1.0 + r
    b0  = 1.0 - r

    # pre‐allocate
    C   = torch.zeros((B, N), device=device)
    C[:, 0] = 1.0
    out = torch.full((B,), float("inf"), device=device)

    for t in range(max_steps):
        C_prev = C

        # Jacobi update on interior points using standard finite‐difference stencil
        numer = (
            b0[:,1:-1] * C_prev[:,1:-1]
            - 0.5 * r[:,1:-1] * (C_prev[:,:-2] + C_prev[:,2:])
        )
        C[:,1:-1] = (numer / a0[:,1:-1]).clamp(0.0, 1.0)

        # apply boundary conditions explicitly
        C[:, 0] = 1.0
        # C[:,-1] remains whatever, we check it below

        # record when the far boundary crosses Ccrit
        done = (C[:, -1] >= Ccrit) & (out == float("inf"))
        out[done] = (t+1)*dt

        if torch.all(out != float("inf")):
            break

    out_np = out.cpu().numpy()
    # replace any that never finished with the max possible time:
    out_np[np.isinf(out_np)] = max_steps * dt
    return out_np


# ─── ANALYTIC L1 ────────────────────────────────────────────────────────────
def lifetime_L1(stack, alpha=1e12, tau0=1e7):
    L = np.array([ly.L_nm*1e-7 for ly in stack])
    D = np.array([ly.D0     for ly in stack])
    return tau0*np.exp(-alpha*(L*D).mean())

# ─── BATCHED DATASET BUILDER ────────────────────────────────────────────────
def build_dataset(N=20000, batch_size=128, seed=42):
    rng  = np.random.default_rng(seed)
    rows = []
    for i in trange(0, N, batch_size):
        B = min(batch_size, N-i)
        stacks = [sample_stack(rng) for _ in range(B)]
        F30    = np.vstack([stack_to_features(s) for s in stacks])
        # L1 vector
        t1 = np.array([lifetime_L1(s) for s in stacks])
        # L2‐1D vector (GPU batched)
        D_batch, dx = prepare_batch_torch(stacks)
        t2 = lifetime_batch_1d(D_batch, dx)
        # PINN (still Python loop, but only 128 calls)
        t3 = np.array([lifetime_L2_pinn(s) for s in stacks])
        # combine
           # combine
        combined = 0.3*t1 + 0.5*t2 + 0.2*t3
        combined = np.clip(combined, 1e-12, None)
        log_life = np.log10(combined)

        # log_life = np.log10(0.3*t1 + 0.5*t2 + 0.2*t3 + 1e-12)
        rows.append(np.hstack([F30, log_life[:,None]]))

    cols = [f"{p}_{j}" for p in ("L","D0","logD0","Ea","E","CTE") for j in range(5)]
    cols += ["target"]
    return pd.DataFrame(np.vstack(rows), columns=cols)

if __name__=="__main__":
    df = build_dataset()
    print(df["target"].describe())
    df.to_csv("encap_multifidelity.csv", index=False)
    print("→ Saved encap_multifidelity.csv")
