# simulation.py
import os
import numpy as np
import torch
import deepxde as dde
from dataclasses import dataclass
from layers import Layer
from surrogate_model import SurrogateModel
# load once
surrogate = SurrogateModel(device="cpu")
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PINN setup
os.environ["DDE_BACKEND"] = "pytorch"
os.environ["DDE_DEVICE_ID"] = "0"
dde.config.default_backend = "pytorch"
dde.config.default_device_id = 0

D, H, T_final = 1e-12, 1.0, 1.0
geom_space = dde.geometry.Rectangle([0, 0], [1, H])
geom_time  = dde.geometry.TimeDomain(0, T_final)
geom       = dde.geometry.GeometryXTime(geom_space, geom_time)

# Define PDE

def pde(x, C):
    Ct  = dde.grad.jacobian(C, x, i=0, j=2)
    Cxx = dde.grad.hessian(C, x, component=0, i=0, j=0)
    Cyy = dde.grad.hessian(C, x, component=0, i=1, j=1)
    return Ct - D*(Cxx + Cyy)

# Boundary & initial conditions
ic      = dde.IC(geom, lambda x: 0.0, lambda _ , on_initial: on_initial)
bc_left = dde.DirichletBC(geom,
    lambda x: 1.0,
    lambda x,on_b: on_b and np.isclose(x[0],0.0)
)

data = dde.data.TimePDE(geom, pde, [ic, bc_left],
                        num_domain=1, num_boundary=1, num_initial=1)
net = dde.maps.FNN([3] + [64]*4 + [1], "tanh", "Glorot uniform")
pinn = dde.Model(data, net)
pinn.compile("L-BFGS")
pinn.restore("pinn_deepxde_model-25000.pt")
# ensure PINN weights on correct device
pinn.net.to(device)



# Lifetime from analytic L1

def lifetime_L1(stack, alpha=1e12, tau0=1e7):
    L_cm = np.array([ly.L_nm*1e-7 for ly in stack])
    D_arr= np.array([ly.D0      for ly in stack])
    return tau0*np.exp(-alpha*(D_arr*L_cm).mean())

# GPU‐accelerated 1D solver

def prepare_batch_torch(stacks, dx_nm=20):
    dx    = dx_nm*1e-7
    B     = len(stacks)
    Nmax  = max(sum(max(1,int(round(ly.L_nm/dx_nm))) for ly in stk) for stk in stacks)
    D_list=[]
    T_K   = torch.empty(1).uniform_(298,358).item()
    kB    = 8.617e-5
    for stk in stacks:
        vals=[]
        for ly in stk:
            n   = max(1,int(round(ly.L_nm/dx_nm)))
            σ   = ly.E*1e9*(ly.CTE-12e-6)*(T_K-298)
            D_T = ly.D0 * np.exp(-ly.Ea/(kB*T_K))
            vals += [D_T*(1+1e-6*σ)]*n
        vals += [vals[-1]]*(Nmax-len(vals))
        D_list.append(vals)
    return torch.tensor(D_list,dtype=torch.float32,device=device), dx


def lifetime_L2_1d_torch(D_batch, dx, dt=0.1, Ccrit=0.1, max_steps=20000):
    B,N = D_batch.shape
    r   = D_batch*(dt/dx**2)
    a0  = 1.0+r
    b0  = 1.0-r
    off = -0.5*r[:,1:]
    C   = torch.zeros_like(D_batch,device=device)
    C[:,0]=1.0
    out = torch.full((B,),float('inf'),device=device)
    for t in range(max_steps):
        rhs = b0*C
        rhs[:,1:-1] -= off[:,1:]*C[:,2:] + off[:,:-1]*C[:,:-2]
        rhs[:,0]    = 1.0
        Cnew        = C.clone()
        Cnew[:,1:-1] = (rhs[:,1:-1] - off[:,1:]*C[:,:-2] - off[:,:-1]*C[:,2:]) / a0[:,1:-1]
        C = torch.clamp(Cnew,0.0,1.0)
        done = (C[:,-1]>=Ccrit)&(out==float('inf'))
        out[done] = (t+1)*dt
        if torch.all(out!=float('inf')):
            break
    return out.cpu().numpy()

# PINN-based lifetime

def lifetime_L2_pinn(stack, Ccrit=0.1, samples=100):
    H_real = sum(ly.L_nm for ly in stack)*1e-9
    Ys     = np.linspace(0,H_real,samples)
    X      = np.stack([np.zeros_like(Ys),Ys,np.ones_like(Ys)*T_final],axis=1)
    X_t    = torch.tensor(X,dtype=torch.float32,device=device)
    with torch.no_grad():
        C_t = pinn.net(X_t)
    C   = C_t[:,0].cpu().numpy()
    idx = np.argmax(C>=Ccrit) if np.any(C>=Ccrit) else samples-1
    return (idx/(samples-1))*T_final

# Assemble features + run both surrogates
# computationally expensive simulation
# def simulate_all(params: dict) -> dict:
#     stack = params['layers']
#     # lifetimes
#     t1 = lifetime_L1(stack)
#     D_batch,dx = prepare_batch_torch([stack])
#     t2 = lifetime_L2_1d_torch(D_batch,dx)[0]
#     t3 = lifetime_L2_pinn(stack)
#     base_log_life = np.log10(0.3*t1 + 0.5*t2 + 0.2*t3 + 1e-12)

#     # prepare 32-d features
#     feat30 = np.hstack([[
#         ly.L_nm,
#         ly.D0,
#         np.log10(ly.D0 if ly.D0>0 else 1e-12),
#         ly.Ea,
#         ly.E,
#         ly.CTE
#     ] for ly in stack])
#     comp   = np.array([params['composition']['Pb'], 1-params['composition']['Pb']])
#     feat32 = np.concatenate([feat30, comp])
#     feat32 = feat32.reshape(1,-1)
#     feat32 = np.nan_to_num(feat32, nan=0.0, posinf=1e6, neginf=-1e6)

#     eff_pred, life_pred = surrogate.predict(feat32)
#     eff = float(eff_pred[0])
#     log_life = float(life_pred[0])

#     # encapsulation optical loss
#     if params.get('encapsulation', False):
#         eff *= 0.98

#     # simple IV curve
#     V_oc, J_sc, FF = eff*1.2, eff*30.0, 0.75
#     V = np.linspace(0, V_oc, 50)
#     I = J_sc*(1 - V/V_oc)*FF
#     iv_curve = np.stack([V, I], axis=1)

#     return {
#         'efficiency': eff,
#         'log_lifetime': log_life,
#         'iv_curve': iv_curve,
#     }


# simulation.py
# def simulate_all_phy(params):
#     comp = params["composition"]
#     # if they passed Cs/MA/FA, derive Pb/Sn for the surrogate
#     if "Cs" in comp and "MA" in comp and "FA" in comp:
#         # here we arbitrarily map Cs→Pb fraction for your toy surrogate
#         pb = comp["Cs"]
#         sn = 1 - pb
#     elif "Pb" in comp:
#         pb, sn = comp["Pb"], comp.get("Sn", 1-comp["Pb"])
#     else:
#         raise KeyError("composition dict must contain either ['Cs','MA','FA'] or ['Pb']")

#     # 1) build the 30‐dim features from layers
#     feat30 = np.hstack([
#         [ly.L_nm, ly.D0, np.log10(max(ly.D0,1e-12)), ly.Ea, ly.E, ly.CTE]
#         for ly in params["layers"]
#     ])

#     mask   = np.ones(30, bool); mask[6:8]=False
#     feat28 = feat30[mask]

#     # 2) append process knobs
#     X32 = np.hstack([feat28, [pb,
#                               params.get("uv_filter",0),
#                               params.get("spin_speed",3000),
#                               params.get("anneal_T",100)]])\
#             .reshape(1,-1)

#     eff_pred, life_pred = surrogate.predict(X32)
#     eff, life = float(eff_pred[0]), float(life_pred[0])
#     if params.get("encapsulation",False):
#         eff *= 0.98
#     return {"efficiency":eff, "log_lifetime":life}


# simulation.py (or wherever simulate_all_phy lives)
import numpy as np
from interfaces import Interface
from surrogate_model import SurrogateModel



def simulate_all_phy(params: dict) -> dict:
    layers = params["layers"]

    # 1) build feat28
    feat30 = np.hstack([
        [ly.L_nm,
         ly.D0,
         np.log10(max(ly.D0, 1e-12)),
         ly.Ea,
         ly.E,
         ly.CTE]
        for ly in layers
    ])
    mask   = np.ones(30, bool)
    mask[6:8] = False
    feat28 = feat30[mask]

    # 2) pick Pb fraction and process knobs
    comp = params["composition"]
    pb   = comp.get("Pb", comp.get("Cs", 0.0))   # if Cs/MA/FA you already mapped Cs→Pb
    uv   = params.get("uv_filter", 0.0)
    spin = params.get("spin_speed", 3000.0)
    ann  = params.get("anneal_T", 100.0)

    X32 = np.hstack([feat28, [pb, uv, spin, ann]]).reshape(1, -1)

    # 3) surrogate
    eff_pred, life_pred = surrogate.predict(X32)
    eff, life = float(eff_pred[0]), float(life_pred[0])

    # 4) band‐alignment penalty
    if getattr(layers[1], "band_offset", 0.0) > 0.2:
        eff *= 0.9
    if getattr(layers[3], "band_offset", 0.0) > 0.2:
        eff *= 0.9

    # 5) interface recombination
    total_recomb = sum(
        iface.recomb_velocity * 1e-3 + iface.trap_density * 1e-12
        for iface in params.get("interfaces", [])
    )
    if total_recomb > 0:
        life *= np.exp(-total_recomb)

    # 6) encapsulation
    if params.get("encapsulation", False):
        eff *= 0.98

    return {"efficiency": eff, "log_lifetime": life}



def simulate_all(params):
    layers = params['layers']
    # 1) extract 30 features
    feat30 = np.hstack([
        [ly.L_nm,
         ly.D0,
         np.log10(max(ly.D0, 1e-12)),
         ly.Ea,
         ly.E,
         ly.CTE]
        for ly in layers
    ])
    # 2) drop indices 6,7
    mask = np.ones(30, bool)
    mask[6:8] = False
    feat28 = feat30[mask]
    # 3) process knobs
    Pb    = params['composition']['Pb']
    uv    = params.get('uv_filter', 0.0)
    spin  = params.get('spin_speed', 3000.0)
    ann   = params.get('anneal_T', 100.0)
    X32   = np.hstack([feat28, [Pb, uv, spin, ann]]).reshape(1, -1)
    # 4) surrogate predict
    eff_pred, life_pred = surrogate.predict(X32)
    eff  = float(eff_pred[0])
    life = float(life_pred[0])
    if params.get('encapsulation', False):
        eff *= 0.98
    return {"efficiency": eff, "log_lifetime": life}


# # simulation.py
# import os
# import numpy as np
# import torch
# import deepxde as dde
# from dataclasses import dataclass
# from layers import Layer
# from surrogate_model import SurrogateModel

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # PINN setup
# os.environ["DDE_BACKEND"] = "pytorch"
# os.environ["DDE_DEVICE_ID"] = "0"
# dde.config.default_backend = "pytorch"
# dde.config.default_device_id = 0

# D, H, T_final = 1e-12, 1.0, 1.0
# geom_space = dde.geometry.Rectangle([0, 0], [1, H])
# geom_time = dde.geometry.TimeDomain(0, T_final)
# geom = dde.geometry.GeometryXTime(geom_space, geom_time)

# def pde(x, C):
#     Ct = dde.grad.jacobian(C, x, i=0, j=2)
#     Cxx = dde.grad.hessian(C, x, component=0, i=0, j=0)
#     Cyy = dde.grad.hessian(C, x, component=0, i=1, j=1)
#     return Ct - D * (Cxx + Cyy)

# ic = dde.IC(geom, lambda x: 0.0, lambda _, on_initial: on_initial)
# bc_left = dde.DirichletBC(geom, lambda x: 1.0, lambda x, on_b: on_b and np.isclose(x[0], 0.0))

# data = dde.data.TimePDE(geom, pde, [ic, bc_left], num_domain=1, num_boundary=1, num_initial=1)
# net = dde.maps.FNN([3] + [64] * 4 + [1], "tanh", "Glorot uniform")
# pinn = dde.Model(data, net)
# pinn.compile("L-BFGS")
# pinn.restore("pinn_deepxde_model-25000.pt")

# # Explicitly move pinn to CUDA
# pinn.net.to(device)

# surrogate = SurrogateModel(device=device)

# @dataclass
# class Layer:
#     name: str
#     L_nm: float
#     D0: float
#     Ea: float
#     E: float
#     CTE: float

# def lifetime_L1(stack, alpha=1e12, tau0=1e7):
#     L_cm = np.array([ly.L_nm * 1e-7 for ly in stack])
#     D_arr = np.array([ly.D0 for ly in stack])
#     score = (D_arr * L_cm).mean()
#     return tau0 * np.exp(-alpha * score)

# def prepare_batch_torch(stacks, dx_nm=20):
#     dx = dx_nm * 1e-7
#     B = len(stacks)
#     Nmax = max(sum(max(1, int(round(ly.L_nm / dx_nm))) for ly in stk) for stk in stacks)
#     D_list = []
#     T_K = torch.empty(1).uniform_(298, 358).item()
#     kB = 8.617e-5
#     for stk in stacks:
#         vals = []
#         for ly in stk:
#             n = max(1, int(round(ly.L_nm / dx_nm)))
#             σ = ly.E * 1e9 * (ly.CTE - 12e-6) * (T_K - 298)
#             D_T = ly.D0 * np.exp(-ly.Ea / (kB * T_K))
#             vals += [D_T * (1 + 1e-6 * σ)] * n
#         vals += [vals[-1]] * (Nmax - len(vals))
#         D_list.append(vals)
#     D_batch = torch.tensor(D_list, dtype=torch.float32, device=device)
#     return D_batch, dx

# def lifetime_L2_1d_torch(D_batch, dx, dt=0.1, Ccrit=0.1, max_steps=20000):
#     B, N = D_batch.shape
#     r = D_batch * (dt / dx**2)
#     a0, b0 = 1.0 + r, 1.0 - r
#     off = -0.5 * r[:, 1:]
#     C = torch.zeros_like(D_batch, device=device)
#     C[:, 0] = 1.0
#     out = torch.full((B,), float("inf"), dtype=torch.float32, device=device)

#     for t in range(max_steps):
#         rhs = b0 * C
#         rhs[:, 1:-1] -= off[:, 1:] * C[:, 2:] + off[:, :-1] * C[:, :-2]
#         rhs[:, 0] = 1.0
#         Cnew = C.clone()
#         Cnew[:, 1:-1] = (rhs[:, 1:-1] - off[:, 1:] * C[:, :-2] - off[:, :-1] * C[:, 2:]) / a0[:, 1:-1]
#         C = torch.clamp(Cnew, 0.0, 1.0)

#         done = (C[:, -1] >= Ccrit) & (out == float("inf"))
#         out[done] = (t + 1) * dt
#         if torch.all(out != float("inf")):
#             break

#     return out.cpu().numpy()

# def lifetime_L2_pinn(stack, Ccrit=0.1, samples=100):
#     H_real = sum(ly.L_nm for ly in stack) * 1e-9
#     Ys = np.linspace(0, H_real, samples)
#     X = np.stack([np.zeros_like(Ys), Ys, np.ones_like(Ys) * T_final], axis=1)
#     X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
#     with torch.no_grad():
#         C_tensor = pinn.net(X_tensor)[:, 0]
#         C_bool = (C_tensor >= Ccrit).to(torch.int32)
#         if torch.any(C_bool):
#             idx = torch.where(C_bool)[0][0].item()
#         else:
#             idx = samples - 1
#     return (idx / (samples - 1)) * T_final

# def layers_to_features(layers):
#     vals = []
#     for ly in layers:
#         D0 = ly.D0 if ly.D0 > 0 else 1e-12
#         vals.extend([ly.L_nm, D0, np.log10(D0), ly.Ea, ly.E, ly.CTE, ly.mobility or 0.0, ly.band_offset or 0.0])
#     return np.array(vals, dtype=float)

# def simulate_all(params: dict) -> dict:
#     stack = params["layers"]
#     t1 = lifetime_L1(stack)
#     D_batch, dx = prepare_batch_torch([stack])
#     t2 = lifetime_L2_1d_torch(D_batch, dx)[0]
#     t3 = lifetime_L2_pinn(stack)
#     log_life = np.log10(0.3 * t1 + 0.5 * t2 + 0.2 * t3 + 1e-12)

#     feat = layers_to_features(stack)[None, :]
#     feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
#     # eff_pred, life_pred = surrogate.predict(feat)
#     eff_pred, life_pred = surrogate.predict(feat[:, :32])  # Trim to expected 32 features

#     eff = float(eff_pred[0])

#     if params.get("encapsulation"):
#         eff *= 0.98

#     V_oc, J_sc, FF = eff * 1.2, eff * 30.0, 0.75
#     V = np.linspace(0, V_oc, 50)
#     I = J_sc * (1 - V / V_oc) * FF
#     iv_curve = np.stack([V, I], axis=1)

#     return {"efficiency": eff, "log_lifetime": log_life, "iv_curve": iv_curve}




# # simulation.py
# import os
# import numpy as np
# import torch
# import deepxde as dde
# from dataclasses import dataclass

# # 0) device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1) PINN setup (copy exactly from your data_gen.py)
# os.environ["DDE_BACKEND"]   = "pytorch"
# os.environ["DDE_DEVICE_ID"] = "0"
# dde.config.default_backend   = "pytorch"
# dde.config.default_device_id = 0

# D       = 1e-12
# H       = 1.0
# T_final = 1.0

# # geom_space = dde.geometry.Rectangle([0, 0], [1, H])
# # geom_time  = dde.geometry.TimeDomain(0, T_final)
# # geom       = dde.geometry.GeometryXTime(geom_space, geom_time)

# # def pde(x, C):
# #     Ct  = dde.grad.jacobian(C, x, i=0, j=2)
# #     Cxx = dde.grad.hessian(C, x, component=0, i=0, j=0)
# #     Cyy = dde.grad.hessian(C, x, component=0, i=1, j=1)
# #     return Ct - D*(Cxx + Cyy)

# # ic = dde.IC(geom, lambda x: 0.0, lambda _, on_initial: on_initial)
# # bc_left = dde.DirichletBC(
# #     geom,
# #     lambda x: 1.0,
# #     lambda x, on_b: on_b and np.isclose(x[0], 0.0),
# # )

# # data = dde.data.TimePDE(geom, pde, [ic, bc_left],
# #                         num_domain=1, num_boundary=1, num_initial=1)
# # net  = dde.maps.FNN([3] + [64]*4 + [1], "tanh", "Glorot uniform")
# # pinn = dde.Model(data, net)
# # pinn.compile("L-BFGS")
# # pinn.restore("pinn_deepxde_model-25000.pt")


# # 2) your existing helpers from data_gen.py
# @dataclass
# class Layer:
#     name: str
#     L_nm: float
#     D0:     float
#     Ea:     float
#     E:      float
#     CTE:    float

# def lifetime_L1(stack, alpha=1e12, tau0=1e7):
#     L_cm = np.array([ly.L_nm*1e-7 for ly in stack])
#     D_arr= np.array([ly.D0     for ly in stack])
#     score= (D_arr*L_cm).mean()
#     return tau0*np.exp(-alpha*score)


# def prepare_batch_torch(stacks, dx_nm=20):
#     """Builds D_batch and dx on torch.cuda."""
#     dx = dx_nm * 1e-7
#     B = len(stacks)
#     Nmax = max(
#         sum(max(1, int(round(ly.L_nm / dx_nm))) for ly in stk) for stk in stacks
#     )
#     D_list = []
#     T_K = torch.empty(1).uniform_(298, 358).item()
#     kB = 8.617e-5
#     for stk in stacks:
#         vals = []
#         for ly in stk:
#             n = max(1, int(round(ly.L_nm / dx_nm)))
#             σ = ly.E * 1e9 * (ly.CTE - 12e-6) * (T_K - 298)
#             D_T = ly.D0 * np.exp(-ly.Ea / (kB * T_K))
#             vals += [D_T * (1 + 1e-6 * σ)] * n
#         vals += [vals[-1]] * (Nmax - len(vals))
#         D_list.append(vals)
#     D_batch = torch.tensor(D_list, dtype=torch.float32, device=device)
#     return D_batch, dx



# def lifetime_L2_1d_torch(D_batch, dx, dt=0.1, Ccrit=0.1, max_steps=20000):
#     """Batched Crank–Nicolson on torch tensors (GPU)."""
#     B, N = D_batch.shape
#     r = D_batch * (dt / dx**2)
#     a0 = 1.0 + r
#     b0 = 1.0 - r
#     off = -0.5 * r[:, 1:]
#     C = torch.zeros_like(D_batch, device=device)
#     C[:, 0] = 1.0
#     out = torch.full((B,), float("inf"), dtype=torch.float32, device=device)

#     for t in range(max_steps):
#         # one Jacobi iteration:
#         rhs = b0 * C
#         rhs[:, 1:-1] -= off[:, 1:] * C[:, 2:]
#         rhs[:, 1:-1] -= off[:, :-1] * C[:, :-2]
#         rhs[:, 0] = 1.0
#         Cnew = C.clone()
#         Cnew[:, 1:-1] = (
#             rhs[:, 1:-1]
#             - off[:, 1:] * C[:, :-2]
#             - off[:, :-1] * C[:, 2:]
#         ) / a0[:, 1:-1]
#         C = torch.clamp(Cnew, 0.0, 1.0)

#         done = (C[:, -1] >= Ccrit) & (out == float("inf"))
#         out[done] = (t + 1) * dt
#         if torch.all(out != float("inf")):
#             break

#     return out.cpu().numpy()


# # def lifetime_L2_pinn(stack, Ccrit=0.1, samples=100):
# #     # sample positions along thickness at final time
# #     H_real = sum(ly.L_nm for ly in stack)*1e-9
# #     Ys = np.linspace(0, H_real, samples)
# #     X  = np.stack([np.zeros_like(Ys), Ys, np.ones_like(Ys)*T_final], axis=1)
# #     C  = pinn.predict(X)[:,0]
# #     idx = np.argmax(C >= Ccrit)
# #     return (idx / (samples-1)) * T_final

# # def lifetime_L2_pinn(stack, Ccrit=0.1, samples=100):
# #     H_real = sum(ly.L_nm for ly in stack) * 1e-9
# #     Ys = np.linspace(0, H_real, samples)
# #     X = np.stack([np.zeros_like(Ys), Ys, np.ones_like(Ys) * 1.0], axis=1)
# #     # DeepXDE predict handles numpy and moves to proper device internally
# #     C = pinn.predict(X)[:, 0]
# #     idx = np.argmax(C >= Ccrit)
# #     return (idx / (samples - 1)) * 1.0


# def perovskite_efficiency(composition: dict) -> float:
#     """
#     Toy surrogate: maps composition (e.g. {'Pb':0.8,'Sn':0.2}) → efficiency [0,1]
#     Replace with your trained MLP or published model.
#     """
#     x = np.array([composition['Pb'], composition['Sn']])
#     return 0.9 - 0.1*abs(x[0]-0.5)  # toy: best at Pb:Sn=50:50

# def uv_degradation(filter_fraction: float) -> float:
#     """
#     Returns a lifetime multiplier ∈ (0,1]: fraction of UV blocked slows aging.
#     """
#     return 1.0 / (1.0 + 5*filter_fraction)

# # def simulate_all(params: dict) -> dict:
# #     """
# #     params: {
# #       'layers': [Layer,...],
# #       'composition': {'Pb':..., 'Sn':...},
# #       'uv_filter': float in [0,1],
# #     }
# #     returns: {'log_lifetime':float, 'efficiency':float}
# #     """
# #     # 1) encapsulation lifetime
# #     stack = params['layers']
# #     # sample L1
# #     t1 = lifetime_L1(stack)
# #     # build torch D_batch
# #     D_batch, dx = prepare_batch_torch(stack)
# #     t2 = lifetime_L2_1d_torch(D_batch, dx).mean()
# #     # optional PINN
# #     t3 = lifetime_L2_pinn(stack)
# #     lifetime = np.log10(0.3*t1 + 0.5*t2 + 0.2*t3 + 1e-12)
# #     # 2) perovskite efficiency
# #     eff = perovskite_efficiency(params['composition'])
# #     # 3) UV aging
# #     life_adj = uv_degradation(params['uv_filter'])
# #     return {'log_lifetime': lifetime + np.log10(life_adj),
# #             'efficiency': eff}


# # def simulate_all(params: dict) -> dict:
# #     stack = params["layers"]  # List[Layer]

# #     # 1) analytic L1
# #     t1 = lifetime_L1(stack)

# #     # 2) 1D torch batch (wrap in list!)
# #     D_batch, dx = prepare_batch_torch([stack])
# #     t2_batch = lifetime_L2_1d_torch(D_batch, dx)
# #     t2 = float(t2_batch[0])

# #     # 3) PINN
# #     t3 = lifetime_L2_pinn(stack)

# #     lifetime = np.log10(0.3 * t1 + 0.5 * t2 + 0.2 * t3 + 1e-12)

# #     # perovskite eff + UV
# #     eff = perovskite_efficiency(params["composition"])
# #     life_adj = uv_degradation(params["uv_filter"])
# #     return {
# #         "log_lifetime": lifetime + np.log10(life_adj),
# #         "efficiency": eff,
# #     }



# # # simulation.py
# # import numpy as np
# # from layers import Layer
# # from surrogate_model import SurrogateModel
# # # from lifetime import lifetime_L1, prepare_batch_torch, lifetime_L2_1d_torch, lifetime_L2_pinn

# # # initialize surrogate once
# # surrogate = SurrogateModel(device="cpu")

# # def layers_to_features(layers):
# #     """Convert a stack of Layer objects into a flat feature vector."""
# #     vals = []
# #     for ly in layers:
# #         vals.extend([
# #             ly.L_nm,
# #             ly.D0,
# #             np.log10(ly.D0),
# #             ly.Ea,
# #             ly.E,
# #             ly.CTE,
# #             ly.mobility or 0.0,
# #             ly.band_offset or 0.0,
# #         ])
# #     return np.array(vals, dtype=float)

# # def simulate_all(params: dict) -> dict:
# #     """
# #     Virtual experiment: given params = { 'layers': [...], 'process': {...} },
# #     returns IV curve points, efficiency, and lifetime.
# #     """
# #     layers = params["layers"]
# #     # 1) compute lifetime (as before)
# #     t1 = lifetime_L1(layers)
# #     D_batch, dx = prepare_batch_torch([layers])
# #     t2 = lifetime_L2_1d_torch(D_batch, dx)[0]
# #     t3 = lifetime_L2_pinn(layers)
# #     log_life = np.log10(0.3*t1 + 0.5*t2 + 0.2*t3 + 1e-12)
# #     # 2) compute efficiency via surrogate
# #     feat = layers_to_features(layers)[None, :]
# #     eff, _ = surrogate.predict(feat)
# #     eff = float(eff[0])
# #     # 3) optionally adjust by UV filter or encapsulation
# #     # if params has 'encapsulation', reduce eff slightly
# #     if params.get("encapsulation"):
# #         eff *= 0.98  # assume 2% loss from encapsulant optics
# #     # 4) generate simple IV curve (single diode approximation)
# #     V_oc = eff * 1.2  # placeholder correlation
# #     J_sc = eff * 30.0  # placeholder mA/cm2
# #     FF = 0.75
# #     V = np.linspace(0, V_oc, 50)
# #     I = J_sc * (1 - V / V_oc) * FF
# #     iv_curve = np.stack([V, I], axis=1)

# #     return {
# #         "efficiency": eff,
# #         "log_lifetime": log_life,
# #         "iv_curve": iv_curve,
# #     }



# import os
# import numpy as np
# import torch
# import deepxde as dde
# from layers import Layer
# from surrogate_model import SurrogateModel
# # from lifetime import lifetime_L1, prepare_batch_torch, lifetime_L2_1d_torch

# # Ensure PINN model is available
# # Rebuild PDE and PINN exactly as in data_gen
# D, H, T_final = 1e-12, 1.0, 1.0
# geom = dde.geometry.GeometryXTime(
#     dde.geometry.Rectangle([0, 0], [1, H]),
#     dde.geometry.TimeDomain(0, T_final),
# )

# def pde(x, C):
#     Ct = dde.grad.jacobian(C, x, i=0, j=2)
#     Cxx = dde.grad.hessian(C, x, component=0, i=0, j=0)
#     Cyy = dde.grad.hessian(C, x, component=0, i=1, j=1)
#     return Ct - D*(Cxx + Cyy)

# ic = dde.IC(geom, lambda x: 0.0, lambda _, on_i: on_i)
# bc_l = dde.DirichletBC(
#     geom,
#     lambda x: 1.0,
#     lambda x, on_b: on_b and np.isclose(x[0], 0.0),
# )
# data = dde.data.TimePDE(geom, pde, [ic, bc_l], num_domain=1, num_boundary=1, num_initial=1)
# net = dde.maps.FNN([3] + [64]*4 + [1], "tanh", "Glorot uniform")
# # pinn = dde.Model(data, net)
# # pinn.compile("L-BFGS")

# # PINN Initialization (Existing)
# pinn = dde.Model(data, net)
# pinn.compile("L-BFGS")
# pinn.restore("pinn_deepxde_model-25000.pt")

# # Explicitly move pinn to CUDA device
# pinn.net.to(device)

# # pinn.restore("pinn_deepxde_model-25000.pt")

# # Surrogate wrapper as before
# device = "cuda" if torch.cuda.is_available() else "cpu"
# surrogate = SurrogateModel(device=device)


# # def layers_to_features(layers):
# #     vals = []
# #     for ly in layers:
# #         vals.extend([
# #             ly.L_nm,
# #             ly.D0,
# #             np.log10(ly.D0),
# #             ly.Ea,
# #             ly.E,
# #             ly.CTE,
# #             ly.mobility or 0.0,
# #             ly.band_offset or 0.0,
# #         ])
# #     return np.array(vals, dtype=float)

# # Keep PINN predict on numpy
# def lifetime_L2_pinn(stack, Ccrit=0.1, samples=100):
#     H_real = sum(ly.L_nm for ly in stack) * 1e-9
#     Ys = np.linspace(0, H_real, samples)
#     X = np.stack([np.zeros_like(Ys), Ys, np.ones_like(Ys) * T_final], axis=1)
#     # DeepXDE predict accepts numpy
#     C = pinn.predict(X)[:, 0]
#     idx = np.argmax(C >= Ccrit)
#     return (idx / (samples - 1)) * T_final


# # def simulate_all(params: dict) -> dict:
# #     """
# #     Returns a dict with keys 'efficiency', 'log_lifetime', 'iv_curve'.
# #     """
# #     stack = params["layers"]
# #     t1 = lifetime_L1(stack)
# #     D_batch, dx = prepare_batch_torch([stack])
# #     t2 = lifetime_L2_1d_torch(D_batch, dx)[0]
# #     t3 = lifetime_L2_pinn(stack)
# #     log_life = np.log10(0.3 * t1 + 0.5 * t2 + 0.2 * t3 + 1e-12)
# #     feat = layers_to_features(stack)[None, :]
# #     eff, _ = surrogate.predict(feat)
# #     eff = float(eff[0])
# #     if params.get("encapsulation"):
# #         eff *= 0.98
# #     V_oc = eff * 1.2
# #     J_sc = eff * 30.0
# #     FF = 0.75
# #     V = np.linspace(0, V_oc, 50)
# #     I = J_sc * (1 - V / V_oc) * FF
# #     iv_curve = np.stack([V, I], axis=1)
# #     return {"efficiency": eff, "log_lifetime": log_life, "iv_curve": iv_curve}


# def layers_to_features(layers):
#     """Convert a stack of Layer objects into a flat feature vector (32 dims)."""
#     vals = []
#     for ly in layers:
#         # ensure D0 > 0 for log10
#         D0 = ly.D0 if ly.D0 > 0 else 1e-12
#         vals.extend([
#             ly.L_nm,
#             D0,
#             np.log10(D0),
#             ly.Ea,
#             ly.E,
#             ly.CTE,
#             ly.mobility or 0.0,
#             ly.band_offset or 0.0,
#         ])
#     return np.array(vals, dtype=float)

# # Keep PINN predict on numpy
# def lifetime_L2_pinn(stack, Ccrit=0.1, samples=100):
#     H_real = sum(ly.L_nm for ly in stack) * 1e-9
#     Ys = np.linspace(0, H_real, samples)
#     X = np.stack([np.zeros_like(Ys), Ys, np.ones_like(Ys) * T_final], axis=1)
#     # DeepXDE predict accepts numpy
#     C = pinn.predict(X)[:, 0]
#     idx = np.argmax(C >= Ccrit)
#     return (idx / (samples - 1)) * T_final


# def simulate_all(params: dict) -> dict:
#     """
#     Returns a dict with keys 'efficiency', 'log_lifetime', 'iv_curve'.
#     """
#     stack = params["layers"]
#     # 1) analytic L1
#     t1 = lifetime_L1(stack)
#     # 2) 1D diffusion
#     D_batch, dx = prepare_batch_torch([stack])
#     t2 = lifetime_L2_1d_torch(D_batch, dx)[0]
#     # 3) PINN-based lifetime
#     t3 = lifetime_L2_pinn(stack)
#     # combine lifetimes
#     log_life = np.log10(0.3 * t1 + 0.5 * t2 + 0.2 * t3 + 1e-12)

#     # 4) feature extraction and sanitization
#     feat = layers_to_features(stack)[None, :]
#     # replace NaNs or infinities to finite bounds for scaler
#     feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)

#     # 5) surrogate prediction
#     eff_pred, life_pred = surrogate.predict(feat)
#     eff = float(eff_pred[0])

#     # 6) optical adjustment
#     if params.get("encapsulation"):
#         eff *= 0.98

#     # 7) IV curve generation
#     V_oc = eff * 1.2
#     J_sc = eff * 30.0
#     FF = 0.75
#     V = np.linspace(0, V_oc, 50)
#     I = J_sc * (1 - V / V_oc) * FF
#     iv_curve = np.stack([V, I], axis=1)

#     return {
#         "efficiency": eff,
#         "log_lifetime": log_life,
#         "iv_curve": iv_curve,
#     }
