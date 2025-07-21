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
    pb   = comp.get("Pb", comp.get("Cs", 0.0))   # if Cs/MA/FA, already mapped Cs→Pb
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
