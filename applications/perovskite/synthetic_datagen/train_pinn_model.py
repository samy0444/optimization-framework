

#!/usr/bin/env python3
import os
# force DeepXDE to use PyTorch + GPU 0
os.environ["DDE_BACKEND"]    = "pytorch"
os.environ["DDE_DEVICE_ID"]  = "0"

import torch, deepxde as dde
import numpy as np

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# we can also set these directly on the config
dde.config.default_backend   = "pytorch"
dde.config.default_device_id = 0

# 1) Problem parameters
D       = 1e-12     # diffusivity
H       = 1.0       # normalized thickness
T_final = 1.0       # normalized final time

# 2) Geometry
geom_space = dde.geometry.Rectangle([0, 0], [1, H])
geom_time  = dde.geometry.TimeDomain(0, T_final)
geom       = dde.geometry.GeometryXTime(geom_space, geom_time)

# 3) PDE residual
def pde(x, C):
    C_t  = dde.grad.jacobian(C, x, i=0, j=2)
    C_xx = dde.grad.hessian(C, x, component=0, i=0, j=0)
    C_yy = dde.grad.hessian(C, x, component=0, i=1, j=1)
    return C_t - D*(C_xx + C_yy)

# 4) Initial condition: C(x,y,0)=0
def init_cond(x):
    return np.zeros((len(x), 1))
ic = dde.IC(geom, init_cond, lambda x, on_initial: on_initial)

# 5) Dirichlet BC at x=0: C=1
def bc_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)
bc_l = dde.DirichletBC(geom, lambda x: 1.0, bc_left)

# 6) Data object
data = dde.data.TimePDE(
    geom,
    pde,
    [ic, bc_l],
    num_domain=40000,
    num_boundary=2000,
    num_initial=2000,
)

# 7) Network
layers = [3] + [64]*4 + [1]   # (x,y,t) → 4×64 → C
net    = dde.maps.FNN(layers, "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# verify model parameters landed on cuda:0
for name, param in model.net.named_parameters():
    print(f"{name}: {param.device}")
    break

# 8) Train
losshistory, train_state = model.train(iterations=10000)

# 9) Switch to L-BFGS for fine tuning
model.compile("L-BFGS")
_ = model.train()

# 10) Save
model.save("pinn_deepxde_model")
#!/usr/bin/env python3
# import os
# from datetime import datetime
# import numpy as np
# import torch
# import deepxde as dde
# from deepxde.nn.pytorch.nn import NN
# from torch_geometric.nn import CGConv, global_mean_pool
# from torch_geometric.data import Data, Batch
# from datetime import datetime

# # Create timestamp for checkpoint filenames
# timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# # 1) Configure backend
# os.environ["DDE_BACKEND"]="pytorch"; os.environ["DDE_DEVICE_ID"]="0"
# dde.config.default_backend="pytorch"; dde.config.default_device_id=0

# # 2) Physical constants & normalization
# T_K, D0, Ea, R_gas = 333.0, 1e-6, 4e4, 8.314
# D = D0*np.exp(-Ea/(R_gas*T_K)); L_real, t_real = 1e-2, 3600*1000
# D_norm = D * t_real/(L_real**2)

# # 3) Geometry & PDE
# geom = dde.geometry.GeometryXTime(
#     dde.geometry.Rectangle([0,0],[1,1]),
#     dde.geometry.TimeDomain(0,1)
# )
# def pde(x,C):
#     Ct = dde.grad.jacobian(C,x,i=0,j=2)
#     Cxx= dde.grad.hessian(C,x,component=0,i=0,j=0)
#     Cyy= dde.grad.hessian(C,x,component=0,i=1,j=1)
#     return Ct - D_norm*(Cxx + Cyy)

# # 4) BCs & IC
# ic  = dde.IC(geom, lambda x: 0.0, lambda _,on_i: on_i)
# bc1 = dde.DirichletBC(geom, lambda x:1.0, lambda x,on_b: on_b and np.isclose(x[0],0))
# bc2 = dde.NeumannBC(geom, lambda x:0.0, lambda x,on_b: on_b and np.isclose(x[0],1))
# bc3 = dde.NeumannBC(geom, lambda x:0.0, lambda x,on_b: on_b and np.isclose(x[1],0))
# bc4 = dde.NeumannBC(geom, lambda x:0.0, lambda x,on_b: on_b and np.isclose(x[1],1))

# # 5) Data object
# data = dde.data.TimePDE(
#     geom, pde, [ic,bc1,bc2,bc3,bc4],
#     num_domain=50000, num_boundary=5000, num_initial=5000
# )

# # 6) PI-GNN model
# class PI_CGCNN(NN):
#     def __init__(self, node_in=5, edge_in=1, hidden=64):
#         super().__init__()
#         self.node_embed = torch.nn.Linear(node_in, hidden)
#         self.convs = torch.nn.ModuleList([CGConv(hidden, edge_in) for _ in range(4)])
#         self.eff_head = torch.nn.Sequential(
#             torch.nn.Linear(hidden,32), torch.nn.ReLU(), torch.nn.Linear(32,1)
#         )
#         self.life_head= torch.nn.Sequential(
#             torch.nn.Linear(hidden,32), torch.nn.ReLU(), torch.nn.Linear(32,1)
#         )
#     def forward(self, data):
#         x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         x = torch.relu(self.node_embed(x))
#         for conv in self.convs:
#             x = x + torch.relu(conv(x, ei, ea))
#         x = global_mean_pool(x, batch)
#         return self.eff_head(x), self.life_head(x)

# net   = PI_CGCNN()
# model = dde.Model(data, net)

# # 7) Train
# model.compile(
#     "adam",
#     lr=1e-3,
#     decay=("step", 5000, 0.9),
#     loss_weights=[1,10,10,10,10]
# )
# callbacks = [
#     dde.callbacks.ModelCheckpoint(f"gcnn_ckpt_{timestamp}.pt", save_better_only=True),
#     dde.callbacks.EarlyStopping(patience=2000)
# ]
# model.train(iterations=20000, callbacks=callbacks, display_every=100)
# model.compile("L-BFGS", loss_weights=[1,10,10,10,10])
# model.train(display_every=100)
# model.save(f"pinn_gcnn_{timestamp}")

