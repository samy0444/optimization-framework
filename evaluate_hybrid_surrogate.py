#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# ─── use the exact same architecture you trained with ─────────────────────────
class BayesianMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── load held-out test sets ─────────────────────────────────────────────────
X_life_test = np.load("heldout_testdata/X_life_test.npy")   # (N_life, D_life)
y_life_test = np.load("heldout_testdata/y_life_test.npy")   # (N_life,)
X_eff_test  = np.load("heldout_testdata/X_eff_test.npy")    # (N_eff, D_eff)
y_eff_test  = np.load("heldout_testdata/y_eff_test.npy")    # (N_eff,)

print(f"Lifetime test set: X {X_life_test.shape}, y {y_life_test.shape}")
print(f"Efficiency test set: X {X_eff_test.shape}, y {y_eff_test.shape}")

# ─── load scalers & hybrid weights ────────────────────────────────────────────
life_scaler = joblib.load("hybrid_lifetime_scaler.pkl")
eff_scaler  = joblib.load("hybrid_efficiency_scaler.pkl")

life_model = BayesianMLP(in_dim=X_life_test.shape[1]).eval()
eff_model  = BayesianMLP(in_dim=X_eff_test.shape[1]).eval()

life_model.load_state_dict(torch.load("hybrid_lifetime_surrogate.pt", map_location="cpu"))
eff_model .load_state_dict(torch.load("hybrid_efficiency_surrogate.pt", map_location="cpu"))

# ─── scale, predict ─────────────────────────────────────────────────────────
with torch.no_grad():
    # lifetime predictions
    Xl_scaled = life_scaler.transform(X_life_test)
    yl_pred   = life_model(torch.tensor(Xl_scaled, dtype=torch.float32)).cpu().numpy()
    # efficiency predictions
    Xe_scaled = eff_scaler.transform(X_eff_test)
    ye_pred   = eff_model(torch.tensor(Xe_scaled,  dtype=torch.float32)).cpu().numpy()

# ─── compute & print metrics ─────────────────────────────────────────────────
print("\n=== Lifetime Surrogate Performance ===")
print(" RMSE:", mean_squared_error(y_life_test, yl_pred, squared=False))
print("   R²:", r2_score(              y_life_test, yl_pred))

print("\n=== Efficiency Surrogate Performance ===")
print(" RMSE:", mean_squared_error(y_eff_test, ye_pred, squared=False))
print("   R²:", r2_score(              y_eff_test, ye_pred))
