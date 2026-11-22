# surrogate_model.py
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# MLP Definition (must match training)
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
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

# ─────────────────────────────────────────────────────────────────────────────
# SurrogateModel Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class SurrogateModel:
    def __init__(
        self,
        eff_model_path="hybrid_efficiency_surrogate.pt",
        eff_imputer_path="hybrid_efficiency_imputer.pkl",
        eff_scaler_path="hybrid_efficiency_scaler.pkl",
        life_model_path="hybrid_lifetime_surrogate.pt",
        life_imputer_path="hybrid_lifetime_imputer.pkl",
        life_scaler_path="hybrid_lifetime_scaler.pkl",
        device="cpu",
    ):
        self.device = torch.device(device)

        # Efficiency pipeline
        self.eff_imp = joblib.load(eff_imputer_path)
        self.eff_scaler = joblib.load(eff_scaler_path)
        self.eff_model = MLP(in_dim=32).to(self.device)
        self.eff_model.load_state_dict(
            torch.load(eff_model_path, map_location=self.device)
        )
        self.eff_model.eval()

        # Lifetime pipeline
        self.life_imp = joblib.load(life_imputer_path)
        self.life_scaler = joblib.load(life_scaler_path)
        self.life_model = MLP(in_dim=30).to(self.device)
        self.life_model.load_state_dict(
            torch.load(life_model_path, map_location=self.device)
        )
        self.life_model.eval()

    def predict(self, X32: np.ndarray, X30: np.ndarray):
        """
        X32: (B,32) efficiency features
        X30: (B,30) lifetime features
        returns: eff (B,), life (B,)
        """
        # Efficiency
        Xe = self.eff_imp.transform(X32)
        Xe = self.eff_scaler.transform(Xe)
        Te = torch.tensor(Xe, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            eff = self.eff_model(Te).cpu().numpy()

        # Lifetime
        Xl = self.life_imp.transform(X30)
        Xl = self.life_scaler.transform(Xl)
        Tl = torch.tensor(Xl, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            life = self.life_model(Tl).cpu().numpy()

        return eff, life

    def predict_single(
        self,
        fractions: dict,
        layers: list,
        uv: float = 0.0,
        spin: float = 3500.0,
        anneal: float = 125.0
    ):
        """
        Predict a single composition + stack:
         - build true 30-D lifetime features
         - build 32-D efficiency features
        """
        # 1) Lifetime features (30 dims)
        life_feats = []
        for ly in layers:
            life_feats += [
                ly.L_nm,
                ly.D0,
                np.log10(max(ly.D0, 1e-12)),
                ly.Ea,
                ly.E,
                ly.CTE,
            ]
        X_life = np.array(life_feats, dtype=np.float32).reshape(1, 30)

        # 2) Efficiency features (mask 28 + comp + uv + spin + anneal)
        mask = np.ones(30, dtype=bool)
        mask[6:8] = False
        feat28 = np.array(life_feats)[mask].tolist()
        pb = fractions.get('Cs', 0.0)
        eff_feats = feat28 + [pb, uv, spin, anneal]
        X_eff = np.array(eff_feats, dtype=np.float32).reshape(1, 32)

        # 3) Predict
        eff, life = self.predict(X_eff, X_life)
        return float(np.clip(eff[0], 0.0, 1.0)), float(life[0])
