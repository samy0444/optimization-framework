#!/usr/bin/env python3
import os, re
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import joblib

from scipy.integrate import simps
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

# ─────────────────────── CONFIG ───────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH         = 64
PRE_EPOCHS    = 100
FT_EPOCHS     = 50
LR            = 1e-3
WD            = 1e-5
DROPOUT_P     = 0.2
HIDDEN        = 128

# paths
SYNTH_EFF_CSV   = "synthetic_datagen/efficiency_dataset_32d.csv"
REAL_PDB_CSV    = "perovskitedatabase_data/data/Perovskite_database_content_all_data.csv"
SYNTH_LIFE_CSV  = "synthetic_datagen/encap_multifidelity-1.csv"
MIT_IMAGES_DIR  = "cluster-perovskite-data/Data/Images"

# outputs
EFF_MODEL_OUT    = "hybrid_efficiency_surrogate.pt"
EFF_SCALER_OUT   = "hybrid_efficiency_scaler.pkl"
EFF_IMPUTER_OUT  = "hybrid_efficiency_imputer.pkl"
EFF_TEST_X_OUT   = "X_eff_test.npy"
EFF_TEST_Y_OUT   = "y_eff_test.npy"

LIFE_MODEL_OUT   = "hybrid_lifetime_surrogate.pt"
LIFE_SCALER_OUT  = "hybrid_lifetime_scaler.pkl"
LIFE_IMPUTER_OUT = "hybrid_lifetime_imputer.pkl"
LIFE_TEST_X_OUT  = "X_life_test.npy"
LIFE_TEST_Y_OUT  = "y_life_test.npy"

# ──────────────────────── MC‑Dropout MLP ──────────────────────────────────────
class BayesianMLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, p=DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_model(model, tr_loader, val_loader):
    model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
    loss_fn = nn.MSELoss()

    def train_epoch():
        model.train()
        tot, cnt = 0.0, 0
        for xb,yb in tr_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
            cnt += xb.size(0)
        return tot/cnt

    def eval_epoch(loader):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for xb,yb in loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                tot += loss_fn(model(xb), yb).item()*xb.size(0)
                cnt += xb.size(0)
        return tot/cnt

    for _ in trange(PRE_EPOCHS, desc="Pre‑train"): sched.step(train_epoch())
    for _ in trange(FT_EPOCHS, desc="Fine‑tune"): sched.step(train_epoch())

    return eval_epoch(tr_loader), eval_epoch(val_loader), model

# ─────────────────────── HYBRID EFFICIENCY ───────────────────────────────────
print("=== Hybrid Efficiency Surrogate ===")
df_synth = pd.read_csv(SYNTH_EFF_CSV)
feat_eff = [c for c in df_synth.columns if c!="efficiency"]
print(f"Synthetic eff: {len(df_synth)} samples, {len(feat_eff)} dims")

# parse real Perovskite DB
df_pk = pd.read_csv(REAL_PDB_CSV, low_memory=False)
def split_items(s): return re.split(r"[;|,]", s) if isinstance(s,str) else []
df_pk["ions"]      = df_pk["Perovskite_composition_a_ions"].map(split_items)
df_pk["coefs_raw"] = df_pk["Perovskite_composition_a_ions_coefficients"].map(split_items)

def parse_coefs(raw):
    out=[]
    for e in raw:
        m = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", e.strip())
        if m: out.append(float(m.group(0)))
    return out
df_pk["coefs"] = df_pk["coefs_raw"].map(parse_coefs)

# build Cs/MA/FA
def make_frac_dict(ions, coefs):
    return {ion.strip():coef/100.0 for ion,coef in zip(ions,coefs)}
frac = [make_frac_dict(df_pk["ions"].iat[i], df_pk["coefs"].iat[i]) for i in range(len(df_pk))]
df_pk["Cs"] = [d.get("Cs",0.0) for d in frac]
df_pk["MA"] = [d.get("MA",0.0) for d in frac]
df_pk["FA"] = 1.0 - np.array(df_pk["Cs"]) - np.array(df_pk["MA"])

df_pk["spin_speed"] = pd.to_numeric(df_pk.get("Perovskite_deposition_number_of_deposition_steps"),errors="coerce").fillna(3500.0)
df_pk["anneal_T"]    = pd.to_numeric(df_pk.get("Perovskite_deposition_thermal_annealing_temperature"),errors="coerce").fillna(125.0)

df_real = df_pk[df_pk["JV_reverse_scan_PCE"].notna()][["Cs","MA","FA","spin_speed","anneal_T","JV_reverse_scan_PCE"]]
df_real = df_real.rename(columns={"JV_reverse_scan_PCE":"efficiency"})
print(f"Real eff: {len(df_real)} samples")

# harmonize & concat
for c in feat_eff:
    if c not in df_real.columns: df_real[c]=0.0
df_real = df_real[feat_eff+["efficiency"]]
df_eff  = pd.concat([df_synth, df_real], axis=0).sample(frac=1, random_state=42)

X_eff  = df_eff[feat_eff].values.astype(np.float32)
y_eff  = df_eff["efficiency"].values.astype(np.float32)

# impute & outlier
imp_e = SimpleImputer(strategy="median"); X_eff = imp_e.fit_transform(X_eff)
joblib.dump(imp_e, EFF_IMPUTER_OUT)
mask  = IsolationForest(contamination=0.02, random_state=42).fit_predict(X_eff)==1
X_eff,y_eff = X_eff[mask], y_eff[mask]
print(f" → cleaned eff: {len(y_eff)} samples")

# split & scale
Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(X_eff,y_eff,test_size=0.2,random_state=42)
sc_e = StandardScaler().fit(Xe_tr); joblib.dump(sc_e, EFF_SCALER_OUT)
Xe_tr_s, Xe_te_s = sc_e.transform(Xe_tr), sc_e.transform(Xe_te)
np.save(EFF_TEST_X_OUT, Xe_te); np.save(EFF_TEST_Y_OUT, ye_te)

tr_le = DataLoader(TensorDataset(torch.tensor(Xe_tr_s), torch.tensor(ye_tr)), batch_size=BATCH, shuffle=True)
te_le = DataLoader(TensorDataset(torch.tensor(Xe_te_s), torch.tensor(ye_te)), batch_size=BATCH)

# train + eval
model_e = BayesianMLP(in_dim=Xe_tr_s.shape[1]).to(DEVICE)
train_mse_e, test_mse_e, model_e = train_model(model_e, tr_le, te_le)
print(f"➡️ Eff train MSE: {train_mse_e:.4f}, test MSE: {test_mse_e:.4f}")
with torch.no_grad():
    preds = np.concatenate([model_e(xb.to(DEVICE)).cpu().numpy() for xb,_ in te_le])
r2_e = 1 - ((ye_te-preds)**2).sum()/((ye_te-ye_te.mean())**2).sum()
print(f"➡️ Eff test R²: {r2_e:.4f}")
torch.save(model_e.state_dict(), EFF_MODEL_OUT)
print(f"✅ Saved {EFF_MODEL_OUT}\n")

# ─────────────────────── HYBRID LIFETIME ─────────────────────────────────────
print("=== Hybrid Lifetime Surrogate ===")
df_ls = pd.read_csv(SYNTH_LIFE_CSV)
life_feat = [c for c in df_ls.columns if c!="target"]
df_ls      = df_ls.rename(columns={"target":"log_life"})
print(f"Synthetic life: {len(df_ls)} samples, {len(life_feat)} dims")

# MIT real-lifetime
def load_mit(images_root):
    for r,ds,_ in os.walk(images_root):
        if os.path.basename(r)=="Calibrated":
            calib = r; break
    df = pd.read_csv(os.path.join(calib,"Samples.csv"))
    Xf = df[["CsPbI","MAPbI","FAPbI"]].values.astype(np.float32)
    R = pd.read_csv(os.path.join(calib,"sample_r_cal.csv"),header=None)
    G = pd.read_csv(os.path.join(calib,"sample_g_cal.csv"),header=None)
    B = pd.read_csv(os.path.join(calib,"sample_b_cal.csv"),header=None)
    inst = [
        simps(np.abs(R.iloc[:,i]-R.iloc[0,i])) +
        simps(np.abs(G.iloc[:,i]-G.iloc[0,i])) +
        simps(np.abs(B.iloc[:,i]-B.iloc[0,i]))
        for i in range(len(df))
    ]
    xrd_dir = os.path.join(images_root.replace("Images",""),"XRD","Scaled")
    purity=[]
    for i in range(len(df)):
        fn = os.path.join(xrd_dir,f"{i}.csv")
        if os.path.exists(fn):
            spec = pd.read_csv(fn)["intensity"].values
            purity.append(0.5 if np.any(spec>1.2*np.median(spec)) else 1.0)
        else:
            purity.append(1.0)
    y = (np.array(inst)*np.array(purity)).astype(np.float32)
    spin = np.full((len(df),1),3500.,dtype=np.float32)
    ann  = np.full((len(df),1),125., dtype=np.float32)
    X = np.hstack([Xf, spin, ann])
    return X, y

# Xr_l, yr_l = load_mit(MIT_IMAGES_DIR)
# df_rl     = pd.DataFrame(Xr_l, columns=life_feat); df_rl["log_life"]=yr_l
# print(f"Real life: {len(df_rl)} samples")

# MIT real-lifetime
Xr_l, yr_l = load_mit(MIT_IMAGES_DIR)   # shape (28,5)
# create a zero‐padded frame with all 30 synthetic features
df_rl = pd.DataFrame(
    0.0,
    index=np.arange(len(Xr_l)),
    columns=life_feat
)
# you could fill in some of those 30 if you want, but at minimum pad zeros:
# df_rl.loc[:, some_stack_feature_cols] = ... (if you had any mapping)
df_rl["log_life"] = yr_l
print(f"Real life: {len(df_rl)} samples")
# harmonize, combine
for c in life_feat:
    if c not in df_rl.columns: df_rl[c]=0.0
df_rl = df_rl[life_feat+["log_life"]]
df_lh = pd.concat([df_ls, df_rl], axis=0).sample(frac=1, random_state=42)

X_lh = df_lh[life_feat].values.astype(np.float32)
y_lh = df_lh["log_life"].values.astype(np.float32)

# impute & outlier
imp_l = SimpleImputer(strategy="median"); X_lh = imp_l.fit_transform(X_lh)
joblib.dump(imp_l, LIFE_IMPUTER_OUT)
maskl = IsolationForest(contamination=0.02, random_state=42).fit_predict(X_lh)==1
X_lh, y_lh = X_lh[maskl], y_lh[maskl]
print(f" → cleaned life: {len(y_lh)} samples")

# split & scale
Xlt, Xle, ylt, yle = train_test_split(X_lh,y_lh,test_size=0.2,random_state=42)
sc_lf = StandardScaler().fit(Xlt); joblib.dump(sc_lf, LIFE_SCALER_OUT)
Xlt_s, Xle_s = sc_lf.transform(Xlt), sc_lf.transform(Xle)
np.save(LIFE_TEST_X_OUT, Xle); np.save(LIFE_TEST_Y_OUT, yle)

tr_ll = DataLoader(TensorDataset(torch.tensor(Xlt_s), torch.tensor(ylt)), batch_size=BATCH, shuffle=True)
te_ll = DataLoader(TensorDataset(torch.tensor(Xle_s), torch.tensor(yle)), batch_size=BATCH)

# train & eval
model_l = BayesianMLP(in_dim=Xlt_s.shape[1]).to(DEVICE)
train_m_l, test_m_l, model_l = train_model(model_l, tr_ll, te_ll)
print(f"➡️ Life train MSE: {train_m_l:.4f}, test MSE: {test_m_l:.4f}")
with torch.no_grad():
    preds_l = np.concatenate([model_l(xb.to(DEVICE)).cpu().numpy() for xb,_ in te_ll])
r2_l = 1 - ((yle-preds_l)**2).sum()/((yle-yle.mean())**2).sum()
print(f"➡️ Life test R²: {r2_l:.4f}")
torch.save(model_l.state_dict(), LIFE_MODEL_OUT)
print(f"✅ Saved {LIFE_MODEL_OUT}")
