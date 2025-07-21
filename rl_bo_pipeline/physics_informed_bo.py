# physics_informed_bo.py - Updated for real MIT data structure

import pandas as pd
import numpy as np
from scipy.stats import norm
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import os
import pickle
import numpy as np
import pandas as pd
from scipy.integrate import simps
from sklearn.preprocessing import StandardScaler
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
# physics_informed_bo.py
import os, glob
import numpy as np
import pandas as pd
from scipy.integrate import simps
# physics_informed_bo.py

import os, glob
import numpy as np
import pandas as pd
from scipy.integrate import simps
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

class PhysicsInformedBO:
    def __init__(self, data_root="cluster-perovskite-data/Data", beta_dft=0.025, device="cpu"):
        self.root = data_root
        self.device = torch.device(device)
        self.beta = beta_dft

        # ─── Find the “Calibrated” folder under Images/**/Calibrated ─────────
        img_root = os.path.join(self.root, "Images")
        all_cals = glob.glob(os.path.join(img_root, "**", "Calibrated"), recursive=True)
        cals = [p for p in all_cals if os.path.isdir(p)]
        assert cals, f"no 'Calibrated' dir under {img_root}"
        calib = cals[0]

        # ─── (1) Read samples.csv WITHOUT using 'Sample' as index ──────────────
        samples_csv = os.path.join(calib, "Samples.csv")
        assert os.path.exists(samples_csv), f"{samples_csv} not found"
        df_samples = pd.read_csv(samples_csv)
        # human IDs in df_samples["Sample"], we’ll keep them for logging but not indexing
        sample_ids = df_samples["Sample"].tolist()
        # our 3‐dim BO‐input is the Cs/F A/MA fractions
        self.X = df_samples[["CsPbI", "FAPbI", "MAPbI"]].values

        # ─── (2) Load R/G/B time‐series tables ─────────────────────────────────
        # these CSVs have NO header row, so read every row as data
        R = pd.read_csv(os.path.join(calib, "sample_r_cal.csv"), header=None)
        G = pd.read_csv(os.path.join(calib, "sample_g_cal.csv"), header=None)
        B = pd.read_csv(os.path.join(calib, "sample_b_cal.csv"), header=None)

        # compute instability ∫|c(t)-c(0)| dt for each sample
        instab = []
        for idx in range(len(sample_ids)):
            # integer‐indexed columns now line up 0→first sample, etc
            r = R.iloc[:, idx].values
            g = G.iloc[:, idx].values
            b = B.iloc[:, idx].values
            I_r = simps(np.abs(r - r[0]), dx=1.0)
            I_g = simps(np.abs(g - g[0]), dx=1.0)
            I_b = simps(np.abs(b - b[0]), dx=1.0)
            instab.append(I_r + I_g + I_b)
        instab = np.array(instab)


        # ─── (3) Load XRD compositions & penalize impure spectra ───────────────
        xrd_comp_csv = os.path.join(self.root, "XRD", "xrd_data_compositions.csv")
        assert os.path.exists(xrd_comp_csv), f"{xrd_comp_csv} missing"
        # we only needed these fractions to compare, 
        # df_xrd = pd.read_csv(xrd_comp_csv, index_col=0)

        purity = []
        scaled_dir = os.path.join(self.root, "XRD", "Scaled")
        for idx in range(len(sample_ids)):
            spec_fn = os.path.join(scaled_dir, f"{idx}.csv")
            if os.path.exists(spec_fn):
                spec = pd.read_csv(spec_fn)["intensity"].values
                med = np.median(spec)
                purity.append(0.5 if np.any(spec > 1.2 * med) else 1.0)
            else:
                purity.append(1.0)
        purity = np.array(purity)

        # ─── (4) Build regression targets y = instability × purity ────────
        y = instab * purity
        self.y = y.reshape(-1, 1)

        # placeholder for GP model
        self.gp = None

    def update_gp(self):
        X_t = torch.tensor(self.X, dtype=torch.double, device=self.device)
        y_t = torch.tensor(self.y, dtype=torch.double, device=self.device)
        gp = SingleTaskGP(X_t, y_t)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        gp.train(); mll.train()
        mll(gp.likelihood(gp(X_t)), y_t)
        self.gp = gp.eval()

    def suggest(self, n_candidates=5):
        assert self.gp is not None, "Call update_gp() first"
        best_f = torch.min(torch.tensor(self.y, dtype=torch.double, device=self.device))
        acq = qExpectedImprovement(model=self.gp, best_f=best_f)
        bounds = torch.stack([torch.zeros(3), torch.ones(3)]).to(self.device)
        cands, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=n_candidates,
            num_restarts=5,
            raw_samples=20,
        )
        return cands.detach().cpu().numpy()

