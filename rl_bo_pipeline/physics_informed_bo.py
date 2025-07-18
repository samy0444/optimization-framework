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
        # we only needed these fractions to compare, but you can drop reading them
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

        # ─── (4) Build your regression targets y = instability × purity ────────
        y = instab * purity
        self.y = y.reshape(-1, 1)

        # placeholder for your GP model
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


# … then your run_bo() can call update_gp(), suggest(), append_data() as before …

# class PhysicsInformedBO:
#     """
#     Physics-informed BO using actual MIT cluster-perovskite-data structure
#     """
#     def __init__(self, 
#                  xrd_compositions_csv="cluster-perovskite-data/Data/XRD/xrd_data_compositions.csv",
#                  beta_dft=0.025,
#                  device="cpu"):
        
#         # Load actual composition data from XRD folder
#         self.compositions_df = pd.read_csv(xrd_compositions_csv)
        
#         # For now, create synthetic instability indices since the MIT data
#         # focuses on clustering rather than quantitative degradation metrics
#         # In practice, you'd compute these from the RGB time-series data
#         self.compositions_df['instability_index'] = self._compute_synthetic_instability()
        
#         # Extract composition arrays
#         if all(col in self.compositions_df.columns for col in ['Cs', 'MA', 'FA']):
#             self.X = self.compositions_df[['Cs', 'MA', 'FA']].values
#         else:
#             # Parse composition from other columns if format is different
#             self.X = self._parse_compositions()
            
#         self.y = self.compositions_df['instability_index'].values.reshape(-1, 1)
        
#         # Synthetic DFT data (you'd replace with actual DFT calculations)
#         self.dft_data = self._create_synthetic_dft_data()
        
#         self.beta = beta_dft
#         self.device = torch.device(device)
#         self.gp = None

#     def _compute_synthetic_instability(self):
#         """
#         Compute synthetic instability indices.
#         In practice, extract from RGB time-series data in Images/ folder
#         """
#         n_samples = len(self.compositions_df)
#         # Create composition-dependent instability
#         return np.random.exponential(scale=2.0, size=n_samples)

#     def _parse_compositions(self):
#         """
#         Parse compositions from MIT data format
#         """
#         # This depends on actual column names in xrd_data_compositions.csv
#         # You'll need to adapt based on the real file structure
#         compositions = []
#         for _, row in self.compositions_df.iterrows():
#             # Example parsing - adapt to actual data format
#             cs = float(row.get('Cs_fraction', 0))
#             ma = float(row.get('MA_fraction', 0)) 
#             fa = float(row.get('FA_fraction', 1))
#             compositions.append([cs*100, ma*100, fa*100])
#         return np.array(compositions)

#     def _create_synthetic_dft_data(self):
#         """
#         Create synthetic DFT mixing energies.
#         Replace with actual DFT calculations
#         """
#         dft_data = {}
#         for x in self.X:
#             cs, ma, fa = x
#             # Synthetic mixing energy based on known perovskite thermodynamics
#             delta_g = -0.05 + 0.001*cs + 0.002*ma - 0.001*fa
#             dft_data[tuple(x)] = delta_g
#         return dft_data

#     def _prob_feasible(self, comp):
#         """Physics constraint from DFT mixing energy"""
#         ΔG = self.dft_data.get(tuple(comp), 0.0)
#         return 1/(1 + np.exp(-ΔG/self.beta))

#     def update_gp(self):
#         """Update GP model with current data"""
#         X_t = torch.tensor(self.X, dtype=torch.double, device=self.device)
#         y_t = torch.tensor(self.y, dtype=torch.double, device=self.device)
#         gp = SingleTaskGP(X_t, y_t)
#         mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#         gp.train(); mll.train()
#         mll(gp.likelihood(gp(X_t)), y_t)
#         self.gp = gp.eval()

#     # def suggest(self, batch_size=10):
#     #     """Suggest new compositions using physics-informed acquisition"""
#     #     # Generate composition grid
#     #     grid = []
#     #     for cs in np.linspace(0, 50, 20):  # Realistic Cs range
#     #         for ma in np.linspace(0, 50, 20):  # Realistic MA range
#     #             fa = 100 - cs - ma
#     #             if fa >= 50:  # FA-dominant compositions
#     #                 grid.append([cs, ma, fa])
        
#     #     grid = np.array(grid)
        
#     #     # GP predictions
#     #     Xg = torch.tensor(grid, dtype=torch.double, device=self.device)
#     #     # μ, σ2 = self.gp(Xg)
#     #     out = self.gp(Xg)
#     #     μ = out.mean.detach().cpu().numpy()
#     #     σ2 = out.variance.detach().cpu().numpy()

#     #     μ = μ.detach().cpu().numpy().ravel()
#     #     σ = np.sqrt(σ2.detach().cpu().numpy().ravel())
        
#     #     # Expected improvement
#     #     f_best = np.min(self.y)
#     #     eis = self._expected_improvement(μ, σ, f_best)
        
#     #     # Physics weighting
#     #     weights = np.array([self._prob_feasible(c) for c in grid])
#     #     scores = eis * weights
        
#     #     # Select top candidates
#     #     idx = np.argsort(scores)[-batch_size:]
#     #     return grid[idx]
#     def suggest(self, batch_size=10):
#         # Generate composition grid
#         grid = []
#         for cs in np.linspace(0, 50, 20):
#             for ma in np.linspace(0, 50, 20):
#                 fa = 100 - cs - ma
#                 if fa >= 50:
#                     grid.append([cs, ma, fa])
#         grid = np.array(grid)

#         # GP predictions
#         Xg = torch.tensor(grid, dtype=torch.double, device=self.device)
#         out = self.gp(Xg)
#         μ = out.mean.detach().cpu().numpy().ravel()
#         σ2 = out.variance.detach().cpu().numpy().ravel()
#         σ = np.sqrt(σ2)

#         # Expected improvement
#         f_best = np.min(self.y)
#         eis = self._expected_improvement(μ, σ, f_best)

#         # Physics weighting
#         weights = np.array([self._prob_feasible(c) for c in grid])
#         scores = eis * weights

#         # Select top candidates
#         idx = np.argsort(scores)[-batch_size:]
#         return grid[idx]

#     def _expected_improvement(self, μ, σ, f_best):
#         """Standard expected improvement"""
#         imp = f_best - μ
#         Z = imp/σ
#         return imp * norm.cdf(Z) + σ * norm.pdf(Z)

#     def run_round(self, batch_size=10):
#         """Run one round of BO"""
#         self.update_gp()
#         new_comps = self.suggest(batch_size)
        
#         # In practice: run experiments and get true instability values
#         # For now: use synthetic values
#         inst = np.random.exponential(scale=1.0, size=len(new_comps))
        
#         # Update data
#         self.X = np.vstack([self.X, new_comps])
#         self.y = np.vstack([self.y, inst.reshape(-1, 1)])
        
#         return new_comps, inst


# # physics_informed_bo.py

# import numpy as np
# import pandas as pd
# from scipy.stats import norm

# class PhysicsInformedBO:
#     def __init__(self, composition_space, dft_data, beta_dft=0.025):
#         """
#         Parameters:
#           - composition_space: dict, e.g. {'Cs':(0,100),'MA':(0,100),'FA':(0,100)}
#           - dft_data: dict mapping (Cs,MA,FA) → ΔG_mix (eV)
#           - beta_dft: float, fusion parameter
#         """
#         self.composition_space = composition_space
#         self.dft_data = dft_data
#         self.beta_dft = beta_dft
#         self.experimental_data = {'compositions': [], 'instability_indices': []}
#         self.gp_model = None

#     def _prob_constraint(self, comp):
#         # Logistic CDF weighting based on ΔG_mix
#         dg = self.dft_data.get(tuple(comp), 0.0)
#         return 1 / (1 + np.exp(-dg / self.beta_dft))

#     def _expected_improvement(self, mu, sigma, f_best):
#         """Standard EI"""
#         imp = f_best - mu
#         Z = imp / sigma
#         return imp * norm.cdf(Z) + sigma * norm.pdf(Z)

#     def _acquisition(self, comp):
#         # Surrogate predictions
#         mu, sigma = self.gp_model.predict(np.array(comp).reshape(1,-1))
#         ei = self._expected_improvement(mu, sigma, min(self.experimental_data['instability_indices']))
#         return ei * self._prob_constraint(comp)

#     def update_model(self):
#         from botorch.models import SingleTaskGP
#         from gpytorch.mlls import ExactMarginalLogLikelihood
#         import torch

#         X = np.vstack(self.experimental_data['compositions'])
#         y = np.array(self.experimental_data['instability_indices']).reshape(-1,1)
#         # Standardize and fit GP
#         X_t = torch.tensor(X, dtype=torch.double)
#         y_t = torch.tensor(y, dtype=torch.double)
#         gp = SingleTaskGP(X_t, y_t)
#         mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#         gp.train()
#         mll.train()
#         mll_output = mll(gp.likelihood(gp(X_t)), y_t)  # one‐step train
#         self.gp_model = gp.eval()

#     def suggest_experiments(self, n_suggestions=10):
#         # Random grid sampling for candidates
#         grid = []
#         for Cs in np.linspace(*self.composition_space['Cs'], 50):
#           for MA in np.linspace(*self.composition_space['MA'], 50):
#             FA = 100 - Cs - MA
#             if FA >= 0:
#                 grid.append([Cs, MA, FA])
#         # Evaluate acquisition
#         acq_vals = [self._acquisition(c) for c in grid]
#         idx = np.argsort(acq_vals)[-n_suggestions:]
#         return np.array(grid)[idx]

#     def run_optimization(self, max_rounds=5, batch_size=10):
#         # 1) Seed with existing experimental data
#         if len(self.experimental_data['compositions']) < batch_size:
#             # random initialization
#             init = np.random.rand(batch_size,3)
#             for c in init:
#                 self.experimental_data['compositions'].append(c * [100,100,100])
#                 self.experimental_data['instability_indices'].append(np.inf)
#         # 2) BO loop
#         for _ in range(max_rounds):
#             self.update_model()
#             candidates = self.suggest_experiments(batch_size)
#             # Here, user must run real experiments or surrogate to get instability
#             # For integration, call simulate_all or experimental loader
#             instab = [self._run_experiment(c) for c in candidates]
#             self.experimental_data['compositions'].extend(candidates)
#             self.experimental_data['instability_indices'].extend(instab)
#         return {
#             'best_composition': self.experimental_data['compositions'][np.argmin(self.experimental_data['instability_indices'])],
#             'best_instability': min(self.experimental_data['instability_indices'])
#         }

#     def _run_experiment(self, comp):
#         # Placeholder: integrate with your dataset results.csv
#         # e.g., lookup matching composition in results.csv
#         df = pd.read_csv('Data/Metadata/results.csv')
#         row = df[(df.Cs==comp[0]) & (df.MA==comp[1]) & (df.FA==comp[2])]
#         return float(row.instability_index) if not row.empty else np.inf
