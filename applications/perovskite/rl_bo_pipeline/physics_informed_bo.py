# physics_informed_bo.py
"""Composition-level Bayesian optimization for the perovskite loop.

The GP starts from the measured MIT PV-Lab degradation data. For each sample,
an instability score is the integrated colour change of its calibrated RGB
images over time, doubled when its XRD spectrum shows impurity peaks. Lower
instability is better, so the GP models stability = -instability and maximizes
it with q-Expected Improvement.

Once the loop has made at least `min_loop_obs` evaluations of its own (passed
in with `add_observations`), the GP is refit on those instead, so one model
never mixes two different quantities.

Compositions are rows of (Cs, FA, MA) fractions; suggestions are constrained
to sum to 1, as the MIT samples do.
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simps
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class PhysicsInformedBO:
    def __init__(self, data_root="cluster-perovskite-data/Data", device="cpu", min_loop_obs=3):
        self.root = data_root
        self.device = torch.device(device)
        self.min_loop_obs = min_loop_obs

        # ─── Find the "Calibrated" folder under Images/**/Calibrated ─────────
        img_root = os.path.join(self.root, "Images")
        all_cals = glob.glob(os.path.join(img_root, "**", "Calibrated"), recursive=True)
        cals = [p for p in all_cals if os.path.isdir(p)]
        assert cals, f"no 'Calibrated' dir under {img_root}"
        calib = cals[0]

        # ─── (1) Sample compositions ─────────────────────────────────────────
        samples_csv = os.path.join(calib, "Samples.csv")
        assert os.path.exists(samples_csv), f"{samples_csv} not found"
        df_samples = pd.read_csv(samples_csv)
        sample_ids = df_samples["Sample"].tolist()
        # 3-D BO input, column order Cs, FA, MA
        self.X = df_samples[["CsPbI", "FAPbI", "MAPbI"]].values

        # ─── (2) Instability = integrated |colour(t) - colour(0)| ───────────
        # these CSVs have no header row; column i is sample i
        R = pd.read_csv(os.path.join(calib, "sample_r_cal.csv"), header=None)
        G = pd.read_csv(os.path.join(calib, "sample_g_cal.csv"), header=None)
        B = pd.read_csv(os.path.join(calib, "sample_b_cal.csv"), header=None)
        instab = []
        for idx in range(len(sample_ids)):
            r = R.iloc[:, idx].values
            g = G.iloc[:, idx].values
            b = B.iloc[:, idx].values
            instab.append(
                simps(np.abs(r - r[0]), dx=1.0)
                + simps(np.abs(g - g[0]), dx=1.0)
                + simps(np.abs(b - b[0]), dx=1.0)
            )
        instab = np.array(instab)

        # ─── (3) XRD phase purity: 0.5 if impurity peaks, else 1.0 ───────────
        xrd_comp_csv = os.path.join(self.root, "XRD", "xrd_data_compositions.csv")
        assert os.path.exists(xrd_comp_csv), f"{xrd_comp_csv} missing"
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

        # ─── (4) Penalised instability (lower is better) ─────────────────────
        # dividing by purity doubles the instability of impure samples
        self.y = (instab / purity).reshape(-1, 1)

        # the loop's own evaluations (higher is better), filled by add_observations
        self.X_obs = np.empty((0, 3))
        self.y_obs = np.empty((0, 1))

        self.gp = None
        self._train_y = None

    def add_observations(self, X_new, y_new):
        """Record loop evaluations: X_new rows are (Cs, FA, MA), higher y is better."""
        self.X_obs = np.vstack([self.X_obs, np.asarray(X_new, dtype=float).reshape(-1, 3)])
        self.y_obs = np.vstack([self.y_obs, np.asarray(y_new, dtype=float).reshape(-1, 1)])

    def _training_data(self):
        if len(self.y_obs) >= self.min_loop_obs:
            return self.X_obs, self.y_obs
        return self.X, -self.y  # stability: higher is better

    def update_gp(self):
        X, y = self._training_data()
        X_t = torch.tensor(X, dtype=torch.double, device=self.device)
        y_t = torch.tensor(y, dtype=torch.double, device=self.device)
        gp = SingleTaskGP(X_t, y_t, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.gp = gp.eval()
        self._train_y = y_t

    def suggest(self, n_candidates=5):
        """Propose compositions as rows of (Cs, FA, MA) fractions summing to 1."""
        assert self.gp is not None, "Call update_gp() first"
        acq = qExpectedImprovement(model=self.gp, best_f=self._train_y.max())
        dt = torch.double
        num_restarts = 5
        bounds = torch.stack([torch.zeros(3, dtype=dt), torch.ones(3, dtype=dt)]).to(self.device)
        simplex = [(torch.arange(3, device=self.device), torch.ones(3, dtype=dt, device=self.device), 1.0)]
        # Random starting points drawn uniformly from the simplex, in double
        # precision. Generating them here (instead of BoTorch's polytope sampler)
        # keeps this independent of global torch defaults that DeepXDE changes.
        start = torch.distributions.Dirichlet(torch.ones(3, dtype=dt)).sample((num_restarts, n_candidates))
        cands, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=n_candidates,
            num_restarts=num_restarts,
            batch_initial_conditions=start.to(self.device),
            equality_constraints=simplex,
        )
        return cands.detach().cpu().numpy()
