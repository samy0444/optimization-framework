# environment_rl_bo.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
# from simulation import simulate_all_phy
from simulation_bandgap import simulate_with_bandgap
def anneal_time_penalty(T):
    return min(0.1 * (T / 200.0), 0.1)

def unnorm(x, low, high):
    return low + (x + 1.0) * (high - low) / 2.0

from layers import default_stack
# app.py (Streamlit excerpt)
from interfaces import Interface

etl_if  = Interface(name="ETL/Perovskite", recomb_velocity=1e3, trap_density=1e10)
perh_if = Interface(name="Perovskite/HTL", recomb_velocity=5e2, trap_density=5e9)

# spin-speed range kept inside the efficiency surrogate's training data (1000-5000 rpm)
SPIN_MIN, SPIN_MAX = 1000.0, 5000.0

default_layers = default_stack()

class FabricationEnv(gym.Env):
    """
    RL environment that optimizes the 6 fabrication knobs
    for a fixed Cs/MA/FA composition.
    Observation: 
      [stage_count, substrate, Pb_frac, uv_filter, spin_speed, anneal_T, encapsulation, Cs, MA, FA]
    Action: 6‐vector in [-1,1] for each step.
    """
    def __init__(self, fixed_composition):
        super().__init__()
        assert all(k in fixed_composition for k in ("Cs","MA","FA"))
        self.fixed_composition = fixed_composition

        # one action per stage: [substrate, Pb_frac, uv_filter, spin_speed, anneal_T, encapsulation]
        self.action_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)

        # observation has 1 (stage count) + 6 history + 3 comp = 10 dims
        low  = np.array([0.0, -1, -1, -1, -1, -1, -1,  0.0,0.0,0.0], dtype=np.float32)
        high = np.array([6.0,  1,  1,  1,  1,  1,  1,  1.0,1.0,1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # quick look‐ahead rewards at stage 1 and 3 (as before):
        self.stage_rewards = {
            1: lambda h: -abs(unnorm(h["Pb_frac"],0,1) - 0.5),
            3: lambda h: -abs(unnorm(h["spin_speed"],SPIN_MIN,SPIN_MAX) - 3000)/1000.0,
        }
        # Any RNG used in the env should be seeded here:
        self._rng = np.random.RandomState()
        self.reset()

    def _build_base_obs(self):
        # [stage_count, substrate, Pb_frac, uv_filter, spin_speed, anneal_T, encapsulation]
        return np.array([
            float(self.stage),
            self.history["substrate"],
            self.history["Pb_frac"],
            self.history["uv_filter"],
            self.history["spin_speed"],
            self.history["anneal_T"],
            self.history["encapsulation"],
        ], dtype=np.float32)

    def seed(self, seed=None):
            """Set random seed for reproducibility."""
            self._rng.seed(seed)
            return [seed]

    def reset(self, *, seed=None, options=None):
        """
        Gym API v0.26+ reset signature.
        Returns obs, info
        """
        # apply the seed if requested
        if seed is not None:
            self.seed(seed)

        self.stage = 0
        self.history = {
            "substrate":     0.0,
            "Pb_frac":       0.0,
            "uv_filter":     0.0,
            "spin_speed":    0.0,
            "anneal_T":      0.0,
            "encapsulation": 0.0,
        }

        base = self._build_base_obs()
        comp = np.array([
            self.fixed_composition["Cs"],
            self.fixed_composition["MA"],
            self.fixed_composition["FA"],
        ], dtype=np.float32)
        obs = np.concatenate([base, comp], axis=0)
        return obs, {}

    # def reset(self):
    #     self.stage = 0
    #     # neutral initial values
    #     self.history = {
    #         "substrate":    0.0,
    #         "Pb_frac":      0.0,
    #         "uv_filter":    0.0,
    #         "spin_speed":   0.0,
    #         "anneal_T":     0.0,
    #         "encapsulation":0.0,
    #     }
    #     base = self._build_base_obs()
    #     comp  = np.array([self.fixed_composition["Cs"],
    #                       self.fixed_composition["MA"],
    #                       self.fixed_composition["FA"]], dtype=np.float32)
    #     return np.concatenate([base, comp], axis=0)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        # decode
        self.history["substrate"]     = -1.0 if int(round(a[0])) == 0 else 1.0
        self.history["Pb_frac"]       = a[1]
        self.history["uv_filter"]     = a[2]
        self.history["spin_speed"]    = a[3]
        self.history["anneal_T"]      = a[4]
        self.history["encapsulation"] = -1.0 if int(round(a[5])) == 0 else 1.0

        # stage‐wise shaping reward
        reward = 0.0
        for s, fn in self.stage_rewards.items():
            reward += fn(self.history)

        # final simulation call
        params = {
            "layers": default_layers,  # standard 5‐layer list of Layer(...)
            "composition": self.fixed_composition,
            "uv_filter":   unnorm(self.history["uv_filter"], 0.0, 1.0),
            "spin_speed":  unnorm(self.history["spin_speed"], SPIN_MIN, SPIN_MAX),
            "anneal_T":    unnorm(self.history["anneal_T"], 50.0, 200.0),
            "substrate":   "Glass" if self.history["substrate"] < 0 else "PET",
            "encapsulation": self.history["encapsulation"] > 0,
            "interfaces": [etl_if, perh_if]
        }
        res = simulate_with_bandgap(params)
        reward += res["efficiency"]*20 + res["log_lifetime"] - 0.05*res.get("time_cost",0)

        # build next obs (episode ends here)
        base = self._build_base_obs()
        comp = np.array([self.fixed_composition["Cs"],
                         self.fixed_composition["MA"],
                         self.fixed_composition["FA"]], dtype=np.float32)
        obs  = np.concatenate([base, comp], axis=0)
        truncated = False    # don’t have a time limit truncation, so False

        return obs, reward, True,truncated, {"result":res}


