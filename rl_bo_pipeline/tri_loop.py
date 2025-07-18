# rl_bo_pipeline/tri_loop.py

import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_bo_pipeline.physics_informed_bo import PhysicsInformedBO
from rl_bo_pipeline.evaluate_fixed_comp import RLEvaluator
from rl_bo_pipeline.environment_rl_bo import FabricationEnv

class TriLoop:
    """
    Orchestrates the tri-level loop:
       1) physics-informed BO → 2) RL fine-tuning → 3) evaluation → 4) BO update
    """
    def __init__(
        self,
        base_agent_path: str = "fab_agent.zip",
        data_root: str = "cluster-perovskite-data/Data",
        bo_device: str = "cpu",
        n_envs: int = 4,
        vec_env: str = "subproc",  # or "dummy"
    ):
        # 1) BO module
        self.bo = PhysicsInformedBO(data_root=data_root, device=bo_device)
        # 2) load base SAC agent (pretrained on some representative comp)
        self.base_agent_path = base_agent_path
        assert os.path.exists(base_agent_path), f"Base agent not found: {base_agent_path}"
        self.history = []
        self.n_envs = n_envs
        self.vec_env_type = vec_env

    def make_env_fn(self, fixed_composition):
        """Return a function that instantiates FabricationEnv with a given comp."""
        def _init():
            return FabricationEnv(fixed_composition=fixed_composition)
        return _init

    def create_vec_env(self, fixed_composition):
        """Wrap either Subproc or Dummy vectorized env factory."""
        env_fns = [self.make_env_fn(fixed_composition) for _ in range(self.n_envs)]
        if self.vec_env_type == "subproc":
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)

    def run_cycle(self, cycle: int, bo_batch: int, rl_steps: int):
        # 1) Fit GP & suggest compositions
        self.bo.update_gp()
        comp_batch = self.bo.suggest(n_candidates=bo_batch)

        for cs, ma, fa in comp_batch:
            fixed = {"Cs": float(cs), "MA": float(ma), "FA": float(fa)}

            # 2) Build a vectorized env for this composition
            vec_env = self.create_vec_env(fixed)

            # 3) Load the base agent and fine-tune
            agent = SAC.load(self.base_agent_path, env=vec_env)
            agent.learn(total_timesteps=rl_steps)

            # 4) Save the fine-tuned agent for next cycles
            agent_path = f"fab_agent_cycle{cycle+1}_{cs:.2f}_{ma:.2f}_{fa:.2f}.zip"
            agent.save(agent_path)
            # next cycle will start from this agent
            self.base_agent_path = agent_path

            # 5) Evaluate performance
            evaluator = RLEvaluator(fixed_composition=fixed)
            sim_res, best_action = evaluator.evaluate(agent)

            # 6) Update BO
            self.bo.X = np.vstack([self.bo.X, [cs, ma, fa]])
            self.bo.y = np.vstack([self.bo.y, [sim_res["log_lifetime"]]])

            # 7) Record history
            self.history.append({
                "cycle":      cycle,
                "composition": fixed,
                "action":     best_action,
                "eff":        sim_res["efficiency"],
                "log_life":   sim_res["log_lifetime"],
                "agent":      agent_path,
            })

        return self.history

    def run(self, num_cycles: int, bo_batch: int, rl_steps: int):
        for c in range(num_cycles):
            print(f"\n=== Starting Cycle {c+1}/{num_cycles} ===")
            self.run_cycle(c, bo_batch, rl_steps)
        return self.history



# # rl_bo_pipeline/tri_loop.py
# from rl_bo_pipeline.physics_informed_bo import PhysicsInformedBO
# # from rl_bo_pipeline.train_fixed_comp_rl import RLTrainer
# from rl_bo_pipeline.evaluate_fixed_comp import RLEvaluator
# import numpy as np


# class TriLoop:
#     """
#     Orchestrates the tri-level loop: physics-informed BO → RL training → evaluation.
#     """
#     def __init__(
#         self,
#         data_root: str = "cluster-perovskite-data/Data",
#         bo_device: str = "cpu",
#     ):
#         # initialize physics-informed BO with real MIT data
#         self.bo = PhysicsInformedBO(data_root=data_root, device=bo_device)
#         self.history = []
#     def make_env():
#         # pick a representative composition to warm-start on (e.g. Cs=0.4, MA=0.4, FA=0.2)
#         fixed = {"Cs": 0.4, "MA": 0.4, "FA": 0.2}
#         return FabricationEnv(fixed_composition=fixed)


#     def run_cycle(self, cycle: int, bo_batch: int, rl_steps: int):
#         # 1) Fit GP & suggest new compositions
#         self.bo.update_gp()
#         comp_batch = self.bo.suggest(n_candidates=bo_batch)

#         for cs, fa, ma in comp_batch:
#             fixed = {"Cs": float(cs), "MA": float(ma), "FA": float(fa)}

#             # 2) Train RL for this fixed composition
#             # trainer = RLTrainer(fixed_composition=fixed)
#             # agent = trainer.train(timesteps=rl_steps)
#             # wrap in a VecEnv for efficiency
#             env = DummyVecEnv([make_env for _ in range(4)])
#             agent = SAC.load("fab_agent.zip", env=env)
#             # 3) Evaluate to get final performance
#             evaluator = RLEvaluator(fixed_composition=fixed)
#             sim_res, best_action = evaluator.evaluate(agent)

#             # 4) Update BO with new data
#             self.bo.X = np.vstack([self.bo.X, [cs, fa, ma]])
#             self.bo.y = np.vstack([self.bo.y, [sim_res["log_lifetime"]]])

#             # 5) Record history
#             self.history.append({
#                 "cycle": cycle,
#                 "composition": fixed,
#                 "action": best_action,
#                 "eff": sim_res["efficiency"],
#                 "log_life": sim_res["log_lifetime"],
#             })

#         return self.history

#     def run(self, num_cycles: int, bo_batch: int, rl_steps: int):
#         for c in range(num_cycles):
#             print(f"=== Cycle {c+1}/{num_cycles} ===")
#             self.run_cycle(c, bo_batch, rl_steps)
#         return self.history
