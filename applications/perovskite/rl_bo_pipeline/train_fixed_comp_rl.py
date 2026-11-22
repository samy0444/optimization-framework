# train_fixed_composition.py

import torch
from stable_baselines3 import SAC
from stable_baselines3 import SAC
from rl_bo_pipeline.environment_rl_bo import FabricationEnv




class RLTrainer:
    """
    Encapsulates RL training for a fixed perovskite composition.
    """
    def __init__(self, fixed_composition: dict):
        self.fixed_composition = fixed_composition

    def train(self, timesteps: int = 20_000):
        env = FabricationEnv(fixed_composition=self.fixed_composition)
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=1e-3,
            batch_size=64,
            train_freq=4, gradient_steps=4, 
            policy_kwargs=dict(net_arch=[64, 64]),
        )
        model.learn(total_timesteps=timesteps)
        return model




