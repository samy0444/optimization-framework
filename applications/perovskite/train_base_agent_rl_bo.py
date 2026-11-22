import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from rl_bo_pipeline.environment_rl_bo import FabricationEnv


def make_env(fixed_composition):
    """
    Factory for creating independent FabricationEnv instances.
    """
    def _init():
        env = FabricationEnv(fixed_composition=fixed_composition)
        return env
    return _init


def train_base_agent(
    fixed_composition={"Cs":0.33,"MA":0.33,"FA":0.34},
    timesteps=100_000,
    n_envs=4,
    save_path="fab_agent.zip",
    seed=42
):
    envs = [make_env(fixed_composition) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(envs)

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=100_000,
        tau=0.005,
        train_freq=4,
        gradient_steps=4,
        policy_kwargs={"net_arch": [64, 64]},
    )

    model.learn(total_timesteps=timesteps)
    if save_path:
        model.save(save_path)
        print(f"Saved fabrication agent to {save_path}")
    return model

if __name__=="__main__":
    train_base_agent()
