# evaluate_fixed_composition.py

from stable_baselines3.common.evaluation import evaluate_policy
# from environment import FabricationEnv
from rl_bo_pipeline.environment_rl_bo import FabricationEnv


# from stable_baselines3.common.evaluation import evaluate_policy
# from environment_rl_bo import FabricationEnv

# rl_bo_pipeline/evaluate_fixed_comp.py

# rl_bo_pipeline/evaluate_fixed_comp.py

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_bo_pipeline.environment_rl_bo import FabricationEnv

# rl_bo_pipeline/evaluate_fixed_comp.py

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_bo_pipeline.environment_rl_bo import FabricationEnv


class RLEvaluator:
    """
    Evaluates a trained RL agent on a fixed composition environment and returns performance.
    """
    def __init__(self, fixed_composition: dict):
        self.fixed_composition = fixed_composition

    def evaluate(self, model, vec_env=None, n_episodes: int = 10):
        # 1) If no vectorized env is provided, wrap a DummyVecEnv around a single env
        if vec_env is None:
            vec_env = DummyVecEnv([
                lambda: FabricationEnv(fixed_composition=self.fixed_composition)
            ])

        # 2) Use SB3's evaluate_policy on the VecEnv
        mean_reward, std_reward = evaluate_policy(
            model,
            vec_env,
            n_eval_episodes=n_episodes,
            deterministic=True
        )
        print(f"Mean reward over {n_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")

        # 3) Perform a single rollout on the raw Gym env to extract the final result
        env = FabricationEnv(fixed_composition=self.fixed_composition)
        reset_out = env.reset()
        # gym API v0.26+: reset returns (obs, info)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        action, _ = model.predict(obs, deterministic=True)

        step_out = env.step(action)
        # gym API v0.26+: step returns (obs, reward, terminated, truncated, info)
        if len(step_out) == 5:
            obs2, reward, terminated, truncated, info = step_out
        else:
            # older Gym: (obs, reward, done, info)
            obs2, reward, done, info = step_out
            terminated = done
            truncated = False

        print("Final rollout result:", info["result"])
        return info["result"], action


