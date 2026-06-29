import mlflow
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

TIMESTEPS = 100_000
LEARNING_RATE = 3e-4
N_STEPS = 2048
POLICY = "MlpPolicy"

class MLflowCallback(BaseCallback):
    """Log la récompense moyenne toutes les 10k étapes."""
    def __init__(self, eval_env, log_freq=10_000):
        super().__init__()
        self.eval_env = eval_env
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            mean_r, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5, warn=False)
            mlflow.log_metric("reward_during_training", mean_r, step=self.num_timesteps)
            print(f"  step {self.num_timesteps:>7} | reward moyen : {mean_r:.1f}")
        return True


mlflow.set_experiment("lunarlander-ppo")

with mlflow.start_run():
    mlflow.log_params({
        "timesteps": TIMESTEPS,
        "learning_rate": LEARNING_RATE,
        "n_steps": N_STEPS,
        "policy": POLICY,
        "env": "LunarLander-v3",
    })

    env = gym.make("LunarLander-v3")
    eval_env = gym.make("LunarLander-v3")
    model = PPO(POLICY, env, learning_rate=LEARNING_RATE, n_steps=N_STEPS, verbose=0)

    print("Entraînement en cours...")
    model.learn(total_timesteps=TIMESTEPS, callback=MLflowCallback(eval_env))

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    mlflow.log_metric("mean_reward_final", mean_reward)
    mlflow.log_metric("std_reward_final", std_reward)

    model.save("lunarlander_mlflow_run")
    mlflow.log_artifact("lunarlander_mlflow_run.zip")

    print(f"\nResultat final : {mean_reward:.1f} ± {std_reward:.1f}")
    print("Run MLflow enregistre.")

    env.close()
    eval_env.close()
