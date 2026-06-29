import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

print("Chargement du modele LunarLander PPO...")
model = PPO.load("lunarlander_ppo")
env = gym.make("LunarLander-v3")

rewards = []
for ep in range(3):
    obs, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    rewards.append(total)
    print(f"Episode {ep + 1}/3 : recompense = {total:.1f}")

print(f"\nMoyenne : {np.mean(rewards):.1f}")
env.close()
