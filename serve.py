from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO
import numpy as np

app = FastAPI(title="LunarLander Inference API")
model = PPO.load("lunarlander_ppo")

class Observation(BaseModel):
    obs: list[float]  # 8 valeurs : position (x,y), vitesse (x,y), angle, vitesse angulaire, contact pieds (G,D)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Observation):
    obs = np.array(data.obs, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action_names = {0: "rien", 1: "moteur gauche", 2: "moteur principal", 3: "moteur droit"}
    return {
        "action": int(action),
        "action_name": action_names[int(action)],
    }
