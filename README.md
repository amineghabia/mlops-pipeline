# MLOps Pipeline — LunarLander RL Agent

End-to-end MLOps pipeline for a Reinforcement Learning model: experiment tracking, REST API inference, and multi-service containerization.

The model (PPO) was trained on the `LunarLander-v3` environment and achieves a mean reward of **279/300** — well above the 200-point threshold for solving the environment.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              docker compose up               │
│                                             │
│  ┌──────────────────┐  ┌─────────────────┐  │
│  │   api (FastAPI)  │  │ mlflow (UI)     │  │
│  │   port 8000      │  │ port 5001       │  │
│  │                  │  │                 │  │
│  │  POST /predict   │  │ Experiment log  │  │
│  │  GET  /health    │  │ Params/metrics  │  │
│  └──────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## Project Structure

```
mlops-pipeline/
├── Dockerfile               # Python 3.11-slim, CPU-only PyTorch
├── docker-compose.yml       # API + MLflow services
├── requirements.txt         # Dependencies
├── serve.py                 # FastAPI inference server
├── train_with_mlflow.py     # PPO training with MLflow tracking
└── run.py                   # Quick local inference test (no API)
```

---

## Quickstart

### Prerequisites
- Docker Desktop running
- A trained PPO model saved as `lunarlander_ppo.zip` (train one with `train_with_mlflow.py`)

### 1. Train the model (with experiment tracking)

```bash
pip install -r requirements.txt
python train_with_mlflow.py
```

This logs every run to MLflow: hyperparameters, reward curve, and the model artifact.

### 2. Launch the full stack

```bash
docker compose up
```

- **API**: http://localhost:8000
- **MLflow UI**: http://localhost:5001

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Inference — send 8 LunarLander observations
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"obs": [0.1, 0.5, -0.3, -0.4, 0.02, 0.1, 0.0, 0.0]}'
```

Response:
```json
{
  "action": 2,
  "action_name": "moteur principal"
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Run PPO inference on an observation |

**Request body for `/predict`:**
```json
{
  "obs": [x, y, vx, vy, angle, angular_velocity, left_contact, right_contact]
}
```
8 floats corresponding to the LunarLander-v3 observation space.

---

## MLflow Tracking

Each training run logs:

| Type | Keys |
|------|------|
| Params | `timesteps`, `learning_rate`, `n_steps`, `policy`, `env` |
| Metrics | `reward_during_training` (every 10k steps), `mean_reward_final`, `std_reward_final` |
| Artifacts | Trained model `.zip` |

---

## Docker Details

- Base image: `python:3.11-slim`
- PyTorch installed CPU-only (`--index-url https://download.pytorch.org/whl/cpu`) to avoid ~2GB of unused CUDA packages
- Final image size: ~1.5 GB

---

## Model Results

| Metric | Value |
|--------|-------|
| Environment | LunarLander-v3 |
| Algorithm | PPO (Proximal Policy Optimization) |
| Training steps | 600,000 |
| Mean reward | **279.2 ± 8.5** |
| Solved threshold | 200 |
| Explained variance | 0.93 |
