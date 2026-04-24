"""
Ray Serve deployment for the humanoid policy.

One warm replica holds the loaded policy; sim workers call `predict` via
`handle.predict.remote(obs).result()` and receive actions.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from ray import serve


class EvalPolicy(nn.Module):
    def __init__(self, obs_dim=75, action_dim=21):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
        )
        self.action_mean = nn.Linear(100, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 400), nn.ELU(),
            nn.Linear(400, 200), nn.ELU(),
            nn.Linear(200, 100), nn.ELU(),
            nn.Linear(100, 1),
        )

    def forward(self, obs):
        return self.action_mean(self.policy_net(obs))


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=16,
)
class PolicyServer:
    def __init__(self, checkpoint_path: str):
        self.policy = EvalPolicy()
        state = torch.load(checkpoint_path, weights_only=False)
        self.policy.load_state_dict(state["policy"])
        self.policy.eval()
        self._call_count = 0
        self._total_latency_ms = 0.0

    async def predict(self, obs: np.ndarray) -> dict:
        t0 = time.time()
        with torch.no_grad():
            actions = torch.clamp(
                self.policy(torch.tensor(obs, dtype=torch.float32)), -1, 1
            ).numpy()
        latency_ms = (time.time() - t0) * 1000
        self._call_count += 1
        self._total_latency_ms += latency_ms
        return {"action": actions, "latency_ms": latency_ms}

    async def get_stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "avg_latency_ms": self._total_latency_ms / max(self._call_count, 1),
        }
