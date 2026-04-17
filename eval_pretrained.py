"""Evaluate pre-trained humanoid policy across parallel sims with Ray."""
import ray
import numpy as np
import torch
import torch.nn as nn
import time

class ActorCritic(nn.Module):
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
        h = self.policy_net(obs)
        mean = self.action_mean(h)
        return mean

ray.init(runtime_env={"env_vars": {
    "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
    "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "ACCEPT_EULA": "Y",
}})

@ray.remote(num_gpus=1)
def evaluate_batch(worker_id, checkpoint_path, num_envs=50, num_steps=2000):
    from env import IsaacLabDirectEnv
    import torch, numpy as np

    env = IsaacLabDirectEnv(task="Isaac-Humanoid-Direct-v0", num_envs=num_envs, device="cuda:0")

    policy = ActorCritic(75, 21)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    obs = env.reset()
    episode_rewards = []
    ep_rewards = np.zeros(num_envs)

    for step in range(num_steps):
        with torch.no_grad():
            mean = policy(torch.tensor(obs, dtype=torch.float32))
            actions = torch.clamp(mean, -1.0, 1.0).numpy()
        obs, rewards, dones, _ = env.step(actions)
        ep_rewards += rewards
        if np.any(dones):
            idx = np.where(dones)[0]
            episode_rewards.extend(ep_rewards[idx].tolist())
            ep_rewards[idx] = 0.0

    env.close()
    return {"worker_id": worker_id, "episode_rewards": episode_rewards}

CHECKPOINT = "/mnt/cluster_storage/checkpoints/humanoid/checkpoint_pretrained.pt"
NUM_WORKERS = 4
ENVS_PER_WORKER = 50
PASS_THRESHOLD = 5000

print(f"Launching {NUM_WORKERS} workers x {ENVS_PER_WORKER} envs = {NUM_WORKERS * ENVS_PER_WORKER} parallel sims")
start = time.time()

futures = [evaluate_batch.remote(i, CHECKPOINT, num_envs=ENVS_PER_WORKER) for i in range(NUM_WORKERS)]
results = ray.get(futures)
elapsed = time.time() - start

all_rewards = []
for r in results:
    all_rewards.extend(r["episode_rewards"])

if all_rewards:
    pass_count = sum(1 for r in all_rewards if r > PASS_THRESHOLD)
    print()
    print("=" * 50)
    print("       POLICY EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Total episodes:     {len(all_rewards)}")
    print(f"  Mean reward:        {np.mean(all_rewards):.1f}")
    print(f"  Std reward:         {np.std(all_rewards):.1f}")
    print(f"  Min / Max:          {np.min(all_rewards):.1f} / {np.max(all_rewards):.1f}")
    print(f"  Pass rate (>{PASS_THRESHOLD}):  {pass_count}/{len(all_rewards)} ({100*pass_count/len(all_rewards):.0f}%)")
    print(f"  Wall clock time:    {elapsed:.1f}s")
    print("=" * 50)
else:
    print("No episodes completed. Increase num_steps.")

ray.shutdown()
