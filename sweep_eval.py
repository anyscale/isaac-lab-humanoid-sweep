"""
Policy Robustness Sweep: Evaluate across perturbation conditions with Ray
=========================================================================

Fan out a pre-trained policy across different noise/perturbation configs
on parallel GPU workers. Aggregate into a pass/fail matrix.

Usage:
    python sweep_eval.py
"""

import ray
import numpy as np
import torch
import torch.nn as nn
import time
import json
import itertools
import os


# ── Policy (400→200→100, matches pre-trained checkpoint) ──

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
        return self.action_mean(h)


# ── Sweep configs ──

SWEEP_CONFIGS = []
obs_noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
action_noise_levels = [0.0, 0.05, 0.1, 0.2]

for obs_noise, act_noise in itertools.product(obs_noise_levels, action_noise_levels):
    SWEEP_CONFIGS.append({
        "obs_noise_std": obs_noise,
        "action_noise_std": act_noise,
    })

print(f"Total configs: {len(SWEEP_CONFIGS)} ({len(obs_noise_levels)} obs_noise x {len(action_noise_levels)} act_noise)")


# ── Remote eval function ──

@ray.remote(num_gpus=1)
def evaluate_config(config_id, config, checkpoint_path, num_envs=50, num_steps=3000):
    """
    Evaluate one perturbation config on one GPU.
    Applies obs noise and action noise around the env step loop.
    """
    from env import IsaacLabDirectEnv

    env = IsaacLabDirectEnv(
        task="Isaac-Humanoid-Direct-v0",
        num_envs=num_envs,
        device="cuda:0",
    )

    policy = ActorCritic(75, 21)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    obs_noise_std = config["obs_noise_std"]
    act_noise_std = config["action_noise_std"]

    obs = env.reset()
    episode_rewards = []
    ep_rewards = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs)

    for step in range(num_steps):
        # Add observation noise (simulates noisy sensors)
        noisy_obs = obs + np.random.normal(0, obs_noise_std, size=obs.shape).astype(np.float32) if obs_noise_std > 0 else obs

        with torch.no_grad():
            actions = torch.clamp(
                policy(torch.tensor(noisy_obs, dtype=torch.float32)), -1.0, 1.0
            ).numpy()

        # Add action noise (simulates motor jitter)
        if act_noise_std > 0:
            actions = actions + np.random.normal(0, act_noise_std, size=actions.shape).astype(np.float32)
            actions = np.clip(actions, -1.0, 1.0)

        obs, rewards, dones, _ = env.step(actions)
        ep_rewards += rewards
        ep_lengths += 1

        if np.any(dones):
            idx = np.where(dones)[0]
            episode_rewards.extend(ep_rewards[idx].tolist())
            ep_rewards[idx] = 0.0
            ep_lengths[idx] = 0

    env.close()

    # Include partial episodes (humanoids still alive at end)
    alive_mask = ep_rewards > 0
    if np.any(alive_mask):
        episode_rewards.extend(ep_rewards[alive_mask].tolist())

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    pass_rate = float(np.mean([r > 3000 for r in episode_rewards])) if episode_rewards else 0.0

    result = {
        "config_id": config_id,
        "obs_noise_std": obs_noise_std,
        "action_noise_std": act_noise_std,
        "mean_reward": mean_reward,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "pass_rate": pass_rate,
        "num_episodes": len(episode_rewards),
    }

    print(
        f"[Config {config_id:2d}] obs_noise={obs_noise_std:.2f} act_noise={act_noise_std:.2f} | "
        f"reward={mean_reward:7.1f} | pass={pass_rate:.0%} | episodes={len(episode_rewards)}",
        flush=True,
    )
    return result


# ── Main ──

def main():
    ray.init(runtime_env={"env_vars": {
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
    }})

    checkpoint = "/mnt/cluster_storage/checkpoints/humanoid/checkpoint_pretrained.pt"

    print(f"\n{'='*60}")
    print(f"  POLICY ROBUSTNESS SWEEP")
    print(f"  {len(SWEEP_CONFIGS)} configs x 50 envs = {len(SWEEP_CONFIGS)*50} total sims")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*60}\n")

    start = time.time()

    # Fan out all configs across GPUs — Ray schedules them
    futures = [
        evaluate_config.remote(i, cfg, checkpoint, num_envs=50, num_steps=3000)
        for i, cfg in enumerate(SWEEP_CONFIGS)
    ]

    results = ray.get(futures)
    elapsed = time.time() - start

    # Sort by config order
    results.sort(key=lambda r: r["config_id"])

    # Print matrix
    print(f"\n{'='*70}")
    print(f"  ROBUSTNESS MATRIX — Pass rate (reward > 3000)")
    print(f"{'='*70}")

    # Header
    act_noises = sorted(set(r["action_noise_std"] for r in results))
    header = f"{'obs_noise':>12s} |"
    for an in act_noises:
        header += f" act={an:.2f} |"
    print(header)
    print("-" * len(header))

    # Rows
    obs_noises = sorted(set(r["obs_noise_std"] for r in results))
    for on in obs_noises:
        row = f"{on:>12.2f} |"
        for an in act_noises:
            match = [r for r in results if r["obs_noise_std"] == on and r["action_noise_std"] == an]
            if match:
                pr = match[0]["pass_rate"]
                mr = match[0]["mean_reward"]
                cell = f" {pr:5.0%}  |"
            else:
                cell = f"   --   |"
            row += cell
        print(row)

    print(f"\n{'='*70}")
    print(f"  REWARD MATRIX — Mean reward per config")
    print(f"{'='*70}")

    header = f"{'obs_noise':>12s} |"
    for an in act_noises:
        header += f" act={an:.2f}  |"
    print(header)
    print("-" * len(header))

    for on in obs_noises:
        row = f"{on:>12.2f} |"
        for an in act_noises:
            match = [r for r in results if r["obs_noise_std"] == on and r["action_noise_std"] == an]
            if match:
                mr = match[0]["mean_reward"]
                cell = f" {mr:7.0f}  |"
            else:
                cell = f"     --   |"
            row += cell
        print(row)

    print(f"\nTotal wall clock time: {elapsed:.1f}s")
    print(f"Configs evaluated: {len(results)}")
    print(f"Total sim episodes: {sum(r['num_episodes'] for r in results)}")

    # Save results as JSON for visualization
    output_path = "/mnt/cluster_storage/sweep_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "obs_noise_levels": obs_noise_levels,
            "action_noise_levels": action_noise_levels,
            "elapsed_seconds": elapsed,
            "total_episodes": sum(r["num_episodes"] for r in results),
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Generate plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        obs_noises = sorted(set(r["obs_noise_std"] for r in results))
        act_noises = sorted(set(r["action_noise_std"] for r in results))

        # Build matrices
        pass_matrix = np.zeros((len(obs_noises), len(act_noises)))
        reward_matrix = np.zeros((len(obs_noises), len(act_noises)))
        for r in results:
            oi = obs_noises.index(r["obs_noise_std"])
            ai = act_noises.index(r["action_noise_std"])
            pass_matrix[oi, ai] = r["pass_rate"] * 100
            reward_matrix[oi, ai] = r["mean_reward"]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("Policy Robustness Sweep — Humanoid Locomotion", fontsize=16, fontweight="bold", y=1.02)

        # 1. Pass rate heatmap
        cmap_pass = LinearSegmentedColormap.from_list("rg", ["#E24B4A", "#FAEEDA", "#1D9E75"])
        im1 = axes[0].imshow(pass_matrix, cmap=cmap_pass, aspect="auto", vmin=0, vmax=100)
        axes[0].set_xticks(range(len(act_noises)))
        axes[0].set_xticklabels([f"{a:.2f}" for a in act_noises])
        axes[0].set_yticks(range(len(obs_noises)))
        axes[0].set_yticklabels([f"{o:.2f}" for o in obs_noises])
        axes[0].set_xlabel("Action Noise (std)")
        axes[0].set_ylabel("Observation Noise (std)")
        axes[0].set_title("Pass Rate (%)")
        for i in range(len(obs_noises)):
            for j in range(len(act_noises)):
                val = pass_matrix[i, j]
                color = "white" if val < 40 or val > 80 else "black"
                axes[0].text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # 2. Mean reward heatmap
        cmap_reward = LinearSegmentedColormap.from_list("br", ["#501313", "#FAEEDA", "#085041"])
        im2 = axes[1].imshow(reward_matrix, cmap=cmap_reward, aspect="auto")
        axes[1].set_xticks(range(len(act_noises)))
        axes[1].set_xticklabels([f"{a:.2f}" for a in act_noises])
        axes[1].set_yticks(range(len(obs_noises)))
        axes[1].set_yticklabels([f"{o:.2f}" for o in obs_noises])
        axes[1].set_xlabel("Action Noise (std)")
        axes[1].set_ylabel("Observation Noise (std)")
        axes[1].set_title("Mean Reward")
        for i in range(len(obs_noises)):
            for j in range(len(act_noises)):
                val = reward_matrix[i, j]
                color = "white" if val < reward_matrix.mean() * 0.5 else "black"
                axes[1].text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9, color=color)
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # 3. Bar chart — reward degradation by obs noise (averaged over action noise)
        avg_by_obs = [np.mean([r["mean_reward"] for r in results if r["obs_noise_std"] == on]) for on in obs_noises]
        avg_by_act = [np.mean([r["mean_reward"] for r in results if r["action_noise_std"] == an]) for an in act_noises]

        x = np.arange(len(obs_noises))
        bars = axes[2].bar(x, avg_by_obs, color=["#1D9E75" if v > 3000 else "#E24B4A" for v in avg_by_obs],
                           edgecolor="white", linewidth=1.5)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"{o:.2f}" for o in obs_noises])
        axes[2].set_xlabel("Observation Noise (std)")
        axes[2].set_ylabel("Mean Reward")
        axes[2].set_title("Reward vs Sensor Noise")
        axes[2].axhline(y=3000, color="#E24B4A", linestyle="--", linewidth=1.5, label="Deploy threshold")
        axes[2].legend()
        for bar, val in zip(bars, avg_by_obs):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()

        plot_path = "/mnt/cluster_storage/sweep_results.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Plots saved to {plot_path}")

        # Also save individual plots
        # Pass rate only
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im = ax2.imshow(pass_matrix, cmap=cmap_pass, aspect="auto", vmin=0, vmax=100)
        ax2.set_xticks(range(len(act_noises)))
        ax2.set_xticklabels([f"{a:.2f}" for a in act_noises])
        ax2.set_yticks(range(len(obs_noises)))
        ax2.set_yticklabels([f"{o:.2f}" for o in obs_noises])
        ax2.set_xlabel("Action Noise (std)", fontsize=12)
        ax2.set_ylabel("Observation Noise (std)", fontsize=12)
        ax2.set_title("Policy Deployment Readiness\nPass Rate by Perturbation Condition", fontsize=14, fontweight="bold")
        for i in range(len(obs_noises)):
            for j in range(len(act_noises)):
                val = pass_matrix[i, j]
                color = "white" if val < 40 or val > 80 else "black"
                ax2.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
        plt.colorbar(im, ax=ax2, shrink=0.8, label="Pass Rate (%)")
        fig2.savefig("/mnt/cluster_storage/sweep_passrate.png", dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Pass rate heatmap saved to /mnt/cluster_storage/sweep_passrate.png")

        plt.close("all")

    except ImportError:
        print("matplotlib not installed — skipping plots. pip install matplotlib to enable.")

    ray.shutdown()


if __name__ == "__main__":
    main()