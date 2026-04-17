"""
Distributed PPO Training: Any Isaac Lab Robot + Ray Core
=========================================================

Train any Isaac Lab robot at scale on Anyscale.

Supported tasks (Isaac Lab built-in):
  - Isaac-Humanoid-Direct-v0
  - Isaac-Ant-Direct-v0
  - Isaac-Cartpole-Direct-v0
  - Isaac-Velocity-Rough-Anymal-C-Direct-v0
  - Isaac-Velocity-Flat-Anymal-C-Direct-v0
  - Isaac-Reach-Franka-Direct-v0

Usage:
  python train_general.py --task Isaac-Cartpole-Direct-v0 --num-workers 2
  python train_general.py --task Isaac-Humanoid-Direct-v0 --num-workers 2
  python train_general.py --task Isaac-Ant-Direct-v0 --num-workers 2
"""

import argparse
import os
import time

import numpy as np
import ray
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Task configs
# ──────────────────────────────────────────────

TASK_CONFIGS = {
    "Isaac-Humanoid-Direct-v0": {"obs_dim": 75, "action_dim": 21, "target_reward": 8000},
    "Isaac-Ant-Direct-v0": {"obs_dim": 37, "action_dim": 8, "target_reward": 6000},
    "Isaac-Cartpole-Direct-v0": {"obs_dim": 4, "action_dim": 1, "target_reward": 500},
    "Isaac-Velocity-Rough-Anymal-C-Direct-v0": {"obs_dim": 48, "action_dim": 12, "target_reward": 20},
    "Isaac-Velocity-Flat-Anymal-C-Direct-v0": {"obs_dim": 48, "action_dim": 12, "target_reward": 20},
    "Isaac-Reach-Franka-Direct-v0": {"obs_dim": 12, "action_dim": 7, "target_reward": 100},
}


# ──────────────────────────────────────────────
# General Actor-Critic
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )
        self.action_mean = nn.Linear(64, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        h = self.policy_net(obs)
        mean = self.action_mean(h)
        mean = torch.clamp(mean, -100.0, 100.0)
        log_std = torch.clamp(self.action_log_std, -5.0, 2.0).expand_as(mean)
        value = self.value_net(obs).squeeze(-1)
        return mean, log_std, value

    def get_action_and_value(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32)
            mean, log_std, value = self.forward(x)
            std = torch.exp(log_std).clamp(min=1e-6)
            try:
                dist = torch.distributions.Normal(mean, std)
                action = torch.clamp(dist.sample(), -1.0, 1.0)
                log_prob = dist.log_prob(action).sum(dim=-1)
            except RuntimeError:
                action = torch.clamp(mean + 0.5 * torch.randn_like(mean), -1.0, 1.0)
                log_prob = torch.zeros(mean.shape[0])
            return action.numpy(), log_prob.numpy(), value.numpy()

    def evaluate_actions(self, obs, actions):
        mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std).clamp(min=1e-6)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_prob, entropy, value


# ──────────────────────────────────────────────
# Worker — runs Isaac Lab directly
# ──────────────────────────────────────────────

@ray.remote(num_gpus=1)
class SimWorker:
    def __init__(self, worker_id, task, obs_dim, action_dim, num_envs=512):
        self.worker_id = worker_id
        self.task = task
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        from env import IsaacLabDirectEnv
        self.env = IsaacLabDirectEnv(
            task=task,
            num_envs=num_envs,
            device="cuda:0",
        )

        self.policy = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
        print(f"[Worker {worker_id}] Ready: {task} with {num_envs} envs", flush=True)

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.policy.state_dict().items()}

    def sample(self, num_steps=32):
        t_start = time.monotonic()
        obs_buf = np.zeros((num_steps, self.num_envs, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((num_steps, self.num_envs, self.action_dim), dtype=np.float32)
        rew_buf = np.zeros((num_steps, self.num_envs), dtype=np.float32)
        done_buf = np.zeros((num_steps, self.num_envs), dtype=np.bool_)
        logp_buf = np.zeros((num_steps, self.num_envs), dtype=np.float32)
        val_buf = np.zeros((num_steps, self.num_envs), dtype=np.float32)

        obs = self.env.reset()
        episode_rewards = []
        ep_rewards = np.zeros(self.num_envs)

        for step in range(num_steps):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            actions, log_probs, values = self.policy.get_action_and_value(obs)
            next_obs, rewards, dones, infos = self.env.step(actions)
            rewards = np.nan_to_num(rewards, nan=0.0)
            rewards = np.clip(rewards, -100.0, 100.0)

            obs_buf[step] = obs
            act_buf[step] = actions
            rew_buf[step] = rewards
            done_buf[step] = dones
            logp_buf[step] = log_probs
            val_buf[step] = values

            ep_rewards += rewards
            if np.any(dones):
                done_idx = dones.nonzero()[0]
                episode_rewards.extend(ep_rewards[done_idx].tolist())
                ep_rewards[done_idx] = 0.0
            obs = next_obs

        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        with torch.no_grad():
            _, _, last_values = self.policy(torch.tensor(obs, dtype=torch.float32))
            last_values = last_values.numpy()

        return {
            "obs": obs_buf, "actions": act_buf, "rewards": rew_buf,
            "dones": done_buf, "log_probs": logp_buf, "values": val_buf,
            "last_values": last_values, "episode_rewards": episode_rewards,
            "sample_time": time.monotonic() - t_start,
        }

    def stop(self):
        self.env.close()


# ──────────────────────────────────────────────
# GAE
# ──────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    num_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(num_envs)
    for t in reversed(range(num_steps)):
        next_val = last_values if t == num_steps - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


# ──────────────────────────────────────────────
# PPO Update
# ──────────────────────────────────────────────

def ppo_update(policy, optimizer, obs, actions, old_log_probs, advantages, returns,
               ppo_epochs=2, minibatch_size=8192, clip_coef=0.2, vf_coef=0.5,
               ent_coef=0.01, max_grad_norm=0.5):
    device = next(policy.parameters()).device
    total = obs.shape[0] * obs.shape[1]

    obs_f = torch.nan_to_num(torch.tensor(obs.reshape(total, -1), dtype=torch.float32, device=device))
    act_f = torch.nan_to_num(torch.tensor(actions.reshape(total, -1), dtype=torch.float32, device=device))
    olp_f = torch.nan_to_num(torch.tensor(old_log_probs.reshape(total), dtype=torch.float32, device=device))
    adv_f = torch.nan_to_num(torch.tensor(advantages.reshape(total), dtype=torch.float32, device=device))
    ret_f = torch.nan_to_num(torch.tensor(returns.reshape(total), dtype=torch.float32, device=device))
    adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

    pl, vl, ent, n = 0.0, 0.0, 0.0, 0
    for _ in range(ppo_epochs):
        idx = torch.randperm(total, device=device)
        for s in range(0, total, minibatch_size):
            b = idx[s:min(s + minibatch_size, total)]
            lp, en, val = policy.evaluate_actions(obs_f[b], act_f[b])
            if torch.isnan(lp).any() or torch.isnan(val).any():
                continue
            ratio = torch.clamp(torch.exp(lp - olp_f[b]), 0.01, 100.0)
            s1 = ratio * adv_f[b]
            s2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv_f[b]
            p_loss = -torch.min(s1, s2).mean()
            v_loss = 0.5 * (val - ret_f[b]).pow(2).mean()
            loss = p_loss + vf_coef * v_loss - ent_coef * en
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            pl += p_loss.item(); vl += v_loss.item(); ent += en.item(); n += 1

    if n == 0:
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0}
    return {"policy_loss": pl / n, "value_loss": vl / n, "entropy": ent / n}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Isaac-Humanoid-Direct-v0",
                        choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=512)
    parser.add_argument("--num-iters", type=int, default=10000)
    parser.add_argument("--steps-per-sample", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=16384)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task]
    obs_dim = cfg["obs_dim"]
    action_dim = cfg["action_dim"]
    target_reward = cfg["target_reward"]

    task_short = args.task.replace("Isaac-", "").replace("-Direct-v0", "").replace("-v0", "").lower()
    checkpoint_dir = f"/mnt/cluster_storage/checkpoints/{task_short}"

    ray.init(runtime_env={"env_vars": {
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
    }})

    print(f"\n{'='*60}")
    print(f"Task:        {args.task}")
    print(f"Obs dim:     {obs_dim}")
    print(f"Action dim:  {action_dim}")
    print(f"Target:      {target_reward}")
    print(f"Workers:     {args.num_workers}")
    print(f"Envs/worker: {args.num_envs_per_worker}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*60}\n")

    device = torch.device("cpu")
    policy = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    start_iter = 0
    total_env_steps = 0
    if args.resume_from and os.path.exists(os.path.join(args.resume_from, "checkpoint.pt")):
        ckpt = torch.load(os.path.join(args.resume_from, "checkpoint.pt"), weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iteration", 0) + 1
        total_env_steps = ckpt.get("total_env_steps", 0)
        print(f"[Driver] Resumed from iter {start_iter}, reward={ckpt.get('mean_reward', 0):.1f}")

    print(f"[Driver] Creating {args.num_workers} workers...")
    workers = [
        SimWorker.remote(i, args.task, obs_dim, action_dim,
                         num_envs=args.num_envs_per_worker)
        for i in range(args.num_workers)
    ]
    print("[Driver] Waiting for Isaac Lab to boot (~2-3 min)...")
    ray.get([w.get_weights.remote() for w in workers])
    print("[Driver] All workers ready!\n")

    best_mean_reward = -float("inf")

    for iteration in range(start_iter, args.num_iters):
        iter_start = time.monotonic()

        weights = {k: v.cpu() for k, v in policy.state_dict().items()}
        ray.get([w.set_weights.remote(weights) for w in workers])

        t0 = time.monotonic()
        try:
            rollouts = ray.get(
                [w.sample.remote(num_steps=args.steps_per_sample) for w in workers],
                timeout=180,
            )
        except Exception as e:
            print(f"[Driver] Sampling error: {e}")
            for i, w in enumerate(workers):
                try:
                    ray.get(w.get_weights.remote(), timeout=10)
                except Exception:
                    print(f"[Driver] Restarting worker {i}")
                    try: ray.kill(w)
                    except: pass
                    workers[i] = SimWorker.remote(
                        i, args.task, obs_dim, action_dim,
                        num_envs=args.num_envs_per_worker
                    )
                    ray.get(workers[i].set_weights.remote(weights))
            continue
        sample_time = time.monotonic() - t0

        all_obs = np.concatenate([r["obs"] for r in rollouts], axis=1)
        all_actions = np.concatenate([r["actions"] for r in rollouts], axis=1)
        all_rewards = np.concatenate([r["rewards"] for r in rollouts], axis=1)
        all_dones = np.concatenate([r["dones"] for r in rollouts], axis=1)
        all_log_probs = np.concatenate([r["log_probs"] for r in rollouts], axis=1)
        all_values = np.concatenate([r["values"] for r in rollouts], axis=1)
        all_last_values = np.concatenate([r["last_values"] for r in rollouts])
        all_episode_rewards = []
        for r in rollouts:
            all_episode_rewards.extend(r["episode_rewards"])

        advantages, returns = compute_gae(all_rewards, all_values, all_dones, all_last_values)

        t0 = time.monotonic()
        losses = ppo_update(policy, optimizer, all_obs, all_actions, all_log_probs,
                           advantages, returns, ppo_epochs=args.ppo_epochs,
                           minibatch_size=args.minibatch_size)
        train_time = time.monotonic() - t0

        env_steps = args.steps_per_sample * args.num_envs_per_worker * args.num_workers
        total_env_steps += env_steps
        mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0.0
        iter_time = time.monotonic() - iter_start

        print(
            f"[{task_short:>10s} Iter {iteration:5d}] "
            f"reward={mean_reward:8.1f} | "
            f"episodes={len(all_episode_rewards):4d} | "
            f"steps={total_env_steps:,} | "
            f"sample={sample_time:.1f}s | train={train_time:.1f}s | total={iter_time:.1f}s",
            flush=True,
        )

        if mean_reward > best_mean_reward and all_episode_rewards:
            best_mean_reward = mean_reward
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "total_env_steps": total_env_steps,
                "mean_reward": mean_reward,
                "task": args.task,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
            print(f"[Driver] New best: {mean_reward:.1f}, saved to {checkpoint_dir}")

        if iteration % 50 == 0 and iteration > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "total_env_steps": total_env_steps,
                "mean_reward": mean_reward,
                "task": args.task,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
            }, os.path.join(checkpoint_dir, f"checkpoint_iter{iteration}.pt"))

        if mean_reward >= target_reward:
            print(f"[Driver] SOLVED {args.task}! reward={mean_reward:.1f}")
            break

    ray.get([w.stop.remote() for w in workers])
    ray.shutdown()
    print("[Driver] Done!")


if __name__ == "__main__":
    main()