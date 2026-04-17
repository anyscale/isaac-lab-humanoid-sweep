# Distributed Robot Sim with Ray Core + Isaac Lab

Train, evaluate, and stress-test robot policies at scale on Anyscale — no RLlib, no custom frameworks, just `@ray.remote`.

```
┌───────────────────────────────────────────────────────────────┐
│                      ANYSCALE CLUSTER                          │
│                                                               │
│  ┌──────────┐        @ray.remote(num_gpus=1)                 │
│  │  Driver   │──────────┬──────────┬──────────┬──────────┐   │
│  │ (notebook)│          │          │          │          │   │
│  └──────────┘          │          │          │          │   │
│            ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│            │  GPU 0  │ │  GPU 1  │ │  GPU 2  │ │  GPU 3  │  │
│            │IsaacLab │ │IsaacLab │ │IsaacLab │ │IsaacLab │  │
│            │ 20 envs │ │ 20 envs │ │ 20 envs │ │ 20 envs │  │
│            │ Perfect │ │ Factory │ │Outdoor  │ │Rain/dust│  │
│            │ sensors │ │  floor  │ │ field   │ │end-life │  │
│            └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
│                                                               │
│            500 parallel sims · 25 configs · 4 GPUs            │
└───────────────────────────────────────────────────────────────┘
```

## What this demo shows

| Capability | What Ray does | Time |
|-----------|--------------|------|
| **Distributed PPO training** | N GPU workers collect rollouts in parallel, driver runs PPO update | ~2 hrs |
| **Robustness sweep** | 25 perturbation configs fanned out across 4 GPUs | ~20 min |
| **Interactive results** | Heatmaps, bar charts, deployment readiness dashboard | instant |

The sweep produces a **deployment readiness heatmap** — green where the policy is safe, red where it fails, with real-world labels like "Factory floor" and "End-of-life motors."

## Why this matters

### Results preview

<p align="center">
  <img src="media/sample.gif" width="400" alt="Trained humanoid walking">
</p>

<p align="center">
  <img src="media/heatmap_passrate.png" width="600" alt="Deployment readiness heatmap">
</p>

<p align="center">
  <img src="media/heatmap_reward.png" width="600" alt="Mean reward heatmap">
</p>

Robotics companies train policies in simulation — but simulation is perfect. Real robots have noisy sensors, worn-out motors, and unpredictable environments. The gap between "works in sim" and "works on hardware" has crashed robots and ended projects.

This demo shows how to close that gap: fan out hundreds of stress tests across GPUs in minutes, and get a map of exactly where the policy is safe to deploy.

Three steps with Ray Core:
1. **Wrap** the sim → `env.py` (subprocess isolation via `multiprocessing.Pipe`)
2. **`@ray.remote(num_gpus=1)`** → runs on any GPU in the cluster
3. **Fan out** 25 configs → Ray schedules, runs, aggregates results


## Files

| File | Description |
|------|-------------|
| `isaac_lab_ray_core_DEMO.ipynb` | **Start here** — end-to-end notebook with heatmaps and deployment dashboard |
| `env.py` | Isaac Lab wrapper — subprocess isolation, `.reset()` / `.step()` interface |
| `train_general.py` | Distributed PPO training script, any Isaac Lab task |
| `sweep_eval.py` | Standalone sweep script (alternative to notebook) |
| `eval_pretrained.py` | Quick evaluation of pre-trained checkpoint |
| `Containerfile` | Anyscale image: Ray + Isaac Sim 5.1 + Isaac Lab |
| `media/sample.mp4` | Video of trained humanoid walking |
| `README.md` | This file |

## Quick start

### 1. Workspace setup

Create an Anyscale workspace with:
- **Image:** Custom image built from `Containerfile` (base: `anyscale/ray:2.53.0-slim-py311-cu128`)
- **Head:** `m5.2xlarge` (CPU)
- **Workers:** 4× `g5.xlarge` (A10G GPU)

### 2. Run the notebook

Open `isaac_lab_ray_core_DEMO.ipynb` and **Run All**. It will:
1. Load pre-trained humanoid policy (reward ~8000)
2. Load pre-computed sweep results from JSON (instant)
3. Render heatmaps and deployment readiness dashboard

The notebook loads pre-computed results by default — **no GPU wait time, plots render instantly.** To run a live sweep, uncomment the sweep cell (~20 min on 4 GPUs).

### 3. (Optional) Train from scratch

```bash
python train_general.py \
  --task Isaac-Humanoid-Direct-v0 \
  --num-workers 4 \
  --num-envs-per-worker 512 \
  --lr 3e-4 \
  --num-iters 5000
```

## The robustness sweep

The core demo is the **policy robustness sweep**. Before deploying a robot, test it under real-world conditions:

| Sensor quality | What it simulates | Noise level |
|---------------|-------------------|-------------|
| **Perfect sensors** | Lab-calibrated IMU, zero drift | 0.00 |
| **Lab-grade IMU** | High-quality industrial sensors | 0.02 |
| **Factory floor** | Standard sensors, some vibration | 0.05 |
| **Outdoor field** | Dust, temperature variation | 0.10 |
| **Rain / dust** | Degraded, wet, dirty sensors | 0.20 |

| Motor condition | What it simulates | Noise level |
|----------------|-------------------|-------------|
| **New motors** | Fresh actuators, no wear | 0.00 |
| **Broken in** | Light use, minimal drift | 0.02 |
| **6-month wear** | Normal operational wear | 0.05 |
| **Needs service** | Overdue for maintenance | 0.10 |
| **End of life** | Worn bearings, backlash | 0.20 |

5 × 5 = **25 configs**, each a separate `@ray.remote` task on a GPU with 20 parallel humanoid sims running 1500 steps.

**The customer story:** "This policy passes 100% in clean conditions but drops to 31% with rain and end-of-life motors — here's exactly where the boundary is. Safe to deploy up to factory floor conditions with serviced motors."

## Supported tasks

| Task | Robot | Obs | Act | Use case |
|------|-------|-----|-----|----------|
| `Isaac-Humanoid-Direct-v0` | Humanoid | 75 | 21 | Bipedal locomotion |
| `Isaac-Ant-Direct-v0` | Ant | 37 | 8 | Multi-legged control |
| `Isaac-Cartpole-Direct-v0` | Cartpole | 4 | 1 | Balancing (tutorial) |
| `Isaac-Reach-Franka-Direct-v0` | Franka arm | 12 | 7 | Manipulation |
| `Isaac-Velocity-Rough-Anymal-C-Direct-v0` | ANYmal | 48 | 12 | Rough terrain locomotion |

Swap any task into the notebook — the env wrapper, training script, and sweep all work with any Isaac Lab task.

## Pre-trained checkpoint

A pre-trained humanoid policy (reward ~8000) is available at:
```
s3://air-example-data/isaac_lab_template_checkpoint/
```

Converted to plain PyTorch format at `/mnt/cluster_storage/checkpoints/humanoid/checkpoint_pretrained.pt`. Architecture: 75→400→200→100 (policy net), separate value net with same dimensions.

## Image setup

The Containerfile pip-installs Isaac Sim 5.1.0 directly into the Anyscale Ray base image. **No nested containers, no Podman, no Docker-in-Docker.**

- Base: `anyscale/ray:2.53.0-slim-py311-cu128` (Python 3.11, CUDA 12.8)
- Isaac Sim via `pip install isaacsim[all]==5.1.0`
- Isaac Lab from source (GitHub)
- Headless GPU rendering with Vulkan + EGL
- Builds in Anyscale's image builder (~20 min first time, cached after)

### Environment wrapper

`env.py` runs Isaac Lab in a **subprocess** via `multiprocessing.Pipe`. Isaac Sim's Kit engine needs its own event loop — this gives it one while keeping Ray's event loop clean.

```python
from env import IsaacLabDirectEnv

env = IsaacLabDirectEnv(task="Isaac-Humanoid-Direct-v0", num_envs=1024, device="cuda:0")
obs = env.reset()                              # (1024, 75) numpy array
obs, rewards, dones, infos = env.step(actions)  # numpy in, numpy out
env.close()
```

The user never sees subprocesses or pipes — just `.reset()` and `.step()`.

## Documentation

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Anyscale workspaces](https://docs.anyscale.com/platform/workspaces)

## Special instructions

- **GPU workers boot Isaac Lab on first use** — takes ~2 min for NVIDIA PhysX to load. This is expected.
- Results auto-save to `/mnt/cluster_storage/sweep_results_full.json` — if the kernel restarts, the plot cells reload from there automatically.
- To scale up: add more noise levels to the sweep grid and add more GPU workers.
- The `omni.physx.plugin` warnings in worker logs are normal Isaac Lab behavior and can be ignored.
