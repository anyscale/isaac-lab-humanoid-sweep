import ray, numpy as np, torch, torch.nn as nn, time, json, itertools

ray.init(runtime_env={'env_vars': {
    'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json',
    'VK_DRIVER_FILES': '/etc/vulkan/icd.d/nvidia_icd.json',
    'OMNI_KIT_ACCEPT_EULA': 'YES', 'ACCEPT_EULA': 'Y',
}})

class EvalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(nn.Linear(75,400),nn.ELU(),nn.Linear(400,200),nn.ELU(),nn.Linear(200,100),nn.ELU())
        self.action_mean = nn.Linear(100,21)
        self.action_log_std = nn.Parameter(torch.zeros(21))
        self.value_net = nn.Sequential(nn.Linear(75,400),nn.ELU(),nn.Linear(400,200),nn.ELU(),nn.Linear(200,100),nn.ELU(),nn.Linear(100,1))
    def forward(self, obs):
        return self.action_mean(self.policy_net(obs))

CHECKPOINT = '/mnt/cluster_storage/checkpoints/humanoid/checkpoint_pretrained.pt'

@ray.remote(num_gpus=1)
def evaluate_config(config_id, config, checkpoint_path, num_envs=20, num_steps=1500):
    from env import IsaacLabDirectEnv
    env = IsaacLabDirectEnv(task='Isaac-Humanoid-Direct-v0', num_envs=num_envs, device='cuda:0')
    policy = EvalPolicy()
    policy.load_state_dict(torch.load(checkpoint_path, weights_only=False)['policy'])
    policy.eval()
    obs = env.reset()
    episode_rewards, ep_rewards = [], np.zeros(num_envs)
    for step in range(num_steps):
        noisy_obs = obs + np.random.normal(0, config['obs_noise_std'], size=obs.shape).astype(np.float32) if config['obs_noise_std'] > 0 else obs
        with torch.no_grad():
            actions = torch.clamp(policy(torch.tensor(noisy_obs, dtype=torch.float32)), -1, 1).numpy()
        if config['action_noise_std'] > 0:
            actions = np.clip(actions + np.random.normal(0, config['action_noise_std'], size=actions.shape).astype(np.float32), -1, 1)
        obs, rewards, dones, _ = env.step(actions)
        ep_rewards += rewards
        if np.any(dones):
            idx = np.where(dones)[0]
            episode_rewards.extend(ep_rewards[idx].tolist())
            ep_rewards[idx] = 0.0
    env.close()
    alive = ep_rewards > 0
    if np.any(alive):
        episode_rewards.extend(ep_rewards[alive].tolist())
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    pass_rate = float(np.mean([r > 3000 for r in episode_rewards])) if episode_rewards else 0.0
    print(f'[Config {config_id:2d}] obs={config["obs_noise_std"]:.2f} act={config["action_noise_std"]:.2f} | reward={mean_reward:7.0f} | pass={pass_rate:.0%}', flush=True)
    return {'config_id': config_id, 'obs_noise_std': config['obs_noise_std'],
        'action_noise_std': config['action_noise_std'], 'mean_reward': mean_reward,
        'pass_rate': pass_rate, 'num_episodes': len(episode_rewards),
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0}

obs_noise_levels = [0.0, 0.1]
action_noise_levels = [0.0, 0.1]
configs = [{'obs_noise_std': on, 'action_noise_std': an} for on, an in itertools.product(obs_noise_levels, action_noise_levels)]
print(f'Launching {len(configs)} configs...')
start = time.time()
futures = [evaluate_config.remote(i, cfg, CHECKPOINT) for i, cfg in enumerate(configs)]
results = ray.get(futures)
elapsed = time.time() - start
results.sort(key=lambda r: r['config_id'])
with open('/mnt/cluster_storage/sweep_results_full.json', 'w') as f:
    json.dump({'results': results, 'obs_noise_levels': obs_noise_levels, 'action_noise_levels': action_noise_levels, 'elapsed': elapsed, 'total_episodes': sum(r['num_episodes'] for r in results)}, f, indent=2)
print(f'Done: {len(results)} configs, {elapsed:.0f}s')
ray.shutdown()