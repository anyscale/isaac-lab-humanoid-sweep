import ray, numpy as np, time, json, itertools
from ray import serve

from policy_server import PolicyServer

ray.init(runtime_env={'env_vars': {
    'VK_ICD_FILENAMES': '/etc/vulkan/icd.d/nvidia_icd.json',
    'VK_DRIVER_FILES': '/etc/vulkan/icd.d/nvidia_icd.json',
    'OMNI_KIT_ACCEPT_EULA': 'YES', 'ACCEPT_EULA': 'Y',
}})

CHECKPOINT = '/mnt/cluster_storage/checkpoints/humanoid/checkpoint_pretrained.pt'

policy_handle = serve.run(
    PolicyServer.bind(checkpoint_path=CHECKPOINT),
    name='policy_server',
)

@ray.remote(num_gpus=1)
def evaluate_config(config_id, config, policy_handle, num_envs=20, num_steps=1500):
    from env import IsaacLabDirectEnv
    env = IsaacLabDirectEnv(task='Isaac-Humanoid-Direct-v0', num_envs=num_envs, device='cuda:0')
    obs = env.reset()
    episode_rewards, ep_rewards = [], np.zeros(num_envs)
    for step in range(num_steps):
        noisy_obs = obs + np.random.normal(0, config['obs_noise_std'], size=obs.shape).astype(np.float32) if config['obs_noise_std'] > 0 else obs
        result = policy_handle.predict.remote(noisy_obs).result()
        actions = result['action']
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
futures = [evaluate_config.remote(i, cfg, policy_handle) for i, cfg in enumerate(configs)]
results = ray.get(futures)
elapsed = time.time() - start
results.sort(key=lambda r: r['config_id'])
policy_stats = policy_handle.get_stats.remote().result()
with open('/mnt/cluster_storage/sweep_results_full.json', 'w') as f:
    json.dump({'results': results, 'obs_noise_levels': obs_noise_levels, 'action_noise_levels': action_noise_levels, 'elapsed': elapsed, 'total_episodes': sum(r['num_episodes'] for r in results), 'policy_server': policy_stats}, f, indent=2)
print(f'Done: {len(results)} configs, {elapsed:.0f}s')
print(f'Policy server: {policy_stats["total_calls"]} calls, avg {policy_stats["avg_latency_ms"]:.2f}ms')
serve.shutdown()
ray.shutdown()
