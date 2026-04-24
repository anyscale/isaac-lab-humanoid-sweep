"""
Isaac Lab environment wrapper using subprocess isolation.
Isaac Sim's Kit engine conflicts with Ray's event loop, so we run the sim
in a child process with a clean Python interpreter state.

Works with both image approaches:
  A) pip-installed Isaac Sim (your Containerfile) — isaaclab importable directly
  B) NVIDIA base image (Omar's Dockerfile) — isaaclab lives in Isaac Sim's own Python

Usage:
    env = IsaacLabDirectEnv(task="Isaac-Cartpole-Direct-v0", num_envs=64, device="cuda:0")
    obs = env.reset()
    obs, rewards, dones, infos = env.step(actions)
    env.close()
"""

import multiprocessing as mp
import os
import sys
import numpy as np


def _find_isaaclab_python():
    """
    Detect which Python has Isaac Lab available.
    Returns None if the current interpreter already has it,
    or a path to the correct python executable.
    """
    # Check if isaaclab is importable in the current env
    try:
        import isaaclab  # noqa
        return None  # current python works
    except ImportError:
        pass

    # Look for Isaac Sim's python.sh (Omar's Dockerfile stores the path)
    candidates = []

    # Omar's Dockerfile writes this file
    if os.path.exists("/etc/isaacsim_python_path"):
        with open("/etc/isaacsim_python_path") as f:
            candidates.append(f.read().strip())

    # Common locations
    candidates += [
        "/opt/isaac-sim/python.sh",
        "/isaac-sim/python.sh",
        "/workspace/isaaclab/_isaac_sim/python.sh",
    ]

    # Also check for isaaclab.sh
    isaaclab_sh_candidates = [
        "/workspace/isaaclab/isaaclab.sh",
        "/isaaclab/isaaclab.sh",
    ]

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    for path in isaaclab_sh_candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def _sim_process(task, num_envs, device, seed, cmd_pipe, result_pipe):
    """
    Runs in a child process with a clean event loop.
    Isaac Sim boots here, completely isolated from Ray.
    """
    import os
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_DRIVER_FILES", "/etc/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")

    import torch
    import gymnasium as gym

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    sim_app = app_launcher.app

    import isaaclab.envs  # noqa
    import isaaclab_tasks  # noqa

    # Resolve config dynamically for any task
    spec = gym.spec(task)
    entry = spec.kwargs.get("env_cfg_entry_point")
    if ":" in entry:
        mod_path, cls_name = entry.rsplit(":", 1)
    else:
        mod_path, cls_name = entry.rsplit(".", 1)

    import importlib
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed

    env = gym.make(task, cfg=cfg)
    unwrapped = env.unwrapped

    # Send env info back
    obs_space = unwrapped.single_observation_space
    act_space = unwrapped.single_action_space

    if isinstance(obs_space, gym.spaces.Dict):
        if "policy" in obs_space.spaces:
            obs_dim = obs_space["policy"].shape[0]
        else:
            obs_dim = sum(s.shape[0] for s in obs_space.spaces.values())
    else:
        obs_dim = obs_space.shape[0]

    act_dim = act_space.shape[0]
    result_pipe.send(("ready", obs_dim, act_dim))

    # Main loop — wait for commands
    while True:
        try:
            cmd = cmd_pipe.recv()
        except EOFError:
            break

        if cmd[0] == "reset":
            obs, info = env.reset()
            obs_np = _extract_obs(obs)
            result_pipe.send(("obs", obs_np))

        elif cmd[0] == "step":
            actions_np = cmd[1]
            actions_t = torch.from_numpy(actions_np).to(device)
            obs, rewards, terminated, truncated, infos = env.step(actions_t)
            dones = terminated | truncated
            obs_np = _extract_obs(obs)
            rew_np = _to_numpy(rewards)
            done_np = _to_numpy(dones)
            result_pipe.send(("step", obs_np, rew_np, done_np))

        elif cmd[0] == "close":
            env.close()
            sim_app.close()
            result_pipe.send(("closed",))
            break

    cmd_pipe.close()
    result_pipe.close()


def _sim_process_external(task, num_envs, device, seed,
                          cmd_pipe, result_pipe, python_path):
    """
    When Isaac Lab lives in a different Python (Omar's Dockerfile),
    we launch a subprocess script using that Python and bridge via pipes.
    """
    import subprocess
    import json
    import tempfile
    import struct

    # Write a self-contained script for the Isaac Lab Python to execute
    script = f'''
import os, sys, json, struct
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("VK_DRIVER_FILES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({{"headless": True, "enable_cameras": False}})
sim_app = app_launcher.app

import isaaclab.envs
import isaaclab_tasks
import importlib

task = "{task}"
num_envs = {num_envs}
device = "{device}"
seed = {seed}

spec = gym.spec(task)
entry = spec.kwargs.get("env_cfg_entry_point")
if ":" in entry:
    mod_path, cls_name = entry.rsplit(":", 1)
else:
    mod_path, cls_name = entry.rsplit(".", 1)

cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
cfg = cfg_cls()
cfg.scene.num_envs = num_envs
cfg.sim.device = device
cfg.seed = seed

env = gym.make(task, cfg=cfg)
unwrapped = env.unwrapped
obs_space = unwrapped.single_observation_space
act_space = unwrapped.single_action_space

if isinstance(obs_space, gym.spaces.Dict):
    if "policy" in obs_space.spaces:
        obs_dim = obs_space["policy"].shape[0]
    else:
        obs_dim = sum(s.shape[0] for s in obs_space.spaces.values())
else:
    obs_dim = obs_space.shape[0]
act_dim = act_space.shape[0]

def extract_obs(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"].detach().cpu().numpy()
        vals = list(obs.values())
        return torch.cat(vals, dim=-1).detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy()
    return obs

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def send_array(arr, f):
    data = arr.tobytes()
    header = json.dumps({{"shape": list(arr.shape), "dtype": str(arr.dtype)}}).encode()
    f.write(struct.pack("!I", len(header)))
    f.write(header)
    f.write(struct.pack("!I", len(data)))
    f.write(data)
    f.flush()

def recv_array(f):
    hlen = struct.unpack("!I", f.read(4))[0]
    header = json.loads(f.read(hlen).decode())
    dlen = struct.unpack("!I", f.read(4))[0]
    data = f.read(dlen)
    return np.frombuffer(data, dtype=header["dtype"]).reshape(header["shape"])

# Signal ready
sys.stdout.write(json.dumps({{"status": "ready", "obs_dim": obs_dim, "act_dim": act_dim}}) + "\\n")
sys.stdout.flush()

# Command loop via stdin/stdout
for line in sys.stdin:
    cmd = json.loads(line.strip())
    if cmd["type"] == "reset":
        obs, info = env.reset()
        obs_np = extract_obs(obs)
        sys.stdout.buffer.write(b"OBS")
        send_array(obs_np, sys.stdout.buffer)
    elif cmd["type"] == "step":
        actions_np = recv_array(sys.stdin.buffer)
        actions_t = torch.from_numpy(actions_np).to(device)
        obs, rewards, terminated, truncated, infos = env.step(actions_t)
        dones = terminated | truncated
        obs_np = extract_obs(obs)
        rew_np = to_numpy(rewards)
        done_np = to_numpy(dones).astype(np.bool_)
        sys.stdout.buffer.write(b"STP")
        send_array(obs_np, sys.stdout.buffer)
        send_array(rew_np, sys.stdout.buffer)
        send_array(done_np, sys.stdout.buffer)
    elif cmd["type"] == "close":
        env.close()
        sim_app.close()
        sys.stdout.write(json.dumps({{"status": "closed"}}) + "\\n")
        sys.stdout.flush()
        break
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # Determine how to invoke — python.sh or isaaclab.sh -p
        if python_path.endswith("python.sh"):
            cmd_line = [python_path, script_path]
        elif python_path.endswith("isaaclab.sh"):
            cmd_line = [python_path, "-p", script_path]
        else:
            cmd_line = [python_path, script_path]

        proc = subprocess.Popen(
            cmd_line,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for ready signal
        ready_line = proc.stdout.readline().decode()
        ready = json.loads(ready_line)
        assert ready["status"] == "ready"
        result_pipe.send(("ready", ready["obs_dim"], ready["act_dim"]))

        while True:
            try:
                cmd = cmd_pipe.recv()
            except EOFError:
                break

            if cmd[0] == "reset":
                proc.stdin.write(json.dumps({"type": "reset"}).encode() + b"\n")
                proc.stdin.flush()
                tag = proc.stdout.read(3)
                obs_np = _recv_array_from_pipe(proc.stdout)
                result_pipe.send(("obs", obs_np))

            elif cmd[0] == "step":
                proc.stdin.write(json.dumps({"type": "step"}).encode() + b"\n")
                proc.stdin.flush()
                _send_array_to_pipe(cmd[1], proc.stdin)
                tag = proc.stdout.read(3)
                obs_np = _recv_array_from_pipe(proc.stdout)
                rew_np = _recv_array_from_pipe(proc.stdout)
                done_np = _recv_array_from_pipe(proc.stdout)
                result_pipe.send(("step", obs_np, rew_np, done_np))

            elif cmd[0] == "close":
                proc.stdin.write(json.dumps({"type": "close"}).encode() + b"\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
                result_pipe.send(("closed",))
                break

    finally:
        os.unlink(script_path)

    cmd_pipe.close()
    result_pipe.close()


def _send_array_to_pipe(arr, pipe):
    import json
    import struct
    data = arr.tobytes()
    header = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}).encode()
    pipe.write(struct.pack("!I", len(header)))
    pipe.write(header)
    pipe.write(struct.pack("!I", len(data)))
    pipe.write(data)
    pipe.flush()


def _recv_array_from_pipe(pipe):
    import json
    import struct
    hlen = struct.unpack("!I", pipe.read(4))[0]
    header = json.loads(pipe.read(hlen).decode())
    dlen = struct.unpack("!I", pipe.read(4))[0]
    data = pipe.read(dlen)
    return np.frombuffer(data, dtype=header["dtype"]).reshape(header["shape"]).copy()


def _extract_obs(obs):
    import torch
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"].detach().cpu().numpy()
        vals = list(obs.values())
        return torch.cat(vals, dim=-1).detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy()
    return obs


def _to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class IsaacLabDirectEnv:
    """
    Wraps Isaac Lab in a child process for event loop isolation.

    Automatically detects whether Isaac Lab is available in the current Python
    (pip install approach) or needs a separate Python runtime (NVIDIA base image).

    Communication via multiprocessing pipes (fast, no serialization overhead for numpy).
    """

    def __init__(
        self,
        task: str = "Isaac-Cartpole-Direct-v0",
        num_envs: int = 1024,
        device: str = "cuda:0",
        seed: int = 42,
        headless: bool = True,
    ):
        self.task = task
        self.num_envs = num_envs
        self.device = device

        # Detect which Python has Isaac Lab
        self._external_python = _find_isaaclab_python()

        # Create pipes for communication
        self._cmd_parent, self._cmd_child = mp.Pipe()
        self._result_parent, self._result_child = mp.Pipe()

        if self._external_python is None:
            # Isaac Lab is in the current Python — use multiprocessing directly
            ctx = mp.get_context("spawn")
            self._proc = ctx.Process(
                target=_sim_process,
                args=(task, num_envs, device, seed,
                      self._cmd_child, self._result_child),
                daemon=True,
            )
            self._proc.start()
            self._external_proc = None
        else:
            # Isaac Lab is in a different Python — bridge via subprocess
            print(f"[IsaacLabDirectEnv] Using external Python: {self._external_python}")
            ctx = mp.get_context("spawn")
            self._proc = ctx.Process(
                target=_sim_process_external,
                args=(task, num_envs, device, seed,
                      self._cmd_child, self._result_child,
                      self._external_python),
                daemon=True,
            )
            self._proc.start()
            self._external_proc = None

        # Wait for sim to boot
        msg = self._result_parent.recv()
        assert msg[0] == "ready", f"Sim process failed to start: {msg}"
        self._obs_dim = msg[1]
        self._act_dim = msg[2]

        print(f"[IsaacLabDirectEnv] Ready: {task}, obs={self._obs_dim}, act={self._act_dim}, "
              f"envs={num_envs}, pid={self._proc.pid}"
              + (f", python={self._external_python}" if self._external_python else ""))

    def reset(self, **kwargs) -> np.ndarray:
        self._cmd_parent.send(("reset",))
        msg = self._result_parent.recv()
        assert msg[0] == "obs"
        return msg[1]

    def step(self, actions: np.ndarray):
        self._cmd_parent.send(("step", actions))
        msg = self._result_parent.recv()
        assert msg[0] == "step"
        return msg[1], msg[2], msg[3], {}

    def close(self):
        try:
            self._cmd_parent.send(("close",))
            # Isaac Sim's shutdown can hang; don't block forever waiting for the ack.
            if self._result_parent.poll(5):
                self._result_parent.recv()
        except (BrokenPipeError, EOFError):
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.kill()

    @property
    def num_obs(self):
        return self._obs_dim

    @property
    def num_actions(self):
        return self._act_dim
