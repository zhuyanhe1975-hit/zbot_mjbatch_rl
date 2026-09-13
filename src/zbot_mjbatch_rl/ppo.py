from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from zbot_mjbatch_rl.env import ACT_DIM, OBS_DIM, REWARD_WEIGHTS, ZBotEnv


class ActorCritic(nn.Module):
  def __init__(self):
    super().__init__()
    self.actor = self._mlp(ACT_DIM)
    self.critic = self._mlp(1)
    self.log_std = nn.Parameter(torch.full((ACT_DIM,), np.log(0.5), dtype=torch.float32))

  @staticmethod
  def _mlp(output: int) -> nn.Sequential:
    return nn.Sequential(
      nn.Linear(OBS_DIM, 128),
      nn.ELU(),
      nn.Linear(128, 128),
      nn.ELU(),
      nn.Linear(128, output),
    )

  def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.actor(obs), self.critic(obs).squeeze(-1)


def log_density(noise: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
  return -0.5 * noise.square().sum(-1) - log_std.sum() - 0.5 * ACT_DIM * np.log(2.0 * np.pi)


@dataclass
class TrainConfig:
  num_envs: int = 512
  iterations: int = 300
  horizon: int = 24
  epochs: int = 5
  minibatches: int = 4
  learning_rate: float = 1e-3
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip: float = 0.2
  entropy_coef: float = 0.005
  seed: int = 0
  save_interval: int = 0


def save_checkpoint(net: ActorCritic, config: TrainConfig, output: Path) -> None:
  """Atomically publish a checkpoint so a watcher never reads a partial file."""
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix(f"{output.suffix}.tmp")
  torch.save({"model": net.state_dict(), "config": vars(config)}, temporary)
  temporary.replace(output)


def format_duration(seconds: float) -> str:
  seconds = max(0, int(seconds))
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_training_status(
  iteration: int,
  config: TrainConfig,
  *,
  fps: float,
  collection_time: float,
  learning_time: float,
  value_loss: float,
  surrogate_loss: float,
  mean_noise_std: float,
  mean_rollout_reward: float,
  reward_terms: dict[str, float],
  mean_episode_reward: float | None,
  mean_episode_length: float | None,
  mean_height: float,
  total_steps: int,
  elapsed: float,
) -> None:
  width = 80
  completed = iteration + 1
  eta = elapsed / completed * (config.iterations - completed)
  episode_reward = "n/a" if mean_episode_reward is None else f"{mean_episode_reward:.2f}"
  episode_length = "n/a" if mean_episode_length is None else f"{mean_episode_length:.2f}"
  title = f" Learning iteration {completed}/{config.iterations} "
  print(f"\n{title.center(width, '#')}")
  print(
    f"{'Computation:':>35} {fps:,.0f} steps/s "
    f"(collection: {collection_time:.3f}s, learning: {learning_time:.3f}s)"
  )
  print(f"{'Value function loss:':>35} {value_loss:.4f}")
  print(f"{'Surrogate loss:':>35} {surrogate_loss:.4f}")
  print(f"{'Mean action noise std:':>35} {mean_noise_std:.3f}")
  print(f"{'Mean rollout reward/step:':>35} {mean_rollout_reward:.3f}")
  for name, contribution in reward_terms.items():
    print(f"{f'Mean reward/{name}:':>35} {contribution:+.4f}")
  print(f"{'Mean episode reward:':>35} {episode_reward}")
  print(f"{'Mean episode length:':>35} {episode_length}")
  print(f"{'Mean base height:':>35} {mean_height:.3f} m")
  print("-" * width)
  print(f"{'Total timesteps:':>35} {total_steps:,}")
  print(f"{'Iteration time:':>35} {collection_time + learning_time:.2f}s")
  print(f"{'Total time:':>35} {format_duration(elapsed)}")
  print(f"{'ETA:':>35} {format_duration(eta)}")


def train(config: TrainConfig, output: Path, device: str = "cpu") -> ActorCritic:
  torch.manual_seed(config.seed)
  np.random.seed(config.seed)
  env = ZBotEnv(config.num_envs, seed=config.seed)
  net = ActorCritic().to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
  obs = env.obs()
  episode_rewards = np.zeros(config.num_envs, np.float32)
  episode_lengths = np.zeros(config.num_envs, np.int32)
  reward_history: deque[float] = deque(maxlen=100)
  length_history: deque[int] = deque(maxlen=100)
  training_start = time.perf_counter()
  total_steps = 0

  for iteration in range(config.iterations):
    iteration_start = time.perf_counter()
    reward_term_sums = {name: 0.0 for name in REWARD_WEIGHTS}
    arrays = {
      "obs": np.empty((config.horizon, config.num_envs, OBS_DIM), np.float32),
      "act": np.empty((config.horizon, config.num_envs, ACT_DIM), np.float32),
      "logp": np.empty((config.horizon, config.num_envs), np.float32),
      "value": np.empty((config.horizon, config.num_envs), np.float32),
      "reward": np.empty((config.horizon, config.num_envs), np.float32),
      "alive": np.empty((config.horizon, config.num_envs), np.float32),
    }

    with torch.no_grad():
      for t in range(config.horizon):
        obs_t = torch.as_tensor(obs, device=device)
        mean, value = net(obs_t)
        noise = torch.randn_like(mean)
        action = mean + net.log_std.exp() * noise
        next_obs, reward, done, terms = env.step(action.cpu().numpy())
        for name, value_np in terms.items():
          reward_term_sums[name] += REWARD_WEIGHTS[name] * float(np.sum(value_np))
        for name, value_np in (
          ("obs", obs),
          ("act", action.cpu().numpy()),
          ("logp", log_density(noise, net.log_std).cpu().numpy()),
          ("value", value.cpu().numpy()),
          ("reward", reward),
          ("alive", (~done).astype(np.float32)),
        ):
          arrays[name][t] = value_np
        episode_rewards += reward
        episode_lengths += 1
        ids = np.flatnonzero(done).astype(np.int32)
        reward_history.extend(episode_rewards[ids].tolist())
        length_history.extend(episode_lengths[ids].tolist())
        episode_rewards[ids] = 0.0
        episode_lengths[ids] = 0
        env.reset(ids)
        obs = env.obs() if ids.size else next_obs
      last_value = net(torch.as_tensor(obs, device=device))[1]
    collection_time = time.perf_counter() - iteration_start

    learning_start = time.perf_counter()
    batch = {name: torch.as_tensor(value, device=device) for name, value in arrays.items()}
    advantage = torch.zeros_like(batch["reward"])
    carry = torch.zeros(config.num_envs, device=device)
    values = torch.cat((batch["value"], last_value[None]), dim=0)
    for t in reversed(range(config.horizon)):
      delta = batch["reward"][t] + config.gamma * batch["alive"][t] * values[t + 1] - values[t]
      carry = delta + config.gamma * config.gae_lambda * batch["alive"][t] * carry
      advantage[t] = carry
    returns = advantage + batch["value"]

    flat_obs = batch["obs"].flatten(0, 1)
    flat_act = batch["act"].flatten(0, 1)
    flat_old_logp = batch["logp"].flatten()
    flat_advantage = advantage.flatten()
    flat_returns = returns.flatten()
    flat_advantage = (flat_advantage - flat_advantage.mean()) / (flat_advantage.std() + 1e-8)

    value_losses = []
    surrogate_losses = []
    for _ in range(config.epochs):
      for indices in torch.randperm(flat_obs.shape[0], device=device).chunk(config.minibatches):
        mean, value = net(flat_obs[indices])
        noise = (flat_act[indices] - mean) / net.log_std.exp()
        logp = log_density(noise, net.log_std)
        ratio = (logp - flat_old_logp[indices]).exp()
        surrogate = torch.min(
          ratio * flat_advantage[indices],
          ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * flat_advantage[indices],
        )
        value_loss = (value - flat_returns[indices]).square().mean()
        surrogate_loss = -surrogate.mean()
        loss = (
          surrogate_loss
          + 0.5 * value_loss
          - config.entropy_coef * net.log_std.sum()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        value_losses.append(value_loss.item())
        surrogate_losses.append(surrogate_loss.item())

    learning_time = time.perf_counter() - learning_start
    iteration_time = time.perf_counter() - iteration_start
    steps_this_iteration = config.horizon * config.num_envs
    total_steps += steps_this_iteration
    print_training_status(
      iteration,
      config,
      fps=steps_this_iteration / iteration_time,
      collection_time=collection_time,
      learning_time=learning_time,
      value_loss=float(np.mean(value_losses)),
      surrogate_loss=float(np.mean(surrogate_losses)),
      mean_noise_std=float(net.log_std.exp().mean().item()),
      mean_rollout_reward=float(arrays["reward"].mean()),
      reward_terms={
        name: total / steps_this_iteration for name, total in reward_term_sums.items()
      },
      mean_episode_reward=float(np.mean(reward_history)) if reward_history else None,
      mean_episode_length=float(np.mean(length_history)) if length_history else None,
      mean_height=float(env.qpos[:, 2].mean()),
      total_steps=total_steps,
      elapsed=time.perf_counter() - training_start,
    )

    if config.save_interval > 0 and (iteration + 1) % config.save_interval == 0:
      save_checkpoint(net, config, output)

  save_checkpoint(net, config, output)
  return net
