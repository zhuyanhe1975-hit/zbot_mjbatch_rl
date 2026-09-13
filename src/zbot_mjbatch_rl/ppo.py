from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from zbot_mjbatch_rl.env import ACT_DIM, OBS_DIM, ZBotEnv


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


def train(config: TrainConfig, output: Path, device: str = "cpu") -> ActorCritic:
  torch.manual_seed(config.seed)
  np.random.seed(config.seed)
  env = ZBotEnv(config.num_envs, seed=config.seed)
  net = ActorCritic().to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
  obs = env.obs()

  for iteration in range(config.iterations):
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
        next_obs, reward, done, _ = env.step(action.cpu().numpy())
        for name, value_np in (
          ("obs", obs),
          ("act", action.cpu().numpy()),
          ("logp", log_density(noise, net.log_std).cpu().numpy()),
          ("value", value.cpu().numpy()),
          ("reward", reward),
          ("alive", (~done).astype(np.float32)),
        ):
          arrays[name][t] = value_np
        ids = np.flatnonzero(done).astype(np.int32)
        env.reset(ids)
        obs = env.obs() if ids.size else next_obs
      last_value = net(torch.as_tensor(obs, device=device))[1]

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
        loss = (
          -surrogate.mean()
          + 0.5 * (value - flat_returns[indices]).square().mean()
          - config.entropy_coef * net.log_std.sum()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

    if iteration == 0 or (iteration + 1) % 10 == 0:
      print(
        f"{iteration + 1:4d}/{config.iterations}  "
        f"reward={arrays['reward'].mean():6.3f}  "
        f"height={env.qpos[:, 2].mean():.3f}"
      )

    if config.save_interval > 0 and (iteration + 1) % config.save_interval == 0:
      save_checkpoint(net, config, output)

  save_checkpoint(net, config, output)
  return net
