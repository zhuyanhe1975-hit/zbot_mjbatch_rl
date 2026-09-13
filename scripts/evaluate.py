from __future__ import annotations

import argparse

import numpy as np
import torch

from zbot_mjbatch_rl import ActorCritic, ZBotEnv
from zbot_mjbatch_rl.env import EPISODE_STEPS


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  args = parser.parse_args()

  net = ActorCritic()
  net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
  net.eval()
  env = ZBotEnv(args.num_envs, seed=123)
  obs = env.obs()
  speeds, rewards = [], []
  falls = 0

  for _ in range(args.steps):
    with torch.no_grad():
      action = net(torch.from_numpy(obs))[0].numpy()
    obs, reward, done, _ = env.step(action)
    speeds.append(env.forward_speed().copy())
    rewards.append(reward)
    falls += int(np.sum(done & (env.steps < EPISODE_STEPS)))
    ids = np.flatnonzero(done).astype(np.int32)
    env.reset(ids)
    if ids.size:
      obs = env.obs()

  speed = np.asarray(speeds)
  print(f"mean forward speed: {speed.mean():.3f} m/s")
  print(f"mean reward:        {np.mean(rewards):.3f}")
  print(f"early falls:        {falls}")
  print(f"evaluated steps:    {args.steps * args.num_envs}")


if __name__ == "__main__":
  main()
