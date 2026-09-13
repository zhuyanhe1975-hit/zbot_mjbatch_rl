from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zbot_mjbatch_rl.ppo import TrainConfig, train


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--iterations", type=int, default=300)
  parser.add_argument(
    "--save-interval",
    type=int,
    default=0,
    help="save an intermediate checkpoint every N iterations; 0 only saves at the end",
  )
  parser.add_argument("--output", type=Path, default=Path("runs/zbot_walk.pt"))
  parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
  args = parser.parse_args()
  config = TrainConfig(
    num_envs=args.num_envs,
    iterations=args.iterations,
    save_interval=args.save_interval,
  )
  train(config, args.output, args.device)


if __name__ == "__main__":
  main()
