from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from zbot_mjbatch_rl.env import (
  DECIMATION,
  JOINT_QPOS,
  JOINT_QVEL,
  JOINT_SPEED,
  STAND,
  build_model,
  quat_rotate,
  quat_rotate_inverse,
)
from zbot_mjbatch_rl.ppo import ActorCritic


def ensure_mjpython() -> None:
  """MuJoCo's passive viewer must run through mjpython on macOS."""
  if sys.platform != "darwin" or os.environ.get("ZBOT_USING_MJPYTHON") == "1":
    return
  mjpython = Path(sys.executable).with_name("mjpython")
  if not mjpython.exists():
    raise RuntimeError(f"mjpython was not found next to {sys.executable}")
  os.environ["ZBOT_USING_MJPYTHON"] = "1"
  os.execv(mjpython, [str(mjpython), *sys.argv])


def observation(data, action):
  gravity = quat_rotate_inverse(data.qpos[3:7][None], np.array([[0.0, 0.0, -1.0]]))
  gyro = data.sensor("gyro").data.copy()
  forward_body = np.cross(gravity, np.array([[0.0, 0.0, 1.0]]))
  forward_body /= np.maximum(np.linalg.norm(forward_body, axis=1, keepdims=True), 1e-8)
  forward_world = quat_rotate(data.qpos[3:7][None], forward_body)
  angular_world = quat_rotate(data.qpos[3:7][None], gyro[None])
  return np.concatenate(
    [
      angular_world[0, 2:3],
      gravity[0],
      -forward_world[0, 1:2],
      data.qpos[JOINT_QPOS] - STAND,
      0.1 * data.qvel[JOINT_QVEL],
      action,
      [JOINT_SPEED],
    ],
    dtype=np.float32,
  )


def checkpoint_signature(path: Path) -> tuple[int, int]:
  stat = path.stat()
  return stat.st_mtime_ns, stat.st_size


def load_policy(net: ActorCritic, checkpoint: Path) -> None:
  state = torch.load(checkpoint, map_location="cpu", weights_only=True)
  net.load_state_dict(state["model"])
  net.eval()


def reset_robot(model, data, action: np.ndarray, target_delta: np.ndarray) -> None:
  action.fill(0.0)
  target_delta.fill(0.0)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)


def has_fallen(data) -> bool:
  return bool(data.qpos[2] < 0.20 or abs(data.qpos[1]) > 0.5)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 runs until closed")
  parser.add_argument(
    "--watch",
    action="store_true",
    help="hot-reload the policy when the checkpoint changes without closing the viewer",
  )
  parser.add_argument("--poll-seconds", type=float, default=2.0)
  args = parser.parse_args()
  ensure_mjpython()
  model, data = build_model(), None
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  net = ActorCritic()
  load_policy(net, args.checkpoint)
  signature = checkpoint_signature(args.checkpoint)
  next_checkpoint_check = time.perf_counter() + args.poll_seconds
  action = np.zeros(6, np.float32)
  target_delta = np.zeros(6, np.float32)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    play_start = time.perf_counter()
    while viewer.is_running():
      start = time.perf_counter()
      if args.watch and start >= next_checkpoint_check:
        next_checkpoint_check = start + args.poll_seconds
        try:
          new_signature = checkpoint_signature(args.checkpoint)
          if new_signature != signature:
            load_policy(net, args.checkpoint)
            signature = new_signature
            reset_robot(model, data, action, target_delta)
            print(f"Reloaded policy from {args.checkpoint}")
        except (OSError, RuntimeError) as error:
          print(f"Checkpoint reload failed; will retry: {error}", file=sys.stderr)
      obs = observation(data, action)
      with torch.no_grad():
        action = np.tanh(net(torch.from_numpy(obs)[None])[0][0].numpy())
      target_delta += np.pi * JOINT_SPEED * action * model.opt.timestep * DECIMATION
      np.clip(target_delta, -np.pi, np.pi, out=target_delta)
      data.ctrl[:] = np.clip(STAND + target_delta, -np.pi, np.pi)
      for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
      if has_fallen(data):
        reset_robot(model, data, action, target_delta)
        print("Robot fell; reset to standing pose")
      viewer.sync()
      if args.duration > 0.0 and time.perf_counter() - play_start >= args.duration:
        break
      time.sleep(max(0.0, model.opt.timestep * DECIMATION - (time.perf_counter() - start)))


if __name__ == "__main__":
  main()
