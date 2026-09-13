from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from mjbatch import Batch

PHYSICS_DT = 0.004
DECIMATION = 5
CTRL_DT = PHYSICS_DT * DECIMATION
EPISODE_STEPS = 500
ACT_DIM = 6
STAND = np.array([0.312, 0.837, -2.02, 2.02, -0.837, -0.312])
# MuJoCo tree order is joint3,2,1,4,5,6; the policy and actuators use joint1..6.
JOINT_QPOS = np.array([9, 8, 7, 10, 11, 12])
JOINT_QVEL = np.array([8, 7, 6, 9, 10, 11])
OBS_DIM = 24
BASE_HEIGHT = 0.254474
JOINT_SPEED = 2.0
TARGET_SPEED = 0.2
TOUCH_THRESHOLD = 0.1
IMPACT_FORCE_SCALE = 100.0
IMPACT_PENALTY = 0.05
SLIP_SPEED_SCALE = 0.25
SLIP_PENALTY = 0.5
REWARD_WEIGHTS = {
  "forward": 1.0,
  "survival": 0.2,
  "single_support": 0.1,
  "heading": -0.5,
  "lateral": -0.2,
  "similar_to_default": -0.1,
  "rate": -0.01,
  "torque": -0.0002,
  "foot_impact": -IMPACT_PENALTY,
  "foot_slip": -SLIP_PENALTY,
  "termination": -20.0,
}


def model_path() -> Path:
  return Path(__file__).parents[2] / "assets" / "zbot_6dof.xml"


def build_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(model_path()))


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vectors by inverse unit quaternions in MuJoCo wxyz order."""
  xyz = q[:, 1:]
  uv = np.cross(xyz, v)
  uuv = np.cross(xyz, uv)
  return v - 2.0 * (q[:, :1] * uv - uuv)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  xyz = q[:, 1:]
  uv = np.cross(xyz, v)
  uuv = np.cross(xyz, uv)
  return v + 2.0 * (q[:, :1] * uv + uuv)


class ZBotEnv:
  """Minimal batched forward-walking environment for the six-joint ZBot proxy."""

  def __init__(self, num_envs: int, num_threads: int = 0, seed: int = 0):
    self.model = build_model()
    self.batch = Batch(self.model, num_envs, num_threads=num_threads, forward=True)
    self.num_envs = num_envs
    self.rng = np.random.default_rng(seed)

    self.qpos = self.batch.bind("qpos")
    self.qvel = self.batch.bind("qvel")
    self.ctrl = self.batch.bind("ctrl")
    self.gyro = self.batch.sensor("gyro")
    self.velocity = self.batch.sensor("velocity")
    self.left_touch = self.batch.sensor("left_touch")[:, 0]
    self.right_touch = self.batch.sensor("right_touch")[:, 0]
    self.left_foot_velocity = self.batch.sensor("left_foot_velocity")
    self.right_foot_velocity = self.batch.sensor("right_foot_velocity")
    self.torque = self.batch.bind("actuator_force")

    self.steps = np.zeros(num_envs, np.int64)
    self.action = np.zeros((num_envs, ACT_DIM), np.float32)
    self.previous_action = np.zeros_like(self.action)
    self.target_delta = np.zeros_like(self.action)
    self.foot_contact = np.zeros((num_envs, 2), dtype=bool)
    self.reset(np.arange(num_envs, dtype=np.int32))

  def reset(self, ids: np.ndarray) -> None:
    ids = np.asarray(ids, dtype=np.int32)
    if ids.size == 0:
      return
    self.batch.reset(ids, keyframe=0)
    self.qpos[ids[:, None], JOINT_QPOS] = STAND + self.rng.uniform(-0.04, 0.04, (ids.size, ACT_DIM))
    self.qvel[ids, :12] = self.rng.uniform(-0.03, 0.03, (ids.size, 12))
    self.batch.forward(ids)
    self.steps[ids] = 0
    self.action[ids] = 0.0
    self.previous_action[ids] = 0.0
    self.target_delta[ids] = 0.0
    self.foot_contact[ids, 0] = self.left_touch[ids] > TOUCH_THRESHOLD
    self.foot_contact[ids, 1] = self.right_touch[ids] > TOUCH_THRESHOLD

  def orientation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gravity_world = np.zeros((self.num_envs, 3))
    gravity_world[:, 2] = -1.0
    gravity_body = quat_rotate_inverse(self.qpos[:, 3:7], gravity_world)
    forward_body = np.cross(gravity_body, np.tile([0.0, 0.0, 1.0], (self.num_envs, 1)))
    forward_body /= np.maximum(np.linalg.norm(forward_body, axis=1, keepdims=True), 1e-8)
    forward_world = quat_rotate(self.qpos[:, 3:7], forward_body)
    return gravity_body, forward_body, forward_world

  def forward_speed(self) -> np.ndarray:
    return np.sum(self.velocity * self.orientation()[1], axis=1)

  def obs(self) -> np.ndarray:
    gravity_body, _, forward_world = self.orientation()
    angular_world = quat_rotate(self.qpos[:, 3:7], self.gyro)
    return np.concatenate(
      (
        angular_world[:, 2:3],
        gravity_body,
        -forward_world[:, 1:2],
        self.qpos[:, JOINT_QPOS] - STAND,
        0.1 * self.qvel[:, JOINT_QVEL],
        self.action,
        np.full((self.num_envs, 1), JOINT_SPEED),
      ),
      axis=1,
      dtype=np.float32,
    )

  def step(
    self, action: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    self.previous_action[:] = self.action
    self.action[:] = np.tanh(action)
    self.target_delta += np.pi * JOINT_SPEED * self.action * CTRL_DT
    np.clip(self.target_delta, -np.pi, np.pi, out=self.target_delta)
    self.ctrl[:] = np.clip(STAND + self.target_delta, -np.pi, np.pi)
    self.batch.step(nstep=DECIMATION)
    self.steps += 1

    _, _, forward_world = self.orientation()
    speed = self.forward_speed()
    contact_force = np.column_stack((self.left_touch, self.right_touch))
    contact = contact_force > TOUCH_THRESHOLD
    touchdown = contact & ~self.foot_contact
    scaled_impact = np.minimum(contact_force / IMPACT_FORCE_SCALE, 3.0) * touchdown
    impact = np.sum(np.square(scaled_impact), axis=1)
    horizontal_foot_speed = np.column_stack(
      (
        np.linalg.norm(self.left_foot_velocity[:, :2], axis=1),
        np.linalg.norm(self.right_foot_velocity[:, :2], axis=1),
      )
    )
    scaled_slip = np.minimum(horizontal_foot_speed / SLIP_SPEED_SCALE, 3.0) * contact
    slip = np.sum(np.square(scaled_slip), axis=1)
    self.foot_contact[:] = contact
    left_contact, right_contact = contact[:, 0], contact[:, 1]
    fallen = (self.qpos[:, 2] < 0.20) | (np.abs(self.qpos[:, 1]) > 0.5)

    terms = {
      "forward": np.clip(speed / TARGET_SPEED, -1.0, 1.0),
      "survival": np.ones(self.num_envs),
      "heading": np.abs(forward_world[:, 1]),
      "lateral": np.abs(self.qpos[:, 1]),
      "single_support": np.logical_xor(left_contact, right_contact).astype(float),
      "similar_to_default": np.sum(np.abs(self.qpos[:, JOINT_QPOS] - STAND), axis=1),
      "rate": np.sum(np.square(self.action - self.previous_action), axis=1),
      "torque": np.sum(np.square(self.torque), axis=1),
      "foot_impact": impact,
      "foot_slip": slip,
      "termination": fallen.astype(float),
    }
    reward = sum(REWARD_WEIGHTS[name] * value for name, value in terms.items()).astype(
      np.float32
    )
    timeout = self.steps >= EPISODE_STEPS
    done = fallen | timeout
    return self.obs(), reward, done, terms
