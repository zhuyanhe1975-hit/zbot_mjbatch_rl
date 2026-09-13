import mujoco
import numpy as np

from zbot_mjbatch_rl.env import ACT_DIM, OBS_DIM, ZBotEnv, build_model


def test_model_shape():
  model = build_model()
  assert model.nu == ACT_DIM
  assert model.nq == 7 + ACT_DIM
  assert model.nv == 6 + ACT_DIM
  assert model.nbody == 13  # world + the 12 rigid modules from zbot_6s_new.usda
  assert model.nmesh == 12
  np.testing.assert_allclose(model.body_mass[1:], 0.25042)
  assert np.all(model.mesh_vertnum > 1000)
  assert [model.actuator(i).name for i in range(model.nu)] == [
    f"joint{i}_ctrl" for i in range(1, 7)
  ]
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  np.testing.assert_allclose(data.body("foot_0").xpos, [0.0, -0.06, 0.0], atol=2e-5)
  np.testing.assert_allclose(data.body("foot_1").xpos, [0.000053, 0.060017, 0.053035], atol=2e-5)


def test_batch_step_and_partial_reset():
  env = ZBotEnv(8, num_threads=2, seed=0)
  before = env.qpos.copy()
  obs, reward, done, terms = env.step(np.zeros((8, ACT_DIM), np.float32))
  assert obs.shape == (8, OBS_DIM)
  assert reward.shape == done.shape == (8,)
  assert np.isfinite(obs).all() and np.isfinite(reward).all()
  assert set(terms) == {
    "forward",
    "heading",
    "lateral",
    "single_support",
    "posture",
    "rate",
    "torque",
    "foot_impact",
    "foot_slip",
  }
  assert np.all(terms["foot_impact"] >= 0.0)
  assert np.all(terms["foot_slip"] >= 0.0)
  assert not np.array_equal(before, env.qpos)
  untouched = env.qpos[1].copy()
  env.reset(np.array([0], np.int32))
  np.testing.assert_array_equal(env.qpos[1], untouched)
