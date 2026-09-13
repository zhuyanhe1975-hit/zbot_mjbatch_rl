from zbot_mjbatch_rl.ppo import ActorCritic, TrainConfig, train


def test_one_training_iteration(tmp_path):
  output = tmp_path / "model.pt"
  net = train(
    TrainConfig(num_envs=8, iterations=1, horizon=4, epochs=1, minibatches=2),
    output,
  )
  assert isinstance(net, ActorCritic)
  assert output.exists()
