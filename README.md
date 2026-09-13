# zbot-mjbatch-rl

用 `mjbatch` 验证 ZBot 6-DoF 双足强化学习路线的最小项目。

本项目只实现一个任务：让 6-DoF ZBot 向前行走。机器人来自
[`zbot_rl_student`](https://github.com/zhuyanhe1975-hit/zbot_rl_student) 的
`zbot_6s_new.usda`，固定在 revision
`aa340d41b803850a5b21e6ec6b9f76012f9580c0`。运行时不依赖 Isaac Lab 或 RSL-RL。

## 边界

- 保留 USD 的 12 个刚体模块、六个倾斜转轴及真实串联拓扑。
- 保留各刚体的网格、质量、质心、主惯量和惯性主轴。
- 保留参考项目的站立关节角、`kp=50`、`kd=5` 和六维动作顺序。
- observation、reward 和 PPO 都刻意保持最小。
- 落足冲击奖励只惩罚脚底从无接触到接触瞬间的峰值力，不惩罚正常站立承重。
- 防滑奖励只惩罚支撑脚接触地面时的水平速度，不限制腾空脚的摆动。
- 不包含全向速度、domain randomization、curriculum、teacher-student 或真机部署。

USD 中以 `foot_0` 为 articulation root；MJCF 以中间的 `base` 为树根，因此左支链的关节被
等价反向，并对初始根位姿做了坐标变换。转换后的两只脚与原 USD 站立姿态对齐。

PhysX 和 MuJoCo 的接触、关节驱动与求解器并不相同，因此该模型用于验证训练路线，不能证明
策略可直接部署到真机。

## 使用

```bash
uv sync --dev
uv run pytest -q
uv run python scripts/train.py --num-envs 512 --iterations 500
uv run python scripts/evaluate.py runs/zbot_walk_poc.pt
uv run python scripts/play.py runs/zbot_walk_poc.pt
```

也可以使用项目根目录下的脚本训练、播放，并在训练期间监控定期保存的策略：

```bash
./train.sh
./run.sh runs/zbot_walk.pt
./watch.sh
```

也可以让训练脚本自动启动监控画面：

```bash
./train.sh --watch
```

`train.sh` 默认每 100 轮原子保存一次 `runs/zbot_walk.pt`；`watch.sh` 只启动一次
viewer，检测到更新后会在原窗口热加载网络权重。可通过 `NUM_ENVS`、`ITERATIONS`、`SAVE_INTERVAL`、`CHECKPOINT`
和 `POLL_SECONDS` 环境变量覆盖默认值。

在 macOS 上，`play.py` 会自动使用当前虚拟环境的 `mjpython` 重新启动 viewer，命令无需改变。

快速检查完整训练闭环：

```bash
uv run python scripts/train.py --num-envs 32 --iterations 2 --output /tmp/zbot_smoke.pt
```

## 验证标准

1. MJCF 能稳定加载并完成批量 stepping。
2. partial reset 不影响未选择的环境。
3. PPO 能完成 rollout、GAE、更新和 checkpoint 保存。
4. 长训练后，确定性策略产生稳定的正向速度，且不会通过频繁跌倒获取奖励。

需要从固定的上游 USD 重新生成 OBJ 网格时：

```bash
uv run python scripts/import_zbot_model.py
```

## 已验证结果

在 Apple Silicon 上使用 256 个环境训练 500 轮的短实验中：

- 未训练的零动作策略平均前向速度约为 `0.0005 m/s`；
- 训练策略平均前向速度约为 `0.093 m/s`；
- 当前策略仍有较多提前终止，checkpoint 只作为路线 PoC，不代表任务已经收敛。

这说明策略已经从真实 ZBot 拓扑的批量 MuJoCo 仿真中学到前向运动，但仍需继续调整奖励和训练。
