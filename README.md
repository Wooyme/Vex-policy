# vex-policy

独立的 Unitree G1 ONNX policy 推理服务。控制命令只从 MQTT 读取，运行时可在多个已预加载 policy 之间安全切换，并发布真实机器人状态供 Vex 控制面板显示。

本项目从 Holosoma 提交 `f5445d1b56a5bad39da17ce468df460477a3a1d5` 的 inference 包迁移而来，内部命名空间、配置、输入和运行时已完全独立；原项目不再是运行或安装依赖。许可与来源见 `LICENSE`、`NOTICE`。

## 环境

`far-unitree-sdk==0.1.4` 不提供 CPython 3.13 wheel，因此项目固定使用 uv 管理的 Python 3.11：

```bash
uv python install 3.11
uv venv --python 3.11 --managed-python
uv sync
```

运行默认 G1 配置：

以下命令应在项目根目录执行；`configs/` 和 `models/` 是与 `src/` 同级的外部运行资源，不会复制进 Python package。

```bash
uv run vex-policy --policy-config configs/g1
```

常用部署项可以覆盖：

```bash
uv run vex-policy \
  --policy-config configs/g1 \
  --mqtt-config configs/mqtt.yaml \
  --mqtt-broker mqtt://localhost:1883 \
  --interface eth0 \
  --domain-id 0
```

## 默认 policies

服务启动时预加载并 retained 发布以下 `robot/policies` 数组。v1 中它们均为互斥的 `full_body` policy：

| name | implementation | accepted inputs |
| --- | --- | --- |
| `g1-fastsac-locomotion` | FastSAC locomotion | `vx`, `vy`, `yaw` |
| `g1-ppo-locomotion` | PPO locomotion | `vx`, `vy`, `yaw` |
| `g1-fastsac-wbt-dancing` | FastSAC WBT | 无，选择后自动播放 clip |
| `g1-ppo-wbt-dancing` | PPO WBT | 无，选择后自动播放 clip |

配置按变化频率拆分：

- `vex_policy/robots/g1.py`：稳定的 G1 硬件、关节映射、启动刚度与动作缩放常量。
- `configs/g1/*.yaml`：每个文件只包含一个完整 policy，包括 observation 和 task；文件之间不使用 anchor 或继承。
- `configs/mqtt.yaml`：独立的 broker、topics、超时和发布频率配置。

CLI 中 `--config` 与 `--policy-config` 等价，可传一个目录加载其中全部 `*.yaml`，也可传单个 YAML 只加载一个策略。文件名和加载顺序不构成运行协议，策略始终由 MQTT 中的 `name` 选择。`--mqtt-config` 可指定另一份 MQTT 配置。`model_path` 仅支持绝对路径或相对进程当前工作目录的本地路径，文件不存在会在加载阶段报错。所有 YAML 层级均严格校验，未知字段会导致启动失败。

替换或增加模型时直接编辑对应 policy：

```yaml
# 关键字段示意；完整字段可复制同类 policy 文件后修改
name: my-locomotion
implementation: locomotion
type: full_body
inputs: [vx, vy, yaw]
observation: # 在单个文件中声明完整 ObservationConfig
  # 完整 obs_dict、obs_dims、obs_scales、history_length_dict
task:
  model_path: /opt/vex/models/my-policy.onnx
  rl_rate: 50.0
  # 其余 TaskConfig 字段同样在 YAML 中显式配置
```

## MQTT 协议

控制输入订阅 `robot/commands`（QoS 0），格式与 `../mujoco-arcade-robot-control-panel/src/hooks/useMqttClient.ts` 一致：

```json
{
  "seq": 1,
  "timestamp": 1750000000000,
  "control": {
    "vx": 0.25,
    "vy": 0.0,
    "yaw": -0.1,
    "pitch": 0.0,
    "height": 0.0,
    "policy": ["g1-fastsac-locomotion"],
    "estop": false
  }
}
```

无效、未知或多选 policy 消息会被丢弃且不会刷新 watchdog。默认 1 秒没有合法消息即锁存。

输出 topics：

- `robot/policies`：QoS 1、retained，面板使用的直接 JSON 数组。
- `robot/status`：QoS 1、retained，字段为 `state`、`active_policy`、`requested_policy`、`reason`、`last_command_seq`；配置了 offline Last Will。
- `robot/g1/real/state`：QoS 0、非 retained、默认 50 Hz；字段与 `../sim/g1_mujoco_sim/mqtt.py` 完全一致：`timestamp`、`simulation_time`、`joint_names`、`joint_values`、`base_xyz`、`base_quat_wxyz`。
- `robot/g1/reference/state`：仅在 WBT policy 激活时发布，QoS 0、非 retained、与真实状态同频且使用相同字段结构；`joint_values` 是 WBT 当前参考关节位置，`base_quat_wxyz` 是当前参考姿态，`base_xyz` 固定为 `[0, 0, 0]`。topic 可通过 `reference_state_topic` 修改。

Unitree 低层接口当前没有世界位置估计，所以真实状态中的 `base_xyz` 为 `[0, 0, 0]`。控制面板添加真实 G1 后，需要把实例 motion topic 从默认的 `robot/g1/mujoco/state` 改为 `robot/g1/real/state`。

## 安全状态机

- 启动状态为 `startup_latched`，不会发送任何低层命令。
- 先收到 `estop=false` 且 `policy=[]` 才进入 `idle`；随后选择一个 policy 才启动推理。
- `estop=true`、合法命令超时、空 policy 和策略切换间隙都不会调用 `write_low_command`，但仍读取并发布机器人状态。
- 急停或超时解除后不会自动恢复；必须先发送非急停空 policy，再重新选择。
- 两个 policy 直接切换时会停止旧实例、更新目标 KP/KD、重置目标状态，并保留一个控制周期的低层命令空档。

这里的“停止”明确表示停止发送低层命令，机器人最终行为由固件 watchdog/当前控制模式决定。部署前必须在安全支撑环境验证固件侧行为，MQTT estop 不能代替物理急停。

## Unitree high-level

迁移的 `vex_policy.sdk.unitree_high_level` 默认只做懒加载。如需使用其 arm/loco client：

```bash
uv sync --extra unitree-high-level
```

该 extra 固定到 Unitree 官方 `unitree_sdk2_python` 提交 `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`，并通过子进程代理隔离 CycloneDDS。

## 验证

```bash
uv run pytest -q
uv run ruff check src tests
uv build
```

测试包含四个真实 ONNX 的加载/单步推理、共享 Unitree 接口、多 policy 切换与锁存、状态格式，以及本机 Mosquitto 真实往返。
