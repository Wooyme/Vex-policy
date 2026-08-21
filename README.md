# vex-policy

独立的 Unitree G1 ONNX policy 推理服务。控制命令只从 MQTT 读取，运行时可在多个已预加载 policy 之间安全切换，并发布真实机器人状态供 Vex 控制面板显示。

本项目从 Holosoma 提交 `f5445d1b56a5bad39da17ce468df460477a3a1d5` 的 inference 包迁移而来，并迁入 GR00T-WholeBodyControl 提交 `c374bae5b9039cd0ee71377e654d11ce1bc69e1d` 的 GEAR-SONIC 部署闭环。策略、encoder、planner、DDS 命令缓存与 CRC 均由 Python 实现，不包含项目自有的 C++、CMake 或 TensorRT 代码。许可与来源见 `LICENSE`、`NOTICE`。

## 环境

项目固定使用 uv 管理的 Python 3.11，并直接依赖 Unitree 官方 `unitree_sdk2py`：

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

## GEAR-SONIC

SONIC 配置位于 `configs/g1/sonic/`，每个 planner mode 都是一个完整且独立的现有格式 YAML。该子目录不会被默认的 `configs/g1` 非递归加载影响。

先将上游三个模型放到 `models/sonic/`：

```text
gear_sonic_deploy/policy/release/model_decoder.onnx       -> models/sonic/model_decoder.onnx
gear_sonic_deploy/policy/release/model_encoder.onnx       -> models/sonic/model_encoder.onnx
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx -> models/sonic/planner_sonic.onnx
```

模型权重不提交到 Git。也可以在 YAML 中把三个路径改成部署机上的绝对路径。启动全部 27 个模式：

```bash
uv run vex-policy --policy-config configs/g1/sonic --interface eth0
```

三个 ONNX session 按模型路径和 provider 共享，27 个 policy 不会重复加载权重。默认 `inference_provider: auto` 优先 CUDA，缺少 CUDA provider 时回退 CPU；也可显式设为 `cpu` 或 `cuda`。

也支持不加载 planner、直接播放 `gear_sonic_deploy` 格式的参考动作。完整配置见 `configs/examples/g1_sonic_motion_directory.yaml`，将其中的 `motion_data_path` 改为动作路径后运行：

```bash
uv run vex-policy \
  --policy-config configs/examples/g1_sonic_motion_directory.yaml \
  --interface eth0
```

`motion_data_path` 可以指向一个动作目录，也可以指向包含多个动作子目录的集合；集合中有多个有效动作时需要填写 `motion_name`。每个动作目录包含带表头的 `joint_pos.csv`、`joint_vel.csv`、`body_pos.csv` 和 `body_quat.csv`，采样率为 50Hz，29 个关节列使用 IsaacLab/policy 顺序，根四元数使用 `wxyz`。`motion_loop: false` 会在动作末帧保持，设为 `true` 则循环播放。目录模式只加载 decoder 和 encoder，不校验也不创建 planner session。

planner 模式保持源部署的三种频率：decoder/encoder 控制循环 50Hz、planner 调度 10Hz、LowCmd writer 500Hz；目录模式不启动 10Hz planner 线程。writer 每 2ms 重发最近的线程安全命令快照，并填充 `mode_pr`、LowState 中的 `mode_machine` 和纯 Python CRC；LowState CRC 错误或超过 `low_state_timeout_s` 未更新时不会发布。

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

这里的“停止”会禁用 500Hz writer 并清空缓存，机器人最终行为由固件 watchdog/当前控制模式决定。部署前必须在安全支撑环境验证固件侧行为，MQTT estop 不能代替物理急停。

## Unitree SDK

低层和 high-level 客户端统一使用 Unitree 官方 `unitree_sdk2_python` 提交 `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5`。不再依赖 `far-unitree-sdk` 或 `unitree_interface` pybind 模块；需要进程隔离时仍可选择 `unitree_mp` backend。

## 验证

```bash
uv run pytest -q
uv run ruff check src tests
uv build
```

测试包含四个真实 ONNX 的加载/单步推理、共享 Unitree 接口、多 policy 切换与锁存、状态格式，以及本机 Mosquitto 真实往返。
