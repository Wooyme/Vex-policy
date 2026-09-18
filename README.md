# vex-policy

独立的 Unitree G1 ONNX policy 推理服务。控制命令只从 MQTT 读取，运行时可在多个已预加载 policy 之间安全切换，并发布真实机器人状态供 Vex 控制面板显示。

本项目从 Holosoma 提交 `f5445d1b56a5bad39da17ce468df460477a3a1d5` 的 inference 包迁移而来，并迁入 GR00T-WholeBodyControl 提交 `c374bae5b9039cd0ee71377e654d11ce1bc69e1d` 的 GEAR-SONIC 部署闭环。策略、encoder 和 planner 由 Python 实现；DDS 订阅、状态/命令缓存、CRC 和低层命令发布线程由 `far-unitree-sdk` 的 C++/pybind11 binding 实现。本仓库不包含自有的 C++、CMake 或 TensorRT 代码。许可与来源见 `LICENSE`、`NOTICE`。

## 环境

项目固定使用 uv 管理的 Python 3.11，并锁定 `far-unitree-sdk==0.1.7`。Python 侧导入的是该包提供的 `unitree_interface` binding，不是 `unitree_sdk2py`：

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
  --sdk-log-dir logs/sdk
```
 
`--sdk-log-dir` 未指定时关闭 SDK 高频日志。启用后，每次运行会在目标目录创建独立会话目录，后台线程将
应用层 `get_low_state` 的返回值（包括空读）和应用层最终传给 `write_low_command` 的 motor-order 命令写成
5 秒一个的压缩 `chunk_*.npz`。它记录的是 Python API 调用，不是 SDK 内部每个 DDS 收发包；默认 50 Hz 控制
时通常每周期有一条 state 记录，只有稳定运行的 active policy 周期才有一条 command 记录。控制线程只向
有界内存队列提交快照；队列满时丢弃最旧记录，日志压缩或磁盘错误不会中断机器人控制。正常退出会刷新
不足 5 秒的尾部分块，异常退出可能丢失尚未落盘的数据。日志不会自动清理。

### 离线日志可视化

日志查看器使用独立命令启动，不会加载 policy、连接 MQTT 或创建机器人接口。可视化依赖是可选项，不会
进入机器人服务的默认安装依赖：

```bash
uv run --extra viewer vex-log-viewer --log-dir logs
```

浏览器打开终端中显示的 `http://127.0.0.1:8050`，即可在已有会话之间切换，查看调用频率、控制周期抖动、
命令写入耗时、无效状态、失败命令、队列丢弃，以及关节和基座时序曲线。关节页面可以对照实际位置、速度、
估算力矩与对应目标命令，并查看电机状态码 `motorstate` 和 KP/KD。29 维 G1 日志显示关节名称；其他维度使用数字索引，且不会进行无法
确认顺序的 state/command 叠加。

查看器是离线快照：启动时只扫描已经完成原子重命名的 `chunk_*.npz`，忽略 `.partial`，不会自动读取后来
写入的新分块。需要查看新数据时重启命令。日志目录、监听地址和端口均可覆盖：

```bash
uv run --extra viewer vex-log-viewer \
  --log-dir logs/sdk \
  --host 127.0.0.1 \
  --port 8050
```

## SDK 读写与频率

进程中只由 `InterfaceManager` 创建并持有一个 robot interface。policy 不直接持有或调用 interface；所有
LowState 读取和合成后的 LowCommand 写入都经过这个单例。配置加载时还会检查所有 policy 的 `rl_rate`
完全一致，因此上下肢组合不会各自建立通信周期。

以仓库锁定的 `far-unitree-sdk==0.1.7` 和默认 `rl_rate: 50.0` 为例，各层含义如下：

| 层级 | 触发频率 | 实际行为 |
| --- | --- | --- |
| 机器人 → FAR SDK LowState | 由机器人发布端和 DDS 回调决定 | C++ 回调异步校验 CRC，并更新 SDK 内部的最新状态缓存；本项目不配置这个接收频率。 |
| `PolicyStateMachine` → `get_low_state` | 每个 `rl_rate` 周期一次；默认 50 Hz，即 20 ms | `read_low_state()` 读取 C++ 最新缓存并转换为项目 joint order，不会为这次调用发起一次 DDS 网络读取。idle、锁存和切换周期也会读一次。 |
| policy 推理 | 每个控制周期至多一次 | 单 policy 直接计算；同时运行的 upper/lower policy 在两个 worker 中并行计算，并共享同一个 LowState 对象。 |
| 应用层 → `write_low_command` | 稳定 running 状态每周期一次，默认至多 50 Hz | 先按受控关节合成一条完整命令并转换为 motor order，再一次性更新 SDK 的命令缓存。两个 policy 不会分别写 SDK。 |
| FAR SDK → 机器人 LowCmd | SDK 内部线程固定 2,000 us 周期，标称 500 Hz | C++ writer 读取最新命令缓存、生成 DDS 消息、计算 CRC 并发布；在应用层没有新命令时仍会重复发布缓存中的上一条命令。 |
| MQTT robot state | `mqtt.state_frequency_hz`，默认 50 Hz | 只控制 MQTT 状态降采样，且必须不高于 `rl_rate`；不改变 SDK 的读写频率。 |

这里的 500 Hz 只描述锁定版本 SDK 的 **LowCmd 发布线程**，不能据此认为 LowState 接收也是 500 Hz。
应用层没有额外的 50 Hz reader 线程或二级状态缓存，所以不会人为再增加一个完整的 20 ms 缓存周期；一次
tick 使用其开始时能取得的最新 SDK 快照，推理完成后更新命令缓存，SDK writer 通常会在下一个约 2 ms
发布周期取到它（不包含操作系统调度和网络抖动）。本项目也没有“新帧”序号检查或 LowState 超时缓存，
所以连续控制周期可能读到同一份底层样本。

Unitree binding 返回的 `q`、`dq`、`ddq`、`tau_est`、`motorstate` 会映射到 `LowState` 中与 `joint_pos` 相同的形状；某个
字段缺失或长度不足时以 `np.zeros_like(joint_pos)` 补齐。由于 binding 当前不提供世界坐标 base position 和
linear velocity，这两项同样为零。写入方向相反：policy joint order 的 `q/dq/tau/kp/kd` 先映射为 motor
order，`write_low_command()` 只更新 C++ 缓存，真正的 DDS 发布发生在 SDK writer 线程。

当前 Unitree backend 的 C++ binding 固定使用 DDS domain 0；虽然 CLI 仍接受 `--domain-id`，该值目前没有
传入 `unitree_interface.create_robot()`，不能用它切换 DDS domain。

## 默认 policies

服务启动时预加载并 retained 发布所选目录顶层 `*.yaml` 中的 policy。默认 `configs/g1` 当前包含
`g1-ppo-locomotion`、`ppo-doggy1`、`ppo-ridding1` 和四个 SONIC 起身/跪下转换 policy；名称、类型和输入
以对应 YAML 为准。`full_body` policy 互斥运行；一个 `lower_body` 和一个 `upper_body` policy 可以同时选择
并按关节合成命令。

## GEAR-SONIC

SONIC 配置位于 `configs/examples/sonic/`，每个 planner mode 都是一个完整且独立的现有格式 YAML。该子目录不会被默认的 `configs/g1` 非递归加载影响。

先将上游三个模型放到 `models/sonic/`：

```text
gear_sonic_deploy/policy/release/model_decoder.onnx       -> models/sonic/model_decoder.onnx
gear_sonic_deploy/policy/release/model_encoder.onnx       -> models/sonic/model_encoder.onnx
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx -> models/sonic/planner_sonic.onnx
```

模型权重不提交到 Git。也可以在 YAML 中把三个路径改成部署机上的绝对路径。启动全部 27 个模式：

```bash
uv run vex-policy --policy-config configs/examples/sonic --interface eth0
```

三个 ONNX session 按模型路径和 provider 共享，27 个 policy 不会重复加载权重。默认 `inference_provider: auto` 优先 CUDA，缺少 CUDA provider 时回退 CPU；也可显式设为 `cpu` 或 `cuda`。

也支持不加载 planner、直接播放 `gear_sonic_deploy` 格式的参考动作。完整配置见 `configs/examples/g1_sonic_motion_directory.yaml`，将其中的 `motion_data_path` 改为动作路径后运行：

```bash
uv run vex-policy \
  --policy-config configs/examples/g1_sonic_motion_directory.yaml \
  --interface eth0
```

`motion_data_path` 可以指向一个动作目录，也可以指向包含多个动作子目录的集合；集合中有多个有效动作时需要填写 `motion_name`。每个动作目录包含带表头的 `joint_pos.csv`、`joint_vel.csv`、`body_pos.csv` 和 `body_quat.csv`，采样率为 50Hz，29 个关节列使用 IsaacLab/policy 顺序，根四元数使用 `wxyz`。`motion_loop: false` 会在动作末帧保持，设为 `true` 则循环播放。目录模式只加载 decoder 和 encoder，不校验也不创建 planner session。

## WBT 启动模式

WBT 的 `task.startup_mode` 支持 `interpolate`（默认）和 `immediate`。`interpolate` 会从激活瞬间的
关节姿态出发，在 `task.init_duration_s`（默认 10 秒）内线性移动到 motion 首帧，完成后才开始推进
motion timestep 和输出策略动作；插值目标同时受 G1 关节位置限制和逐周期速度限制保护，速度安全系数由
机器人配置 `joint_interpolation_slew_safety_factor` 控制。`immediate` 保留直接启用策略动作的行为。

## UFO-Deploy

`configs/examples/ufo/` 提供 UFO G1 发布策略的离线 tracking、reward 和 goal 示例。默认路径引用同级
`../../UFO/model/g1_policy` 发布物；模型和 context 也可以改为部署机上的任意绝对路径或相对启动工作目录的
本地路径。启动三类示例：

```bash
uv run vex-policy --policy-config configs/examples/ufo --interface eth0
```

每个 reward/goal YAML 固定选择一个命名 latent，通过 policy 名切换，不需要新的 MQTT 输入类型。reward
使用不依赖 Torch 的 `reward_locomotion_numpy.pkl`。多个 UFO policy 共享同一路径/provider 的 ONNX session。

选中 UFO policy 后会先用其发布 KP/KD，并通过与 WBT 共用的 G1 关节插值器在 10 秒内移动到 UFO
默认站姿；推理同时运行以预热
4 帧历史，完成后才推进 tracking。tracking 播放到 `end_frame` 后固定使用 `stop_frame` latent。状态、context、
推理动作或最终目标出现非法值时，本周期不会写命令且服务进入全局 `latched`；需先发送非急停空 policy，
再重新选择。UFO context 使用 pickle/joblib 格式，只应加载可信发布物。本集成不包含实时 ZMQ/PICO teleop
或 backward encoder。

将 `task.startup_mode` 设为 `prefill` 可跳过 10 秒插值，由当前关节姿态反算等价 residual action，并立即
填满 4 帧 observation/action 历史；该配置只接受 `prefill` 和 `interpolate`。启动姿态检查与模式独立，
由顶层 `guard.startup_joint_tolerance_rad` 和 `guard.startup_gravity_tolerance` 配置、通用的 `InitialPoseGuard`
执行；检查失败时拒绝激活并进入锁存。

UFO、waist locomotion、passive locomotion 和 WBT 共用 `InitialPoseGuard` / `GuardConfig`。
Guard 接收独立的 `InitialPose`（关节名、位置、基座 `wxyz` 四元数），按机器人关节顺序对齐后检查全身关节。
UFO 参考模型默认姿态和直立朝向，waist/passive 参考 motion 最后一帧，WBT 参考配置起始帧；
WBT 的躯干参考朝向先通过初始关节位置的 FK 转换为基座朝向。参考姿态不会随推理帧推进而改变。

每个关节可以单独设置阈值，未列出的关节使用 `startup_joint_tolerance_rad`：

```yaml
guard:
  startup_joint_tolerance_rad: 0.2
  startup_joint_tolerances_rad:
    left_knee_joint: 0.35
    right_knee_joint: 0.35
  startup_gravity_tolerance: 0.2
```

两个默认阈值均为 `0.2`，覆盖映射默认为空。阈值必须有限且大于零，未知关节名会报错。
关节绝对误差超过各自阈值才拒绝启动；动作 mask 不排除检查中的关节。
重力阈值是完整投影重力向量差的欧氏范数，不是角度，也不检查航向角。
检查还会拒绝形状错误、非有限的关节位置/速度、基座角速度，以及非法四元数；有效四元数会归一化，
不修改输入状态。waist/passive 必须配置 `guard`，UFO/WBT 省略时不启用检查。

WBT 启用 Guard 后同样检查全身和完整重力向量，原来的仅腿部、仅重力 Z 分量检查已移除。
旧 `bad_lower_joint_pos_threshold`、`bad_ref_ori_threshold` 字段和各策略专用 Guard/配置类不再支持；
迁移时需明确设置新阈值，旧重力阈值不能直接视为新阈值。

策略还可配置输出限位器 `limiter` 和运行急停器 `estop`，不配置时保持原有行为。
配置中的角度统一为 rad、速度为 rad/s，关节名与机器人配置一致：

```yaml
limiter:
  joints:
    left_knee_joint:
      pos: {min: -0.05, max: 2.5}
      vel: 10.0
estop:
  joints:
    left_knee_joint:
      pos: {min: -0.08, max: 2.7}
      vel: 15.0
  rpy:
    roll: {min: -0.5, max: 0.5}
    pitch: {min: -0.6, max: 0.6}
    yaw: {min: -3.14, max: 3.14}
  fallback:
    policy: recovery-policy
    inputs:
      target_height: 0.25
```

`pos`、`vel` 和各 RPY 轴均可独立省略。范围满足 `min <= max`，速度上限非负，所有值必须有限；
等于边界允许。未知关节名、非法范围和无效替代策略会在启动时报错。

`limiter` 位于所有策略共有的输出后处理阶段，也覆盖启动插值和保持姿态。
它按最终计入校准偏移的关节坐标裁剪 `q`，只对命令实际控制的关节生效；
G1 配置与硬件边界取交集，无交集时报错。`vel` **只裁剪 `dq` 命令**，不限制 `q` 的逐周期变化率；
多数位置策略的 `dq` 为零，其已有插值和位置限速逻辑仍独立生效。限位不修改动作历史、力矩或 PD 参数。

`estop` 在激活前和每周期推理前检查实测状态，配置的关节不受动作 mask 排除。
基座 RPY 为归一化四元数对应的世界系 ZYX 欧拉角，roll/yaw 使用 `[-π, π)` 主值，pitch 使用
`[-π/2, π/2]`；不支持跨越 ±π 接缝的范围或累计航向。任意项目单次严格越界即触发，
该周期不执行原策略推理、不下发命令，上下半身并行策略会一起停止。

省略 `fallback` 时进入现有 `latched` 状态，停止下发命令，不自动发送保持或阻尼命令。
配置 `fallback` 时，其 `policy` 必须是已加载的另一个全身策略，`inputs` 可覆盖该策略声明的部分
默认输入，未覆盖参数沿用默认值。替代策略正常执行自己的急停检查和启动 Guard，通过后进入
`fallback` 状态，下一周期才开始下发命令。非法实测状态直接锁存，不尝试替代。

替代期间固定使用上述输入，忽略外部非空策略选择及输入；先发送非急停空选择结束替代，之后才能
重新选策略。外部急停、命令超时和状态缺失仍会停止替代，因此仍需持续发送心跳。
状态消息保留实际 `active_policy`、外部 `requested_policy` 和触发 `reason`。
多个检查器同时触发时，停止优先；仅当替代目标和有效输入完全一致时执行替代，否则锁存。
替代策略启动被拒绝、再次越界或运行失败时直接锁存，不继续递归替代。

模型无关的完整示例位于 `configs/examples/safety/`，可使用
`vex-policy --config configs/examples/safety` 加载。`limited-hold` 越界后切换为 `recovery-hold`，
后者捕获并保持切换时的实测姿态。示例阈值只在该示例目录中启用，现有部署配置不会自动启用限制。

配置按变化频率拆分：

- `src/vex_policy/robots/g1.py`：稳定的 G1 硬件、关节映射、启动刚度与动作缩放常量。
- `configs/g1/*.yaml`：每个文件只包含一个完整 policy，包括 observation 和 task；文件之间不使用 anchor 或继承。
- `configs/g1/action_masks/*.yaml`：按关节名定义的模型输出 mask；该子目录不会被 policy 目录加载器扫描。
- `configs/mqtt.yaml`：独立的 broker、topics、超时和发布频率配置。

CLI 中 `--config` 与 `--policy-config` 等价，可传一个目录加载其中全部 `*.yaml`，也可传单个 YAML 只加载一个策略。文件名和加载顺序不构成运行协议，策略始终由 MQTT 中的 `name` 选择。`--mqtt-config` 可指定另一份 MQTT 配置。`model_path` 仅支持绝对路径或相对进程当前工作目录的本地路径，文件不存在会在加载阶段报错。所有 YAML 层级均严格校验，未知字段会导致启动失败。

替换或增加模型时直接编辑对应 policy：

```yaml
# 关键字段示意；完整字段可复制同类 policy 文件后修改
name: my-locomotion
implementation: locomotion
type: lower_body
inputs:
- type: joystick
  x: {name: vx, min: -1.0, max: 1.0, default: 0.0}
  y: {name: vy, min: -1.0, max: 1.0, default: 0.0}
- type: slider
  parameter: {name: yaw, min: -1.0, max: 1.0, default: 0.0}
observation: # 在单个文件中声明完整 ObservationConfig
  # 完整 obs_dict、obs_dims、obs_scales、history_length_dict
task:
  model_path: /opt/vex/models/my-policy.onnx
  action_mask_path: action_masks/disable_upper_body.yaml
  rl_rate: 50.0
  # 其余 TaskConfig 字段同样在 YAML 中显式配置
```

`action_mask_path` 相对当前 policy YAML 所在目录解析；设为 `null` 表示不屏蔽。mask 文件使用
`masked_joints` 列出需要置零的 residual action。仓库提供 `disable_upper_body.yaml`（双臂 14 个关节）和
`disable_lower_body.yaml`（双腿与腰部 15 个关节）。屏蔽维单独运行时保持 `default_dof_angles`，组合运行时
不会写入合成命令，因此可由另一身体区域的策略接管。`lower_body`/`upper_body` 的未屏蔽关节不得越过各自区域。

Holosoma `waist_loco` 分支的 pelvis-sine 策略使用独立示例
`configs/examples/g1_waist_locomotion.yaml`。将对应的 105→29 ONNX 模型复制为
`models/loco/g1_29dof/ppo_g1_waist.onnx` 后，可通过
`vex-policy --config configs/examples/g1_waist_locomotion.yaml` 单独启动。该策略保留模型的全 29 维输出，
并使用 `motion_data_path` 所指 NPZ motion 的最后一帧作为关节残差零位；激活前会检查当前关节和机身倾斜
是否接近该帧。motion 必须包含根姿态格式为 `[xyz,wxyz]` 的 `joint_pos` 和可用于关节重排的 `joint_names`。
启动检查由独立的
`InitialPoseGuard` 执行，其关节和重力误差阈值配置在 policy YAML 顶层的 `guard` 中。

105 维 pelvis-sine 分支将六个真实物理量分别声明为滑条：`amplitude` 范围 `[0.05,0.20]`、默认 `0.125`，
`frequency` 范围 `[0.2,2.0]`、默认 `1.1`，方向 `x/y/z` 范围均为 `[-1,1]`、默认
`[1,0,0]`，基座相对右脚踝高度增量 `height_delta` 范围为 `[-0.1,0.1]` 米、默认 `0`。方向在策略内归一化，
全零方向回退到配置的默认方向。这些范围和默认值只在 `inputs` 中
配置，不在 task 中重复。waist locomotion 的 `motion_data_path` 与 `model_path` 一样相对进程当前工作目录
解析。pelvis command 开头的 `sin_phase`/`cos_phase` 由策略按当前频率和 `rl_rate` 内部推进，不占用额外
MQTT 参数。`pelvis_orientation_error` 以每次策略成功启动时机器人的真实 quaternion 为参考，因此允许启动
姿态与 motion 中的根 quaternion 不一致；策略停用后再次启动会重新采样该参考姿态。命令中的目标高度差以
启动瞬间的高度差加 `height_delta` 得到；当前高度差 observation 使用 ONNX 内嵌 URDF 对关节角做正运动学，
并与 IMU 投影重力结合求出，因此部署时不依赖无法观测的基座或脚踝绝对世界高度。

同一个 `implementation: waist_locomotion` 也支持 Holosoma `waist_loco0` 的
`g1_29dof_loco_lite`（hip-pitch-sine）模型。加载时根据 ONNX 的 `actor_obs[1,105]` 或
`actor_obs[1,102]` 自动选择分支，并调整命令 observation 的名称、维度和字母排序，无需额外的分支开关。
102 维分支使用三个滑条：`amplitude` 为髋关节摆幅（**弧度**，训练范围 5°–20° 对应约
`[0.0872664626,0.3490658504]`），`frequency` 为 Hz，`height_delta` 为米；不声明 `x/y/z`。
其五维命令为 `[sin_phase, cos_phase, amplitude_rad, frequency_hz, target_height]`，按训练定义排在
`dof_vel` 之后、`pelvis_orientation_error` 之前。输入滑条必须匹配检测到的分支，避免把旧模型的米制摆幅
误用于新模型。两种分支均使用 NPZ 最后一帧关节作为残差零位；102 维分支的姿态误差以该帧根 quaternion
为参考，105 维分支保留启动瞬间的实测参考。现有 `configs/g1/ppo_doggy0.yaml` 已对应本地的 102→29
模型；独立示例见 `configs/examples/g1_waist_locomotion_lite.yaml`，可用
`vex-policy --config configs/examples/g1_waist_locomotion_lite.yaml` 启动。

## Passive locomotion

Holosoma `passive_waist_loco` 分支的 `exp:g1-29dof` 使用独立的
`implementation: passive_locomotion`。完整示例为
`configs/g1/g1_passive_locomotion.yaml`，不会被默认 `configs/g1` 加载。
将该实验导出的 ONNX 放到 `models/loco/g1_29dof/ppo_g1_passive.onnx` 后运行：

```bash
uv run vex-policy --policy-config configs/examples/g1_passive_locomotion.yaml --interface eth0
```

模型必须提供 float32 `actor_obs[1,1960] → action[1,29]`，并包含 `dof_names`、
`action_scale`、`robot_urdf` 和 KP/KD 元数据。历史力估计器已包含在 ONNX 内，无需独立
encoder、外部力传感器输入或训练用的 `force_target`。旧的 100/102/105 维模型不能用于该示例。
模型权重不随适配代码提供。

策略以 50 Hz 保存 20 帧历史。七项观测按字母排序，每项内部按从旧到新排列：
`actions(29)`、`base_ang_vel(3)`、`base_right_foot_height_difference(1)`、`dof_pos(29)`、
`dof_vel(29)`、`passive_command(4)`、`projected_gravity(3)`。角速度缩放为 0.25，关节速度
缩放为 0.05，其余为 1。每次激活清空历史并在前端补零，不重复填充当前帧。

四个滑条直接形成 `[down_vel, up_vel, target_height, min_height]`，无相位或高度偏移：

| 参数 | 含义 | 范围 | 默认值 |
| --- | --- | --- | --- |
| `down_vel` | 下压速度，m/s | 0.01–0.05 | 0.03 |
| `up_vel` | 回升速度，m/s | 0.01–0.10 | 0.055 |
| `target_height` | pelvis link 原点的恢复目标世界高度，m | 0.15–0.40 | 0.275 |
| `min_height` | pelvis link 原点的最低世界高度，m | 0.05–0.15 | 0.10 |

命令中的高度是训练定义的世界高度；高度 observation 则是基座相对右脚踝的高度差，由
ONNX 内嵌 URDF 的 FK 和 IMU 投影重力计算。URDF 中的 `hip_bar_yaw_joint`、
`hip_bar_pitch_joint` 保持零位，不进入 29 维观测关节或输出动作。

示例 `motion_data_path` 指向上游现有的 `ridding1.npz`，与 `model_path` 一样相对启动工作目录
解析。**必须使用该模型训练时 `--robot.init-state.initial-pose-file` 对应的同一姿态文件**。
读取 NPZ 最后一帧，按 `joint_names` 重排关节，作为动作残差零位；根姿态采用 `[xyz,wxyz]`。
激活前必须先到达该参考姿态，关节最大误差默认不超过 0.2 rad，投影重力向量误差默认不超过
0.2，均由顶层 `guard` 配置。检查通过后直接推理，不自动插值。

动作历史保留裁剪前模型输出；下发位置使用动作裁剪至 `[-100,100]` 后乘 0.25，再叠加参考姿态
并限制到 G1 关节范围。该类型仅支持 `full_body`，不接受 action mask。运行中非法状态、观测、
动作或推理错误会触发现有全局锁存，本周期不写入机器人命令。

## Enhanced inputs

每个 policy 的 `inputs` 是有序的 UI 组件数组，控制面板按 `type` 自动渲染。`joystick` 同时声明明确的
`x`、`y` 参数，`slider` 声明一个 `parameter`；每个参数都包含唯一的 `name`、闭区间 `min/max` 和
`default`。范围、默认值和参数名在加载阶段严格校验。没有手动输入的 policy 使用 `inputs: []`。

```yaml
inputs:
- type: joystick
  x: {name: vx, min: -1.0, max: 1.0, default: 0.0}
  y: {name: vy, min: -1.0, max: 1.0, default: 0.0}
- type: slider
  parameter: {name: yaw, min: -1.0, max: 1.0, default: 0.0}
```

## Hold position

`hold_position` 是不加载 ONNX 网络的保持策略，完整示例见 `configs/examples/g1_hold_position.yaml`。每次
activate 使用当前状态机周期已经读取的共享 LowState，保存当时的全部 `dof_pos`，不会额外读取 SDK；之后
持续将该姿态作为位置目标。deactivate 后丢弃快照，因此再次 activate 会捕获新的姿态。它没有控制参数，
使用 `inputs: []`，task 只配置 `rl_rate` 和可选的 `action_mask_path`。

该策略优先使用 robot config 的 `motor_kp/motor_kd`，缺失时使用 `stiff_startup_kp/stiff_startup_kd`。
`action_mask_path` 与其他策略相同：被 mask 的关节不会写入组合命令，可交给同时运行的另一个身体区域策略。
例如将 policy 设为 `lower_body` 并使用 `configs/g1/action_masks/disable_upper_body.yaml`，即可只保持下半身。

## Interpolation

`interpolation` 是不加载或运行神经网络的独立插值策略，完整示例见
`configs/examples/g1_interpolation.yaml`。将 `task.motion_data_path` 指向本地 NPZ，填写必需的
`duration_s`（期望插值秒数），并用 `target_frame: first` 或 `last` 选择第一帧或最后一帧，默认 `first`。
动作路径相对进程当前工作目录解析；配置不需要 `model_path`，手动输入使用 `inputs: []`。

NPZ 必须包含一维字符串数组 `joint_names` 和弧度单位的 `joint_pos`，后者支持 `(帧数, 关节数)` 或
`(帧数, 7 + 关节数)`；带根姿态时跳过前 7 列 `[xyz,wxyz]`，按名称重排到硬件关节顺序。
只读取所选端点作为固定目标，不播放中间帧，也不要求 `joint_vel` 或刚体姿态数组。

每次激活从共享 LowState 捕获实际关节位置，按 `rl_rate` 周期线性推进，标称步数为
`max(1, round(duration_s * rl_rate))`。目标包含机器人配置的校准偏移，并裁剪到 G1 关节位置限制；
每周期位移受 G1 速度限制与 `robot.joint_interpolation_slew_safety_factor` 约束。
期望时间不足时继续受限插值，直到输出位置命令到达裁剪后的目标，然后持续保持，实际到达时间可能更长。
这不表示实际关节已经无误差地跟踪到目标。停用后清除进度，下次激活从新的实际姿态开始。

KP/KD 优先使用 `motor_kp/motor_kd`，缺失时使用 `stiff_startup_kp/stiff_startup_kd`；
目标速度与前馈力矩为零。`action_mask_path` 的路径和关节归属规则与保持策略相同，支持上下半身组合。
该策略仍通过现有 MQTT 状态机激活、停用与接收心跳。

## MQTT 协议

控制输入订阅 `robot/commands`（QoS 0），格式与 `../vex-panel/src/hooks/useMqttClient.ts` 一致：

```json
{
  "seq": 1,
  "timestamp": 1750000000000,
  "control": {
    "policy": ["g1-ppo-locomotion"],
    "inputs": {
      "g1-ppo-locomotion": {
        "vx": 0.25,
        "vy": 0.0,
        "yaw": -0.1
      }
    },
    "estop": false
  }
}
```

policy 数组可为空、包含一个 policy，或同时包含一个 `lower_body` 和一个 `upper_body` policy。`full_body`
必须独占。`inputs` 必须按 policy 名分组，其键与所选 policy 精确一致；每组必须提供该 policy 声明的全部
参数且不得包含额外参数，数值必须在声明的闭区间内。空 policy 必须搭配空 inputs。重复类型、未知名称、
参数不完整、越界和超过两个 policy 的消息都会被丢弃且不会刷新 watchdog。默认 1 秒没有合法消息即锁存。

输出 topics：

- `robot/policies`：QoS 1、retained，面板使用的直接 JSON 数组；每项包含 `name`、`type` 和完整 enhanced
  inputs 组件描述。
- `robot/status`：QoS 1、retained，字段为 `state`、数组类型的 `active_policy`/`requested_policy`、`reason`、`last_command_seq`；配置了 offline Last Will。
- `robot/g1/real/state`：QoS 0、非 retained、默认 50 Hz；字段与 `../sim/g1_mujoco_sim/mqtt.py` 完全一致：`timestamp`、`simulation_time`、`joint_names`、`joint_values`、`base_xyz`、`base_quat_wxyz`。
- `robot/g1/reference/state`：仅在单个、可提供参考状态的 policy 激活时发布；组合策略运行时不发布。QoS 0、非 retained、与真实状态同频且使用相同字段结构。

Unitree 低层接口当前没有世界位置估计，所以真实状态中的 `base_xyz` 为 `[0, 0, 0]`。控制面板添加真实 G1 后，需要把实例 motion topic 从默认的 `robot/g1/mujoco/state` 改为 `robot/g1/real/state`。

## 安全状态机

- 启动状态为 `startup_latched`，不会发送任何低层命令。
- 先收到 `estop=false` 且 `policy=[]` 才进入 `idle`；随后选择一个合法 policy 或上下半身组合才启动推理。
- `estop=true`、合法命令超时、空 policy 和策略切换周期都不会在应用层调用 `write_low_command`，但仍读取并发布机器人状态。
- “不调用 `write_low_command`”只表示不再更新 SDK 命令缓存：如果此前已经写过命令，FAR SDK 的 500 Hz writer 仍会重发最后一条缓存命令。这些状态机行为不是底层电机停机或阻尼命令。
- 急停或超时解除后不会自动恢复；必须先发送非急停空 policy，再重新选择。
- 选择变化时仅停止被移除的实例并初始化新增实例，未变化策略保留历史与相位；切换时跳过一个控制周期的应用层命令更新，SDK 仍可能重发上一条缓存命令。

## Policy 生命周期与扩展

所有策略直接继承 `BasePolicy`。基类只管理公共关节信息、控制范围、guard、生命周期和周期计时；
不创建模型、观测历史、步态或启动插值。实例构造完成后处于 inactive，模型与静态动作数据在构造期间预加载。
基类构造函数不调用子类初始化钩子，子类需显式调用 `super().__init__(config)`，然后创建自身组件。

公共接口为 `activate(state)`、`apply_control(control)`、`step(state)`、`deactivate()`、`close()`，
实现策略时覆盖以下钩子，保留公共接口的统一外壳：

| 钩子 | 职责 |
| --- | --- |
| `_on_activate(state)` | 清空上一回合的动作、历史、相位与播放进度，恢复配置输入默认值，采样当前姿态并启动本回合任务；成功返回 `None`，预期拒绝返回原因字符串。 |
| `_apply_control(control)` | 解释该策略的运行输入；无输入的策略可使用默认空实现。 |
| `_compute_command(state)` | 必须实现；从共享 `LowState` 计算一条完整硬件关节顺序的 `PolicyJointCommand`，不读写硬件。 |
| `_on_deactivate()` | 停止并等待后台任务，清理本回合状态；也必须能清理未完成的激活。 |
| `_on_close()` | 释放实例资源；inactive 实例也会调用。 |

`get_reference_state()` 仍为可选接口。`activate()` 在 guard 通过后进入激活钩子，钩子失败会回滚；
`PolicyRuntimeFault` 表示应触发全局锁存的预期数据/推理故障，其余异常清理后继续抛出。
运行中的重复 `activate()` 不重启；`deactivate()` 和 `close()` 可重复调用，closed 实例不可复用。
inactive/closed 实例调用 `step()` 或 `apply_control()` 会报错。生命周期调用与本实例的单步计算串行化，
后台任务不得反向调用公共生命周期接口。

停用后再次激活统一开始新回合，包含 locomotion 的历史、动作、站立状态和相位重置。
WBT 的初始化/跟踪、UFO 的插值/预填充、SONIC 的规划/播放仍由各策略自己管理。
组合切换中未移除的策略继续当前回合。切换周期仍不计算命令，下一运行周期先应用最新 MQTT 输入再执行单步。

公共组件按需组合：`ObservationHistory` 保持 PPO 模型的逐项排序、缩放、时间历史及零填充；
`OnnxActor`、`load_metadata`、`resolve_control_gains` 提供模型支持；`PositionAction` 与
`position_command` 处理动作和位置命令。SONIC/UFO 保留专属观测布局与动作顺序，共用按路径/provider
缓存的 ONNX session；一个策略关闭时不会销毁其他策略正在使用的共享 session，缓存随进程结束释放。
关节命令仍使用校准前约定，由状态机合成后统一添加偏移。

模型与静态数据跨回合复用，SONIC planner 在停用时停止并等待当前推理结束，WBT 时钟订阅在关闭时释放。
构造函数必须自行清理部分创建的资源；状态机会关闭批量预加载中已成功创建的实例。
组合推理失败时先等待所有本周期任务结束，再清理并锁存，整个失败周期不发送命令。
旧的 `_init_*` 初始化覆写、`_handle_start_policy`、`compute_joint_command` 和实例内多模型切换接口已移除，
外部策略实现需迁移到上述钩子；YAML、MQTT 和策略注册名称保持兼容。

## 验证

```bash
uv run pytest -q
uv run ruff check src tests
uv build
```

现有测试覆盖 SDK 日志分块/丢弃/错误隔离、BaseInterface 日志包装、InterfaceManager 单例、组合 policy
每周期只读一次 LowState、上下肢并行计算、合成后只写一次命令，以及各策略重启、激活回滚和资源清理；
使用 fake ONNX 检查观测排列和命令数值，不包含真实机器人、真实 ONNX 推理或
Mosquitto 集成测试。
