"""Pure-Python Unitree G1 low-level DDS interface.

The policy loop only updates a command snapshot. A dedicated Python thread
publishes the latest snapshot at 500 Hz, matching the GEAR-SONIC deployment.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from loguru import logger

from vex_policy.config.config_types import RobotConfig
from vex_policy.sdk.base.base_interface import BaseInterface

_HG_LOW_CMD_FORMAT = "<2B2x" + "B3x5fI" * 35 + "5I"
_HG_LOW_STATE_FORMAT = "<2I2B2xI" + "13fh2x" + "B3x4f2hf7I" * 35 + "40B5I"


def _crc32_words(packed: bytes) -> int:
    """Unitree CRC32 over every 32-bit word except the trailing CRC word."""
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for (word,) in struct.iter_unpack("<I", packed[:-4]):
        bit = 1 << 31
        for _ in range(32):
            crc = ((crc << 1) & 0xFFFFFFFF) ^ (polynomial if crc & 0x80000000 else 0)
            if word & bit:
                crc ^= polynomial
            bit >>= 1
    return crc


def _pack_hg_low_command(command) -> bytes:
    values: list[int | float] = [command.mode_pr, command.mode_machine]
    for motor in command.motor_cmd:
        values.extend((motor.mode, motor.q, motor.dq, motor.tau, motor.kp, motor.kd, motor.reserve))
    values.extend(command.reserve)
    values.append(command.crc)
    return struct.pack(_HG_LOW_CMD_FORMAT, *values)


def _pack_hg_low_state(state) -> bytes:
    values: list[int | float] = [*state.version, state.mode_pr, state.mode_machine, state.tick]
    imu = state.imu_state
    values.extend((*imu.quaternion, *imu.gyroscope, *imu.accelerometer, *imu.rpy, imu.temperature))
    for motor in state.motor_state:
        values.extend(
            (
                motor.mode,
                motor.q,
                motor.dq,
                motor.ddq,
                motor.tau_est,
                *motor.temperature,
                motor.vol,
                *motor.sensor,
                motor.motorstate,
                *motor.reserve,
            )
        )
    values.extend((*state.wireless_remote, *state.reserve, state.crc))
    return struct.pack(_HG_LOW_STATE_FORMAT, *values)


def command_crc(command) -> int:
    return _crc32_words(_pack_hg_low_command(command))


def state_crc(state) -> int:
    return _crc32_words(_pack_hg_low_state(state))


@dataclass(frozen=True)
class _CommandSnapshot:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


class UnitreeInterface(BaseInterface):
    """Unitree HG DDS transport implemented with unitree_sdk2py."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        if robot_config.message_type.upper() != "HG":
            raise ValueError("The Python Unitree backend currently supports HG messages only")
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._publish_rate_hz = 500.0
        self._state_timeout_s = 0.1
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._latest_state = None
        self._latest_state_at = 0.0
        self._latest_command: _CommandSnapshot | None = None
        self._writer_enabled = threading.Event()
        self._writer_stop = threading.Event()
        self._mode_machine = 0
        self._closed = False
        self._init_sdk()
        self._writer_thread = threading.Thread(
            target=self._command_writer_loop, name="unitree-lowcmd-500hz", daemon=True
        )
        self._writer_thread.start()

    def _init_sdk(self) -> None:
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        except ImportError as exc:
            raise ImportError("unitree-sdk2py is required for the Unitree backend") from exc

        interface = None if self.interface_str in {None, "", "auto"} else self.interface_str
        ChannelFactoryInitialize(self.domain_id, interface)
        self._low_command_factory = unitree_hg_msg_dds__LowCmd_
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._low_state_handler, 10)
        self._release_high_level_mode(MotionSwitcherClient)

        if self.robot_config.robot.lower() == "go2":
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    @staticmethod
    def _release_high_level_mode(client_type) -> None:
        client = client_type()
        client.SetTimeout(5.0)
        client.Init()
        for _ in range(10):
            status, result = client.CheckMode()
            if status != 0 or not (result or {}).get("name"):
                return
            client.ReleaseMode()
            time.sleep(0.2)
        raise RuntimeError("Unable to release Unitree high-level motion mode")

    def configure_writer(self, publish_rate_hz: float, state_timeout_s: float = 0.1) -> None:
        if publish_rate_hz <= 0 or state_timeout_s <= 0:
            raise ValueError("Writer rate and state timeout must be positive")
        self._publish_rate_hz = float(publish_rate_hz)
        self._state_timeout_s = float(state_timeout_s)

    def start_command_writer(self) -> None:
        self._writer_enabled.set()

    def stop_command_writer(self) -> None:
        self._writer_enabled.clear()
        with self._command_lock:
            self._latest_command = None

    def _low_state_handler(self, message) -> None:
        try:
            if message.crc and state_crc(message) != message.crc:
                logger.warning("Discarding Unitree LowState with invalid CRC")
                return
        except (AttributeError, struct.error, TypeError, ValueError):
            logger.exception("Unable to validate Unitree LowState CRC")
            return
        with self._state_lock:
            self._latest_state = message
            self._latest_state_at = time.monotonic()
            self._mode_machine = int(message.mode_machine)

    def _state_snapshot(self):
        with self._state_lock:
            return self._latest_state, self._latest_state_at, self._mode_machine

    def get_low_state(self) -> np.ndarray | None:
        state, received_at, _ = self._state_snapshot()
        if state is None or time.monotonic() - received_at > self._state_timeout_s:
            return None
        base_pos = np.zeros(3, dtype=np.float32)
        quat = np.asarray(state.imu_state.quaternion, dtype=np.float32)
        base_lin_vel = np.zeros(3, dtype=np.float32)
        base_ang_vel = np.asarray(state.imu_state.gyroscope, dtype=np.float32)
        joint_pos = np.zeros(self.robot_config.num_joints, dtype=np.float32)
        joint_vel = np.zeros(self.robot_config.num_joints, dtype=np.float32)
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        for joint_id, motor_id in enumerate(motor_order[: self.robot_config.num_joints]):
            joint_pos[joint_id] = state.motor_state[motor_id].q
            joint_vel[joint_id] = state.motor_state[motor_id].dq
        return np.concatenate((base_pos, quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel)).reshape(1, -1)

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray | None = None,
        kp_override: np.ndarray | None = None,
        kd_override: np.ndarray | None = None,
    ) -> None:
        del dof_pos_latest
        count = self.robot_config.num_motors
        q = np.zeros(count, dtype=np.float32)
        dq = np.zeros(count, dtype=np.float32)
        tau = np.zeros(count, dtype=np.float32)
        kp = np.zeros(count, dtype=np.float32)
        kd = np.zeros(count, dtype=np.float32)
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        source_kp = np.asarray(kp_override if kp_override is not None else self.robot_config.motor_kp, dtype=np.float32)
        source_kd = np.asarray(kd_override if kd_override is not None else self.robot_config.motor_kd, dtype=np.float32)
        for joint_id, motor_id in enumerate(motor_order[: self.robot_config.num_joints]):
            q[motor_id] = cmd_q[joint_id]
            dq[motor_id] = cmd_dq[joint_id]
            tau[motor_id] = cmd_tau[joint_id]
            kp[motor_id] = source_kp[joint_id] * self._kp_level
            kd[motor_id] = source_kd[joint_id] * self._kd_level
        snapshot = _CommandSnapshot(q.copy(), dq.copy(), tau.copy(), kp.copy(), kd.copy())
        with self._command_lock:
            self._latest_command = snapshot

    def _write_latest_command(self) -> bool:
        with self._command_lock:
            snapshot = self._latest_command
        state, received_at, mode_machine = self._state_snapshot()
        if snapshot is None or state is None or time.monotonic() - received_at > self._state_timeout_s:
            return False
        command = self._low_command_factory()
        command.mode_pr = int(self.robot_config.unitree_legged_const.get("MODE_PR", 0))
        command.mode_machine = mode_machine
        for motor_id in range(self.robot_config.num_motors):
            motor = command.motor_cmd[motor_id]
            motor.mode = 1
            motor.q = float(snapshot.q[motor_id])
            motor.dq = float(snapshot.dq[motor_id])
            motor.tau = float(snapshot.tau[motor_id])
            motor.kp = float(snapshot.kp[motor_id])
            motor.kd = float(snapshot.kd[motor_id])
        command.crc = command_crc(command)
        self._publisher.Write(command)
        return True

    def _command_writer_loop(self) -> None:
        next_deadline = time.perf_counter()
        while not self._writer_stop.is_set():
            if not self._writer_enabled.wait(timeout=0.05):
                next_deadline = time.perf_counter()
                continue
            self._write_latest_command()
            next_deadline += 1.0 / self._publish_rate_hz
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                self._writer_stop.wait(delay)
            else:
                next_deadline = time.perf_counter()

    def get_joystick_msg(self):
        state, _, _ = self._state_snapshot()
        if state is None or not self.use_joystick:
            return None
        remote = bytes(state.wireless_remote)
        return SimpleNamespace(
            lx=struct.unpack("<f", remote[4:8])[0],
            rx=struct.unpack("<f", remote[8:12])[0],
            ly=struct.unpack("<f", remote[20:24])[0],
            keys=int(remote[2]) | (int(remote[3]) << 8),
        )

    def get_joystick_key(self, wc_msg=None):
        wc_msg = self.get_joystick_msg() if wc_msg is None else wc_msg
        return None if wc_msg is None else self._wc_key_map.get(getattr(wc_msg, "keys", 0))

    def get_raw_motor_state(self) -> dict | None:
        state, _, _ = self._state_snapshot()
        if state is None:
            return None
        return {
            "q": [motor.q for motor in state.motor_state],
            "dq": [motor.dq for motor in state.motor_state],
            "tau_est": [motor.tau_est for motor in state.motor_state],
            "temperature": [list(motor.temperature) for motor in state.motor_state],
            "imu_quat": list(state.imu_state.quaternion),
            "imu_omega": list(state.imu_state.gyroscope),
            "imu_accel": list(state.imu_state.accelerometer),
        }

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = float(value)

    @property
    def kd_level(self):
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        self._kd_level = float(value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_command_writer()
        self._writer_stop.set()
        self._writer_thread.join(timeout=1.0)
