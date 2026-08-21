"""Multiprocess proxy for UnitreeInterface.

Runs the real Python DDS UnitreeInterface in a spawned child process.
The proxy implements the same BaseInterface API via RPC-over-queues.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, NamedTuple

import numpy as np

from vex_policy.config.config_types import RobotConfig
from vex_policy.sdk.base.base_interface import BaseInterface


class JoystickMsg(NamedTuple):
    """Picklable wireless-controller snapshot."""

    lx: float
    ly: float
    rx: float
    keys: int


# Sentinel that tells the worker to shut down.
_STOP = None


# ── child process ──────────────────────────────────────────────────────


def _worker(
    robot_config: RobotConfig,
    domain_id: int,
    interface_str: str | None,
    use_joystick: bool,
    req_q: mp.Queue,
    res_q: mp.Queue,
):
    """Event loop that owns the real UnitreeInterface."""
    import os

    from vex_policy.sdk.unitree.unitree_interface import UnitreeInterface

    robot = UnitreeInterface(robot_config, domain_id, interface_str, use_joystick)

    try:
        while True:
            msg = req_q.get()
            if msg is _STOP:
                break

            method, args, kwargs = msg
            try:
                if method == "__get_kp_level":
                    res_q.put(("ok", robot.kp_level))
                elif method == "__set_kp_level":
                    robot.kp_level = args[0]
                    res_q.put(("ok", None))
                elif method == "__get_kd_level":
                    res_q.put(("ok", robot.kd_level))
                elif method == "__set_kd_level":
                    robot.kd_level = args[0]
                    res_q.put(("ok", None))
                elif method == "get_joystick_msg":
                    wc = robot.get_joystick_msg()
                    if wc is None:
                        res_q.put(("ok", None))
                    else:
                        res_q.put(
                            (
                                "ok",
                                JoystickMsg(
                                    lx=getattr(wc, "lx", 0.0),
                                    ly=getattr(wc, "ly", 0.0),
                                    rx=getattr(wc, "rx", 0.0),
                                    keys=getattr(wc, "keys", 0),
                                ),
                            )
                        )
                elif method == "get_raw_motor_state":
                    res_q.put(("ok", robot.get_raw_motor_state()))
                elif method == "update_config":
                    robot.update_config(*args, **kwargs)
                    res_q.put(("ok", None))
                elif method == "configure_writer":
                    robot.configure_writer(*args, **kwargs)
                    res_q.put(("ok", None))
                elif method == "start_command_writer":
                    robot.start_command_writer()
                    res_q.put(("ok", None))
                elif method == "stop_command_writer":
                    robot.stop_command_writer()
                    res_q.put(("ok", None))
                else:
                    result = getattr(robot, method)(*args, **kwargs)
                    res_q.put(("ok", result))
            except Exception as exc:
                res_q.put(("err", exc))
    finally:
        # The DDS runtime owns background resources. Exit the isolated worker
        # after dropping the interface so parent shutdown remains deterministic.
        del robot
        os._exit(0)


# ── parent-side proxy ──────────────────────────────────────────────────


class UnitreeInterfaceMP(BaseInterface):
    """Drop-in replacement for UnitreeInterface that runs in a child process."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str: str | None = None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)

        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker,
            args=(robot_config, domain_id, interface_str, use_joystick, self._req_q, self._res_q),
            daemon=True,
        )
        self._proc.start()

    # ── RPC helper ─────────────────────────────────────────────────────

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._req_q.put((method, args, kwargs))
        tag, payload = self._res_q.get()
        if tag == "err":
            raise payload
        return payload

    # ── BaseInterface implementation ───────────────────────────────────

    def get_low_state(self) -> np.ndarray:
        return self._call("get_low_state")

    def send_low_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None, kp_override=None, kd_override=None):
        return self._call("send_low_command", cmd_q, cmd_dq, cmd_tau, dof_pos_latest, kp_override, kd_override)

    def get_joystick_msg(self):
        return self._call("get_joystick_msg")

    def get_joystick_key(self, wc_msg=None):
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    def get_raw_motor_state(self) -> dict:
        """Get raw motor/IMU state as a dict (fields not in BaseInterface)."""
        return self._call("get_raw_motor_state")

    def update_config(self, robot_config: RobotConfig):
        super().update_config(robot_config)
        self._call("update_config", robot_config)

    def configure_writer(self, publish_rate_hz: float, state_timeout_s: float = 0.1):
        self._call("configure_writer", publish_rate_hz, state_timeout_s)

    def start_command_writer(self):
        self._call("start_command_writer")

    def stop_command_writer(self):
        self._call("stop_command_writer")

    @property
    def kp_level(self):
        return self._call("__get_kp_level")

    @kp_level.setter
    def kp_level(self, value):
        self._call("__set_kp_level", value)

    @property
    def kd_level(self):
        return self._call("__get_kd_level")

    @kd_level.setter
    def kd_level(self, value):
        self._call("__set_kd_level", value)

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self):
        self._req_q.put(_STOP)
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.is_alive():
            self.close()
