import threading
import time
from types import SimpleNamespace

import numpy as np

from vex_policy.sdk.unitree.unitree_interface import (
    UnitreeInterface,
    _CommandSnapshot,
    command_crc,
)


def _motor_command():
    return SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0)


def _low_command():
    return SimpleNamespace(
        mode_pr=0,
        mode_machine=0,
        motor_cmd=[_motor_command() for _ in range(35)],
        reserve=[0, 0, 0, 0],
        crc=0,
    )


def test_pure_python_hg_crc_known_vectors():
    command = _low_command()
    assert command_crc(command) == 4262932383
    command.mode_machine = 5
    for index, motor in enumerate(command.motor_cmd[:29]):
        motor.mode = 1
        motor.q = index / 10
        motor.dq = -index / 20
        motor.tau = 0.1
        motor.kp = 20 + index
        motor.kd = 1.5
    assert command_crc(command) == 3564284098


def test_writer_fills_modes_motors_crc_and_honors_freshness():
    interface = object.__new__(UnitreeInterface)
    interface.robot_config = SimpleNamespace(num_motors=29, unitree_legged_const={"MODE_PR": 0})
    interface._command_lock = threading.Lock()
    interface._state_lock = threading.Lock()
    interface._state_timeout_s = 0.1
    interface._latest_state = object()
    interface._latest_state_at = time.monotonic()
    interface._mode_machine = 5
    interface._low_command_factory = _low_command
    writes = []
    interface._publisher = SimpleNamespace(Write=writes.append)
    values = np.arange(29, dtype=np.float32)
    interface._latest_command = _CommandSnapshot(values, -values, values / 10, values + 1, values + 2)

    assert interface._write_latest_command()
    assert len(writes) == 1
    command = writes[0]
    assert command.mode_pr == 0
    assert command.mode_machine == 5
    assert command.motor_cmd[10].q == 10
    assert command.motor_cmd[10].dq == -10
    assert command.crc == command_crc(command)

    interface._latest_state_at -= 1.0
    assert not interface._write_latest_command()
    assert len(writes) == 1
