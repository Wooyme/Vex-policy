import json

import numpy as np

from vex_policy.config import load_runtime_config
from vex_policy.mqtt import CommandInbox, encode_robot_state, parse_broker


def packet(*, policy=None, estop=False, vx=0.5):
    return json.dumps(
        {
            "seq": 7,
            "timestamp": 1_750_000_000_000,
            "control": {
                "vx": vx,
                "vy": -0.2,
                "yaw": 0.1,
                "pitch": 0.0,
                "height": 0.0,
                "policy": [] if policy is None else policy,
                "estop": estop,
            },
        }
    )


def test_inbox_accepts_only_strict_known_policies():
    inbox = CommandInbox(["walk"], clock=lambda: 12.5)
    assert inbox.accept(packet(policy=["walk"]))
    assert inbox.snapshot().received_at == 12.5
    assert inbox.snapshot().packet.control.vx == 0.5
    assert not inbox.accept(packet(policy=["missing"]))
    assert not inbox.accept(packet(policy=["walk", "missing"]))
    assert not inbox.accept(packet(policy=["walk"], vx=2.0))
    assert inbox.invalid_messages == 3
    assert inbox.snapshot().packet.seq == 7


def test_inbox_accepts_lower_upper_pair_and_rejects_invalid_compositions():
    runtime, _ = load_runtime_config()
    lower = next(policy for policy in runtime.policies if policy.type == "lower_body")
    full = next(policy for policy in runtime.policies if policy.type == "full_body")
    upper = lower.model_copy(update={"name": "arms", "type": "upper_body"})
    inbox = CommandInbox({policy.name: policy for policy in (lower, upper, full)})

    assert inbox.accept(packet(policy=[upper.name, lower.name]))
    assert not inbox.accept(packet(policy=[full.name, lower.name]))
    assert not inbox.accept(packet(policy=[lower.name, lower.name]))
    assert not inbox.accept(packet(policy=[lower.name, upper.name, full.name]))


def test_accepted_input_filter_zeros_unadvertised_fields():
    inbox = CommandInbox(["walk"])
    assert inbox.accept(packet(policy=["walk"]))
    control = inbox.snapshot().packet.control.values_for(["vx", "yaw"])
    assert control.vx == 0.5
    assert control.vy == 0.0
    assert control.yaw == 0.1
    assert control.pitch == 0.0
    assert control.height == 0.0


def test_state_encoder_matches_simulator_shape():
    state = np.arange(7 + 29 + 6 + 29, dtype=float).reshape(1, -1)
    names = [f"joint_{index}" for index in range(29)]
    payload = json.loads(encode_robot_state(state, names, started_at=10.0, monotonic_now=12.5, timestamp=100.25))
    assert payload == {
        "timestamp": 100.25,
        "simulation_time": 2.5,
        "joint_names": names,
        "joint_values": state[0, 7:36].tolist(),
        "base_xyz": [0.0, 1.0, 2.0],
        "base_quat_wxyz": [3.0, 4.0, 5.0, 6.0],
    }


def test_parse_broker_supports_credentials_and_tls():
    endpoint = parse_broker("mqtts://user:pass@example.test")
    assert (endpoint.host, endpoint.port, endpoint.tls) == ("example.test", 8883, True)
    assert (endpoint.username, endpoint.password) == ("user", "pass")
