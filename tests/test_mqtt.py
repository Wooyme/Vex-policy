import json

import numpy as np

from vex_policy.config import load_runtime_config
from vex_policy.mqtt import CommandInbox, encode_robot_state, parse_broker


def default_inputs(*specs):
    return {spec.name: {parameter.name: parameter.default for parameter in spec.input_parameters} for spec in specs}


def packet(*, policy=None, inputs=None, estop=False, seq=7):
    return json.dumps(
        {
            "seq": seq,
            "timestamp": 1_750_000_000_000,
            "control": {
                "policy": [] if policy is None else policy,
                "inputs": {} if inputs is None else inputs,
                "estop": estop,
            },
        }
    )


def test_inbox_accepts_only_strict_known_policies():
    runtime, _ = load_runtime_config()
    walk = next(policy for policy in runtime.policies if policy.implementation == "locomotion")
    values = default_inputs(walk)
    values[walk.name].update({"vx": 0.5, "vy": -0.2, "yaw": 0.1})
    inbox = CommandInbox({walk.name: walk}, clock=lambda: 12.5)

    assert inbox.accept(packet(policy=[walk.name], inputs=values))
    assert inbox.snapshot().received_at == 12.5
    assert inbox.snapshot().packet.control.inputs[walk.name]["vx"] == 0.5
    assert not inbox.accept(packet(policy=["missing"], inputs={"missing": {}}))
    assert not inbox.accept(packet(policy=[walk.name, "missing"], inputs=values | {"missing": {}}))
    values[walk.name]["vx"] = 2.0
    assert not inbox.accept(packet(policy=[walk.name], inputs=values, seq=8))
    values[walk.name]["vx"] = float("nan")
    assert not inbox.accept(packet(policy=[walk.name], inputs=values, seq=9))
    assert inbox.invalid_messages == 4
    assert inbox.snapshot().packet.seq == 7


def test_inbox_accepts_lower_upper_pair_and_rejects_invalid_compositions():
    runtime, _ = load_runtime_config()
    lower = next(policy for policy in runtime.policies if policy.type == "lower_body")
    full = next(policy for policy in runtime.policies if policy.type == "full_body")
    upper = lower.model_copy(update={"name": "arms", "type": "upper_body"})
    inbox = CommandInbox({policy.name: policy for policy in (lower, upper, full)})

    assert inbox.accept(packet(policy=[upper.name, lower.name], inputs=default_inputs(upper, lower)))
    assert not inbox.accept(packet(policy=[full.name, lower.name], inputs=default_inputs(full, lower)))
    assert not inbox.accept(packet(policy=[lower.name, lower.name], inputs=default_inputs(lower)))
    assert not inbox.accept(
        packet(policy=[lower.name, upper.name, full.name], inputs=default_inputs(lower, upper, full))
    )


def test_inputs_must_exactly_match_selected_policy_and_parameter_schema():
    runtime, _ = load_runtime_config()
    walk = next(policy for policy in runtime.policies if policy.implementation == "locomotion")
    inbox = CommandInbox({walk.name: walk})
    valid = default_inputs(walk)

    assert inbox.accept(packet(policy=[walk.name], inputs=valid))
    valid[walk.name]["vx"] = -1.0
    valid[walk.name]["vy"] = 1.0
    assert inbox.accept(packet(policy=[walk.name], inputs=valid))
    assert not inbox.accept(packet(policy=[walk.name], inputs={}))
    assert not inbox.accept(packet(policy=[], inputs=valid))
    assert not inbox.accept(packet(policy=[walk.name], inputs={walk.name: {"vx": 0.0}}))
    extra = default_inputs(walk)
    extra[walk.name]["extra"] = 0.0
    assert not inbox.accept(packet(policy=[walk.name], inputs=extra))


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
