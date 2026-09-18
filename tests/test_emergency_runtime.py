import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from vex_policy.config import ResolvedPolicy, load_runtime_config, resolve_policies
from vex_policy.config.config_types import (
    InferenceConfig,
    PolicySpec,
    RobotRuntimeConfig,
    RuntimeConfig,
)
from vex_policy.policies.base import BasePolicy, PolicyRuntimeFault
from vex_policy.policies.policy_state_machine import PolicyStateMachine
from vex_policy.policies.utils.joint_command import position_command
from vex_policy.robots import G1_29DOF
from vex_policy.sdk.base.base_interface import LowState

JOINT = G1_29DOF.dof_names[0]


def _stop(target=None, **extra):
    data = {"joints": {JOINT: {"pos": {"min": -0.2, "max": 0.2}}}}
    if target is not None:
        data["fallback"] = {"policy": target}
    return data | extra


def _spec(name, **extra):
    return PolicySpec.model_validate(
        {
            "name": name,
            "implementation": "hold_position",
            "inputs": [],
            "observation": {"obs_dict": {}, "obs_dims": {}, "obs_scales": {}, "history_length_dict": {}},
            "task": {},
            **extra,
        }
    )


def _slider(name, default):
    return {"type": "slider", "parameter": {"name": name, "min": 0, "max": 1, "default": default}}


class Probe(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.events = []
        self.controls = []
        self.failure = None

    def _on_activate(self, state):
        self.events.append("activate")
        return None

    def _on_deactivate(self):
        self.events.append("deactivate")

    def _apply_control(self, inputs):
        self.controls.append(dict(inputs))

    def _compute_command(self, state):
        self.events.append("step")
        if self.failure:
            raise self.failure
        return position_command(np.zeros(29), np.ones(29), np.ones(29), self.controlled_joint_mask)


@pytest.fixture
def runtime_factory():
    machines = []

    def make(*specs):
        runtime = RuntimeConfig(robot=RobotRuntimeConfig(config=G1_29DOF), policies=specs)
        resolved = tuple(
            ResolvedPolicy(
                spec,
                InferenceConfig(
                    robot=G1_29DOF,
                    inputs=spec.inputs,
                    observation=spec.observation,
                    task=spec.task,
                    limiter=spec.limiter,
                    estop=spec.estop,
                ),
                spec.implementation,
            )
            for spec in specs
        )
        policies = {item.spec.name: Probe(item.config) for item in resolved}
        robot_state = LowState(
            base_pos=np.zeros((1, 3)),
            base_quat=np.array([[1.0, 0, 0, 0]]),
            joint_pos=np.zeros((1, 29)),
            joint_vel=np.zeros((1, 29)),
            base_lin_vel=np.zeros((1, 3)),
            base_ang_vel=np.zeros((1, 3)),
        )
        h = SimpleNamespace(time=0.0, robot_state=robot_state, writes=[], statuses=[], states=[], reads=0, seq=0)

        def read():
            h.reads += 1
            return h.robot_state

        transport = SimpleNamespace(
            close=lambda: None,
            publish_status=lambda value: h.statuses.append(value),
            publish_state=lambda value: h.states.append(value),
            publish_reference_state=lambda value: None,
        )
        machine = PolicyStateMachine(
            runtime,
            resolved,
            instances=policies,
            clock=lambda: h.time,
            transport=transport,
            interface_manager=SimpleNamespace(
                get_low_state=read, send_low_command=lambda *args, **kw: h.writes.append(args)
            ),
        )
        machines.append(machine)
        h.machine, h.policies = machine, policies

        def send(*names, estop=False, inputs=None):
            h.seq += 1
            controls = (
                inputs
                if inputs is not None
                else {name: {p.name: p.default for p in machine._specs[name].input_parameters} for name in names}
            )
            assert machine.inbox.accept(
                json.dumps(
                    {
                        "seq": h.seq,
                        "timestamp": 0,
                        "control": {"policy": list(names), "inputs": controls, "estop": estop},
                    }
                )
            )
            machine.tick()

        h.send = send
        h.send()  # Clear the existing startup latch.
        return h

    yield make
    for machine in machines:
        machine._policy_executor.shutdown(wait=True)
        machine._close_policies(machine.policies.values())


def test_preflight_rejects_before_activation_and_requires_empty_rearm(runtime_factory):
    h = runtime_factory(_spec("source", estop=_stop()))
    h.robot_state.joint_pos[0, 0] = 0.3
    h.send("source")
    assert h.machine.state == "latched"
    assert JOINT in h.machine.reason
    assert not h.policies["source"].events
    assert not h.writes
    h.robot_state.joint_pos.fill(0)
    h.send("source")
    assert h.machine.state == "latched"
    h.send()
    h.send("source")
    assert h.machine.state == "running"
    assert h.policies["source"].events == ["activate"]


def test_group_stop_uses_one_snapshot_before_any_control_or_inference(runtime_factory):
    h = runtime_factory(_spec("lower", type="lower_body", estop=_stop()), _spec("upper", type="upper_body"))
    h.send("lower", "upper")
    h.send("lower", "upper")
    assert len(h.writes) == 1
    reads = h.reads
    h.robot_state.joint_pos[0, 0] = 0.3
    h.send("lower", "upper")
    assert h.reads == reads + 1
    assert len(h.writes) == 1
    assert h.machine.active_policy == ()
    for policy in h.policies.values():
        assert policy.events == ["activate", "step", "deactivate"]
        assert len(policy.controls) == 1


def test_fallback_inputs_latch_and_empty_release(runtime_factory):
    h = runtime_factory(
        _spec("source", estop=_stop("recovery", fallback={"policy": "recovery", "inputs": {"height": 0.4}})),
        _spec("recovery", inputs=[_slider("height", 0.1), _slider("speed", 0.2)]),
        _spec("other"),
    )
    h.send("source")
    h.robot_state.joint_pos[0, 0] = 0.3
    h.send("source")
    assert h.machine.state == "fallback"
    assert h.machine.active_policy == ("recovery",)
    assert h.machine.requested_policy == ("source",)
    reason = h.machine.reason
    assert not h.writes  # Transition gap; neither source nor fallback inferred.
    for selection in ("source", "other", "recovery"):
        h.send(selection)
        assert h.machine.state == "fallback"
        assert h.machine.reason == reason
        assert h.policies["recovery"].controls[-1] == {"height": 0.4, "speed": 0.2}
    assert len(h.writes) == 3
    assert h.policies["source"].events == ["activate", "deactivate"]
    h.send()
    assert h.machine.state == "idle"
    assert not h.machine.active_policy
    h.robot_state.joint_pos.fill(0)
    h.send("source")
    assert h.machine.state == "running"
    assert h.machine.reason is None


@pytest.mark.parametrize("failure", ["guard", "estop", "activate", "step", "step_fault", "later_estop"])
def test_fallback_failure_never_chains(runtime_factory, failure):
    recovery_stop = _stop("third", joints={JOINT: {"pos": {"min": -0.5, "max": 0.5}}})
    h = runtime_factory(
        _spec("source", estop=_stop("recovery")), _spec("recovery", estop=recovery_stop), _spec("third")
    )
    recovery = h.policies["recovery"]
    if failure == "guard":
        recovery.guard = SimpleNamespace(start_check=lambda _: (False, "bad_initial_pose"))
    elif failure == "activate":

        def fail(_):
            raise ValueError("activation broke")

        recovery._on_activate = fail
    elif failure in ("step", "step_fault"):
        recovery.failure = ValueError("step broke") if failure == "step" else PolicyRuntimeFault("step broke")
    h.send("source")
    h.robot_state.joint_pos[0, 0] = 0.6 if failure == "estop" else 0.3
    h.send("source")
    if failure in ("step", "step_fault", "later_estop"):
        assert h.machine.state == "fallback"
        if failure == "later_estop":
            h.robot_state.joint_pos[0, 0] = 0.6
        h.send("source")
    assert h.machine.state == "latched"
    assert not h.machine.active_policy
    assert "source" in h.machine.reason
    assert not h.policies["third"].events
    assert not h.writes


@pytest.mark.parametrize("action", ["estop", "timeout", "missing_state"])
def test_external_failures_override_fallback(runtime_factory, action):
    h = runtime_factory(_spec("source", estop=_stop("recovery")), _spec("recovery"))
    h.robot_state.joint_pos[0, 0] = 0.3
    h.send("source")
    assert h.machine.state == "fallback"
    if action == "estop":
        h.send("source", estop=True)
    elif action == "timeout":
        h.time = 2
        h.machine.tick()
    else:
        h.robot_state = None
        h.send("source")
    assert h.machine.state == "latched"
    assert not h.policies["recovery"].is_active
    assert not h.writes


@pytest.mark.parametrize("response", ["stop", "same", "different_target", "different_inputs"])
def test_parallel_responses(runtime_factory, response):
    first = _stop("recovery")
    second = {
        "stop": _stop(),
        "same": _stop("recovery"),
        "different_target": _stop("third"),
        "different_inputs": _stop("recovery", fallback={"policy": "recovery", "inputs": {"speed": 0.7}}),
    }[response]
    h = runtime_factory(
        _spec("lower", type="lower_body", estop=first),
        _spec("upper", type="upper_body", estop=second),
        _spec("recovery", inputs=[_slider("speed", 0.2)]),
        _spec("third"),
    )
    h.send("lower", "upper")
    h.robot_state.joint_pos[0, 0] = 0.3
    h.send("lower", "upper")
    assert h.machine.state == ("fallback" if response == "same" else "latched")
    assert not h.policies["lower"].is_active
    assert not h.policies["upper"].is_active
    assert not h.writes
    if response.startswith("different"):
        assert "fallback_conflict" in h.machine.reason


def test_invalid_state_never_falls_back_or_breaks_status_telemetry(runtime_factory):
    h = runtime_factory(_spec("source", estop=_stop("recovery")), _spec("recovery"))
    h.robot_state.joint_pos[0, 0] = np.nan
    h.time = 0.1
    h.send("source")
    assert h.machine.state == "latched"
    assert "invalid_state" in h.machine.reason
    assert not h.policies["recovery"].events
    assert h.statuses[-1]["state"] == "latched"
    assert all(np.isfinite(json.loads(payload)["joint_values"]).all() for payload in h.states)


@pytest.mark.parametrize(
    "change", ["unknown_joint", "unknown_target", "self", "half_body", "input_name", "input_value"]
)
def test_invalid_cross_policy_config(change):
    source = _spec("source", estop=_stop("target"))
    target = _spec("target", inputs=[_slider("speed", 0.2)])
    data = source.model_dump()
    if change == "unknown_joint":
        data["limiter"] = {"joints": {"typo": {"vel": 1}}}
    elif change == "unknown_target":
        data["estop"]["fallback"]["policy"] = "missing"
    elif change == "self":
        data["estop"]["fallback"]["policy"] = "source"
    elif change == "half_body":
        target = target.model_copy(update={"type": "upper_body"})
    else:
        data["estop"]["fallback"]["inputs"] = {"typo": 0.2} if change == "input_name" else {"speed": 2}
    with pytest.raises(ValueError):
        RuntimeConfig(robot=RobotRuntimeConfig(config=G1_29DOF), policies=(PolicySpec.model_validate(data), target))


def test_yaml_loader_propagates_limits_and_fallback(tmp_path):
    source = _spec("source", limiter={"joints": {JOINT: {"vel": 2}}}, estop=_stop("target"))
    for spec in (source, _spec("target")):
        (tmp_path / f"{spec.name}.yaml").write_text(yaml.safe_dump(spec.model_dump(mode="json")))
    mqtt = Path(__file__).resolve().parents[1] / "configs/mqtt.yaml"
    runtime, path = load_runtime_config(tmp_path, mqtt)
    resolved = resolve_policies(runtime, path)
    config = next(item.config for item in resolved if item.spec.name == "source")
    assert config.limiter.joints[JOINT].vel == 2
    assert config.estop.fallback.policy == "target"


def test_example_configs_load_together():
    root = Path(__file__).resolve().parents[1]
    runtime, path = load_runtime_config(root / "configs/examples/safety", root / "configs/mqtt.yaml")
    resolved = resolve_policies(runtime, path)
    assert {item.spec.name for item in resolved} == {"limited-hold", "recovery-hold"}
    config = next(item.config for item in resolved if item.spec.name == "limited-hold")
    assert config.estop.fallback.policy == "recovery-hold"
    assert config.limiter.joints["left_knee_joint"].vel == 10
