from dataclasses import asdict, fields
from pathlib import Path

import pytest
import yaml

from vex_policy.config import default_mqtt_config_path, load_runtime_config, resolve_policies
from vex_policy.config.config_types import (
    ActionMaskConfig,
    DebugConfig,
    MqttConfig,
    ObservationConfig,
    PolicySpec,
    RuntimeConfig,
)
from vex_policy.robots import G1_29DOF


def test_packaged_config_resolves_all_g1_policies():
    runtime, path = load_runtime_config()
    resolved = resolve_policies(runtime, path)
    assert {item.spec.name: item.kind for item in resolved} == {
        "g1-sonic-slow-walk": "sonic",
        "g1-ppo-locomotion": "locomotion",
        "g1-wbt-example": "wbt",
        "sonic-doggy1": "sonic",
        "sonic-knee-down": "sonic",
        "g1-doggy1": "wbt",
    }
    assert all(Path(item.config.task.model_path).is_file() for item in resolved)
    locomotion = next(item for item in resolved if item.kind == "locomotion")
    assert locomotion.spec.type == "lower_body"
    assert locomotion.config.action_mask is not None
    assert set(locomotion.config.action_mask.masked_joints) == set(G1_29DOF.dof_names_upper_body)


def test_single_policy_yaml_loads_only_that_policy():
    _, directory = load_runtime_config()
    runtime, path = load_runtime_config(directory / "ppo_locomotion.yaml")
    assert path.name == "ppo_locomotion.yaml"
    assert [policy.name for policy in runtime.policies] == ["g1-ppo-locomotion"]


def test_packaged_policy_and_mqtt_yaml_expose_every_field():
    _, path = load_runtime_config()
    policy_paths = tuple(path.glob("*.yaml"))
    assert len(policy_paths) == 6
    for policy_path in policy_paths:
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        assert set(policy) <= set(PolicySpec.model_fields)
        assert {"name", "implementation", "type", "inputs", "observation", "task"} <= set(policy)
        assert set(policy["observation"]) == {field.name for field in fields(ObservationConfig)}
        task_type = type(PolicySpec.model_validate(policy).task)
        assert set(policy["task"]) <= {field.name for field in fields(task_type)}
        assert "action_mask_path" in policy["task"]
        assert set(policy["task"]["debug"]) == {field.name for field in fields(DebugConfig)}
    mqtt_data = yaml.safe_load(default_mqtt_config_path().read_text(encoding="utf-8"))
    assert set(mqtt_data) == set(MqttConfig.model_fields)
    assert G1_29DOF.robot_type == "g1_29dof"


def test_mqtt_config_is_loaded_from_its_own_file(tmp_path):
    data = yaml.safe_load(default_mqtt_config_path().read_text(encoding="utf-8"))
    data["broker"] = "mqtt://broker.internal:1884"
    mqtt_path = tmp_path / "mqtt.yaml"
    mqtt_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    runtime, _ = load_runtime_config(mqtt_path=mqtt_path)
    assert runtime.mqtt.broker == "mqtt://broker.internal:1884"


def test_config_rejects_duplicate_policy_names():
    runtime, path = load_runtime_config()
    base = yaml.safe_load(next(path.glob("*.yaml")).read_text(encoding="utf-8"))
    duplicate = PolicySpec.model_validate(base)
    with pytest.raises(ValueError, match="unique"):
        RuntimeConfig(robot=runtime.robot, mqtt=runtime.mqtt, policies=(duplicate, duplicate))


def test_model_override_accepts_absolute_path(tmp_path):
    model = tmp_path / "custom.onnx"
    model.touch()
    _, default_path = load_runtime_config()
    data = yaml.safe_load((default_path / "ppo_locomotion.yaml").read_text(encoding="utf-8"))
    data["name"] = "custom"
    data["task"]["model_path"] = str(model)
    data["task"]["action_mask_path"] = str((default_path / "action_masks/disable_upper_body.yaml").resolve())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    runtime, loaded_path = load_runtime_config(config_path)
    resolved = resolve_policies(runtime, loaded_path)
    assert resolved[0].config.task.model_path == str(model)


@pytest.mark.parametrize("model_path", ["https://example.test/policy.onnx", "wandb://team/project/run/policy.onnx"])
def test_model_path_rejects_remote_sources(model_path):
    runtime, path = load_runtime_config()
    policy = next(policy for policy in runtime.policies if policy.implementation == "locomotion")
    task = asdict(policy.task)
    task["model_path"] = model_path
    data = yaml.safe_load((path / "ppo_locomotion.yaml").read_text(encoding="utf-8"))
    data["task"] = task
    configured_policy = PolicySpec.model_validate(data)
    configured = RuntimeConfig(robot=runtime.robot, mqtt=runtime.mqtt, policies=(configured_policy,))
    with pytest.raises(ValueError, match="local file"):
        resolve_policies(configured, path)


def test_nested_yaml_rejects_unknown_fields():
    _, path = load_runtime_config()
    data = yaml.safe_load((path / "ppo_locomotion.yaml").read_text(encoding="utf-8"))
    data["task"]["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        PolicySpec.model_validate(data)


def test_action_mask_path_is_relative_to_policy_yaml(tmp_path):
    _, default_path = load_runtime_config()
    data = yaml.safe_load((default_path / "ppo_locomotion.yaml").read_text(encoding="utf-8"))
    model_path = Path(data["task"]["model_path"]).resolve()
    data["task"]["model_path"] = str(model_path)
    data["task"]["action_mask_path"] = "masks/upper.yaml"
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "upper.yaml").write_text(
        yaml.safe_dump({"masked_joints": list(G1_29DOF.dof_names_upper_body)}), encoding="utf-8"
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    runtime, loaded_path = load_runtime_config(policy_path)
    resolved = resolve_policies(runtime, loaded_path)[0]

    assert resolved.config.action_mask == ActionMaskConfig(masked_joints=G1_29DOF.dof_names_upper_body)


@pytest.mark.parametrize(
    ("masked_joints", "message"),
    [
        (("not_a_joint",), "unknown joints"),
        ((G1_29DOF.dof_names_upper_body[0],) * 2, "duplicate"),
        (G1_29DOF.dof_names, "at least one"),
        ((), "controls upper-body"),
    ],
)
def test_invalid_action_masks_are_rejected(tmp_path, masked_joints, message):
    _, default_path = load_runtime_config()
    data = yaml.safe_load((default_path / "ppo_locomotion.yaml").read_text(encoding="utf-8"))
    data["task"]["model_path"] = str(Path(data["task"]["model_path"]).resolve())
    data["task"]["action_mask_path"] = "mask.yaml"
    (tmp_path / "mask.yaml").write_text(yaml.safe_dump({"masked_joints": list(masked_joints)}), encoding="utf-8")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    runtime, loaded_path = load_runtime_config(policy_path)
    with pytest.raises(ValueError, match=message):
        resolve_policies(runtime, loaded_path)
