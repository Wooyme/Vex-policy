"""Strict YAML loading and policy configuration resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from vex_policy.config.config_types import (
    ActionMaskConfig,
    InferenceConfig,
    MqttConfig,
    PolicySpec,
    RobotRuntimeConfig,
    RuntimeConfig,
    SonicTaskConfig,
    WaistLocomotionGuardConfig,
    WaistLocomotionTaskConfig,
)
from vex_policy.robots import G1_29DOF


@dataclass(frozen=True)
class ResolvedPolicy:
    spec: PolicySpec
    config: InferenceConfig
    kind: str


def default_config_path() -> Path:
    return Path("configs/g1")


def default_mqtt_config_path() -> Path:
    return Path("configs/mqtt.yaml")


def load_runtime_config(
    path: str | Path | None = None,
    mqtt_path: str | Path | None = None,
) -> tuple[RuntimeConfig, Path]:
    config_path = Path(path).expanduser().resolve() if path else default_config_path()
    if config_path.is_dir():
        policy_paths = tuple(sorted(config_path.glob("*.yaml")))
        if not policy_paths:
            raise ValueError(f"Policy config directory contains no YAML files: {config_path}")
    elif config_path.is_file():
        policy_paths = (config_path,)
    else:
        raise ValueError(f"Policy config path does not exist: {config_path}")

    policies: list[PolicySpec] = []
    for policy_path in policy_paths:
        with policy_path.open(encoding="utf-8") as stream:
            spec = PolicySpec.model_validate(yaml.safe_load(stream))
        resolved_task_paths: dict[str, str] = {}
        if spec.task.action_mask_path:
            mask_path = Path(spec.task.action_mask_path).expanduser()
            if not mask_path.is_absolute():
                mask_path = policy_path.parent / mask_path
            resolved_task_paths["action_mask_path"] = str(mask_path.resolve())
        if resolved_task_paths:
            task = replace(spec.task, **resolved_task_paths)
            spec = spec.model_copy(update={"task": task})
        policies.append(spec)

    resolved_mqtt_path = Path(mqtt_path).expanduser().resolve() if mqtt_path else default_mqtt_config_path()
    with resolved_mqtt_path.open(encoding="utf-8") as stream:
        mqtt = MqttConfig.model_validate(yaml.safe_load(stream))

    runtime = RuntimeConfig(
        robot=RobotRuntimeConfig(config=G1_29DOF),
        mqtt=mqtt,
        policies=tuple(policies),
    )
    return runtime, config_path


def resolve_policies(runtime: RuntimeConfig, config_path: Path) -> tuple[ResolvedPolicy, ...]:
    resolved: list[ResolvedPolicy] = []
    rates: set[float] = set()
    for spec in runtime.policies:
        path_fields = ["model_path"]
        if spec.implementation == "sonic":
            if not isinstance(spec.task, SonicTaskConfig):
                raise ValueError(f"Policy {spec.name!r} requires SonicTaskConfig")
            path_fields.append("encoder_model_path")
            if spec.task.motion_source == "planner":
                path_fields.append("planner_model_path")
        elif spec.implementation == "waist_locomotion" and not isinstance(
            spec.task, WaistLocomotionTaskConfig
        ):
            raise ValueError(f"Policy {spec.name!r} requires WaistLocomotionTaskConfig")
        elif spec.implementation == "waist_locomotion" and not isinstance(
            spec.guard, WaistLocomotionGuardConfig
        ):
            raise ValueError(f"Policy {spec.name!r} requires WaistLocomotionGuardConfig")
        resolved_paths: dict[str, str] = {}
        for field_name in path_fields:
            model_path = getattr(spec.task, field_name, None)
            if not model_path:
                raise ValueError(f"Policy {spec.name!r} {field_name} must be configured")
            if "://" in str(model_path):
                raise ValueError(f"Policy {spec.name!r} {field_name} must reference a local file")
            candidate = Path(model_path).expanduser().resolve()
            if not candidate.is_file():
                raise ValueError(f"Policy {spec.name!r} model file does not exist: {candidate}")
            resolved_paths[field_name] = str(candidate)
        if spec.implementation == "sonic" and spec.task.motion_source == "directory":
            motion_data_path = spec.task.motion_data_path
            if not motion_data_path:
                raise ValueError(f"Policy {spec.name!r} motion_data_path must be configured for directory motion")
            motion_directory = Path(motion_data_path).expanduser().resolve()
            if not motion_directory.is_dir():
                raise ValueError(f"Policy {spec.name!r} motion directory does not exist: {motion_directory}")
            resolved_paths["motion_data_path"] = str(motion_directory)
        if isinstance(spec.task, WaistLocomotionTaskConfig):
            motion_data_path = Path(spec.task.motion_data_path).expanduser().resolve()
            if not motion_data_path.is_file():
                raise ValueError(f"Policy {spec.name!r} motion file does not exist: {motion_data_path}")
            resolved_paths["motion_data_path"] = str(motion_data_path)
        task = replace(spec.task, **resolved_paths)
        action_mask = _load_action_mask(spec, runtime.robot.config)
        policy_config = InferenceConfig(
            runtime.robot.config,
            spec.inputs,
            spec.observation,
            task,
            spec.guard,
            action_mask,
        )
        rates.add(policy_config.task.rl_rate)
        resolved.append(ResolvedPolicy(spec, policy_config, spec.implementation))

    if len(rates) != 1:
        raise ValueError(f"All policies must use one control rate, got {sorted(rates)}")
    rate = next(iter(rates))
    if runtime.mqtt.state_frequency_hz > rate:
        raise ValueError("mqtt.state_frequency_hz must not exceed the policy control rate")
    return tuple(resolved)


def _load_action_mask(spec: PolicySpec, robot) -> ActionMaskConfig | None:
    """Load a referenced mask and validate its joint ownership contract."""
    action_mask: ActionMaskConfig | None = None
    if spec.task.action_mask_path:
        mask_path = Path(spec.task.action_mask_path)
        if not mask_path.is_file():
            raise ValueError(f"Policy {spec.name!r} action mask file does not exist: {mask_path}")
        with mask_path.open(encoding="utf-8") as stream:
            action_mask = ActionMaskConfig(**yaml.safe_load(stream))

    masked_names = action_mask.masked_joints if action_mask is not None else ()
    if len(set(masked_names)) != len(masked_names):
        raise ValueError(f"Policy {spec.name!r} action mask contains duplicate joint names")

    known = set(robot.dof_names)
    unknown = sorted(set(masked_names) - known)
    if unknown:
        raise ValueError(f"Policy {spec.name!r} action mask contains unknown joints: {unknown}")

    controlled = known - set(masked_names)
    if not controlled:
        raise ValueError(f"Policy {spec.name!r} action mask must leave at least one joint controlled")

    if spec.type == "lower_body":
        invalid = sorted(controlled - set(robot.dof_names_lower_body))
        if invalid:
            raise ValueError(f"Lower-body policy {spec.name!r} controls upper-body joints: {invalid}")
    elif spec.type == "upper_body":
        invalid = sorted(controlled - set(robot.dof_names_upper_body))
        if invalid:
            raise ValueError(f"Upper-body policy {spec.name!r} controls lower-body joints: {invalid}")

    return action_mask
