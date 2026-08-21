"""Strict YAML loading and policy configuration resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from vex_policy.config.config_types import (
    InferenceConfig,
    MqttConfig,
    PolicySpec,
    RobotRuntimeConfig,
    RuntimeConfig,
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
            policies.append(PolicySpec.model_validate(yaml.safe_load(stream)))

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
            path_fields.extend(("encoder_model_path", "planner_model_path"))
        resolved_paths: dict[str, str] = {}
        for field_name in path_fields:
            model_path = getattr(spec.task, field_name, None)
            if not model_path:
                raise ValueError(f"Policy {spec.name!r} {field_name} must be configured")
            if "://" in model_path:
                raise ValueError(f"Policy {spec.name!r} {field_name} must be a local file path")
            candidate = Path(model_path).expanduser().resolve()
            if not candidate.is_file():
                raise ValueError(f"Policy {spec.name!r} model file does not exist: {candidate}")
            resolved_paths[field_name] = str(candidate)
        task = replace(spec.task, **resolved_paths)
        policy_config = InferenceConfig(runtime.robot.config, spec.observation, task)
        rates.add(policy_config.task.rl_rate)
        resolved.append(ResolvedPolicy(spec, policy_config, spec.implementation))

    if len(rates) != 1:
        raise ValueError(f"All policies must use one control rate, got {sorted(rates)}")
    rate = next(iter(rates))
    if runtime.mqtt.state_frequency_hz > rate:
        raise ValueError("mqtt.state_frequency_hz must not exceed the policy control rate")
    return tuple(resolved)
