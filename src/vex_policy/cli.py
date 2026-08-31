"""Command-line entry point."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from loguru import logger

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.policies.policy_state_machine import PolicyStateMachine
from vex_policy.sdk import HighFrequencyLogConfig, InterfaceManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MQTT-controlled Unitree policy inference")
    parser.add_argument(
        "--config", "--policy-config", dest="policy_config", type=Path, help="Policy YAML file or directory"
    )
    parser.add_argument("--mqtt-config", type=Path, help="MQTT YAML; defaults to packaged mqtt.yaml")
    parser.add_argument("--mqtt-broker", help="Override mqtt.broker")
    parser.add_argument("--interface", help="Override robot.interface")
    parser.add_argument("--domain-id", type=int, help="Override robot.domain_id")
    parser.add_argument(
        "--sdk-log-dir",
        type=Path,
        help="Enable asynchronous SDK state/command logging under this directory",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runtime, config_path = load_runtime_config(args.policy_config, args.mqtt_config)
    mqtt = runtime.mqtt
    robot = runtime.robot
    if args.mqtt_broker is not None:
        mqtt = mqtt.model_copy(update={"broker": args.mqtt_broker})
    if args.interface is not None:
        robot = robot.model_copy(update={"interface": args.interface})
    if args.domain_id is not None:
        robot = robot.model_copy(update={"domain_id": args.domain_id})
    if args.sdk_log_dir is not None:
        robot_config = replace(
            robot.config,
            high_frequency_log=HighFrequencyLogConfig(directory=args.sdk_log_dir),
        )
        robot = robot.model_copy(update={"config": robot_config})
    runtime = runtime.model_copy(update={"mqtt": mqtt, "robot": robot})
    resolved = resolve_policies(runtime, config_path)
    logger.info(f"Loaded {len(resolved)} policies from {config_path}")
    interface_manager = InterfaceManager.initialize(
        runtime.robot.config,
        runtime.robot.domain_id,
        runtime.robot.interface,
        False,
    )
    try:
        PolicyStateMachine(runtime, resolved, interface_manager=interface_manager).run()
    finally:
        InterfaceManager.close()


if __name__ == "__main__":
    main()
