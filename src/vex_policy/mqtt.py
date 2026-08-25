"""MQTT command ingress and robot-state/status egress."""

from __future__ import annotations

import json
import os
import socket
import ssl
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import unquote, urlsplit

import numpy as np
import paho.mqtt.client as mqtt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vex_policy.config.config_types import MqttConfig, PolicySpec

InputValue = Annotated[float, Field(allow_inf_nan=False)]


class CommandControl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    policy: tuple[str, ...]
    inputs: dict[str, dict[str, InputValue]]
    estop: bool

    @field_validator("policy")
    @classmethod
    def valid_policy_count(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 2:
            raise ValueError("at most two active policies are supported")
        if len(set(value)) != len(value):
            raise ValueError("policy names must not repeat")
        return value


class ControlPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seq: int = Field(ge=0)
    timestamp: int = Field(ge=0)
    control: CommandControl


@dataclass(frozen=True)
class ReceivedCommand:
    packet: ControlPacket
    received_at: float


class CommandInbox:
    """Thread-safe newest-valid-command store."""

    def __init__(
        self,
        policies: dict[str, PolicySpec],
        clock: Callable[[], float] = time.monotonic,
    ):
        self._policy_specs = dict(policies)
        self._policy_names = frozenset(policies)
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: ReceivedCommand | None = None
        self.invalid_messages = 0

    def accept(self, payload: bytes | str) -> bool:
        try:
            packet = ControlPacket.model_validate_json(payload)
            unknown = set(packet.control.policy) - self._policy_names
            if unknown:
                raise ValueError(f"unknown policies: {sorted(unknown)}")
            selected = [self._policy_specs[name] for name in packet.control.policy]
            types = [policy.type for policy in selected]
            if "full_body" in types and len(types) > 1:
                raise ValueError("full_body policies must run alone")
            if len(types) != len(set(types)):
                raise ValueError("only one policy of each body type may be active")
            if set(packet.control.inputs) != set(packet.control.policy):
                raise ValueError("control input policy names must exactly match selected policies")
            for spec in selected:
                values = packet.control.inputs[spec.name]
                parameters = {parameter.name: parameter for parameter in spec.input_parameters}
                if set(values) != set(parameters):
                    raise ValueError(f"control inputs for {spec.name!r} must exactly match its parameters")
                for name, value in values.items():
                    parameter = parameters[name]
                    if not parameter.min <= value <= parameter.max:
                        raise ValueError(f"control input {spec.name}.{name} is outside its configured range")
        except (ValueError, TypeError):
            with self._lock:
                self.invalid_messages += 1
            return False
        received = ReceivedCommand(packet, self._clock())
        with self._lock:
            self._latest = received
        return True

    def snapshot(self) -> ReceivedCommand | None:
        with self._lock:
            return self._latest


@dataclass(frozen=True)
class BrokerEndpoint:
    host: str
    port: int
    tls: bool
    username: str | None = None
    password: str | None = None


def parse_broker(value: str) -> BrokerEndpoint:
    raw = value.strip()
    if not raw:
        raise ValueError("MQTT broker must not be empty")
    parsed = urlsplit(raw if "://" in raw else f"mqtt://{raw}")
    if parsed.scheme not in {"mqtt", "mqtts"} or not parsed.hostname:
        raise ValueError("MQTT broker must use mqtt:// or mqtts://")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("MQTT broker must not contain a path, query, or fragment")
    tls = parsed.scheme == "mqtts"
    return BrokerEndpoint(
        host=parsed.hostname,
        port=parsed.port or (8883 if tls else 1883),
        tls=tls,
        username=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else None,
    )


def encode_robot_state(
    robot_state: np.ndarray,
    joint_names: Iterable[str],
    *,
    started_at: float,
    monotonic_now: float | None = None,
    timestamp: float | None = None,
) -> str:
    """Encode the exact state shape consumed by the control panel and simulator."""
    names = list(joint_names)
    state = np.asarray(robot_state, dtype=np.float64).reshape(-1)
    required = 7 + len(names)
    if state.size < required:
        raise ValueError(f"robot state needs at least {required} values, got {state.size}")
    now = time.monotonic() if monotonic_now is None else monotonic_now
    payload = {
        "timestamp": time.time() if timestamp is None else timestamp,
        "simulation_time": max(0.0, now - started_at),
        "joint_names": names,
        "joint_values": state[7:required].tolist(),
        "base_xyz": state[:3].tolist(),
        "base_quat_wxyz": state[3:7].tolist(),
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


class MqttTransport:
    def __init__(self, config: MqttConfig, policies: tuple[PolicySpec, ...], inbox: CommandInbox):
        self.config = config
        self.policies = policies
        self.inbox = inbox
        self._connected = threading.Event()
        self._connect_error: str | None = None
        client_id = config.client_id or f"vex-policy-{socket.gethostname()}-{os.getpid()}"
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.will_set(
            config.status_topic,
            self._json(
                {
                    "state": "offline",
                    "active_policy": [],
                    "requested_policy": [],
                    "reason": "mqtt_disconnect",
                    "last_command_seq": None,
                }
            ),
            qos=1,
            retain=True,
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)

    def start(self) -> None:
        endpoint = parse_broker(self.config.broker)
        if endpoint.username is not None:
            self._client.username_pw_set(endpoint.username, endpoint.password)
        if endpoint.tls:
            self._client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self._client.connect_async(endpoint.host, endpoint.port, keepalive=30)
        self._client.loop_start()
        if not self._connected.wait(self.config.connect_timeout_s):
            self.close(graceful=False)
            detail = f": {self._connect_error}" if self._connect_error else ""
            raise RuntimeError(f"Timed out connecting to MQTT broker {endpoint.host}:{endpoint.port}{detail}")

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        del userdata, flags, properties
        if reason_code != 0:
            self._connect_error = str(reason_code)
            return
        client.subscribe(self.config.command_topic, qos=0)
        self._connected.set()
        self.publish_policies()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties) -> None:
        del client, userdata, disconnect_flags, properties
        self._connected.clear()
        if reason_code != 0:
            self._connect_error = str(reason_code)

    def _on_message(self, client, userdata, message) -> None:
        del client, userdata
        if message.topic == self.config.command_topic:
            self.inbox.accept(message.payload)

    def publish_policies(self) -> None:
        payload = [
            {
                "name": policy.name,
                "type": policy.type,
                "inputs": [component.model_dump(mode="json") for component in policy.inputs],
            }
            for policy in self.policies
        ]
        self._client.publish(self.config.policies_topic, self._json(payload), qos=1, retain=True)

    def publish_status(self, payload: dict[str, Any]) -> None:
        self._client.publish(self.config.status_topic, self._json(payload), qos=1, retain=True)

    def publish_state(self, payload: str) -> None:
        self._client.publish(self.config.state_topic, payload, qos=0, retain=False)

    def publish_reference_state(self, payload: str) -> None:
        self._client.publish(self.config.reference_state_topic, payload, qos=0, retain=False)

    def close(self, *, graceful: bool = True) -> None:
        client = getattr(self, "_client", None)
        if client is None:
            return
        if graceful and self._connected.is_set():
            self.publish_status(
                {
                    "state": "offline",
                    "active_policy": [],
                    "requested_policy": [],
                    "reason": "shutdown",
                    "last_command_seq": None,
                }
            )
        try:
            client.disconnect()
        finally:
            client.loop_stop()
