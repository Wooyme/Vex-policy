import json
import shutil
import socket
import subprocess
import time

import paho.mqtt.client as mqtt
import pytest

from vex_policy.config import load_runtime_config
from vex_policy.config.config_types import MqttConfig
from vex_policy.mqtt import CommandInbox, MqttTransport

pytestmark = pytest.mark.skipif(shutil.which("mosquitto") is None, reason="mosquitto is not installed")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def port_is_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.05):
            return True
    except OSError:
        return False


def test_real_broker_round_trip_and_retained_outputs(tmp_path):
    port = free_port()
    config_file = tmp_path / "mosquitto.conf"
    config_file.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\n", encoding="utf-8")
    process = subprocess.Popen(
        ["mosquitto", "-c", str(config_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    config = MqttConfig(broker=f"mqtt://127.0.0.1:{port}", connect_timeout_s=2)
    runtime, _ = load_runtime_config()
    policy = next(policy for policy in runtime.policies if policy.implementation == "locomotion")
    inbox = CommandInbox({policy.name: policy})
    transport = MqttTransport(config, (policy,), inbox)
    received = {}
    observer = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_message(client, userdata, message):
        del client, userdata
        received[message.topic] = message.payload.decode()

    observer.on_message = on_message
    try:
        wait_until(lambda: process.poll() is None and port_is_open(port))
        transport.start()
        observer.connect("127.0.0.1", port)
        observer.subscribe("robot/#")
        observer.loop_start()
        wait_until(lambda: config.policies_topic in received)
        assert json.loads(received[config.policies_topic]) == [
            {
                "name": policy.name,
                "type": policy.type,
                "inputs": [component.model_dump(mode="json") for component in policy.inputs],
            }
        ]

        observer.publish(
            config.command_topic,
            json.dumps(
                {
                    "seq": 1,
                    "timestamp": 1,
                    "control": {
                        "policy": [policy.name],
                        "inputs": {
                            policy.name: {
                                "vx": 0.2,
                                "vy": 0.0,
                                "yaw": 0.0,
                            }
                        },
                        "estop": False,
                    },
                }
            ),
        )
        wait_until(lambda: inbox.snapshot() is not None)
        transport.publish_status(
            {
                "state": "running",
                "active_policy": [policy.name],
                "requested_policy": [policy.name],
                "reason": None,
                "last_command_seq": 1,
            }
        )
        transport.publish_state('{"state":true}')
        transport.publish_reference_state('{"reference":true}')
        wait_until(
            lambda: (
                config.status_topic in received
                and config.state_topic in received
                and config.reference_state_topic in received
            )
        )
        assert json.loads(received[config.status_topic])["state"] == "running"
        assert json.loads(received[config.reference_state_topic]) == {"reference": True}
    finally:
        observer.disconnect()
        observer.loop_stop()
        transport.close()
        process.terminate()
        process.wait(timeout=3)
