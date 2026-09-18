"""Model loading utilities, independent of the policy lifecycle."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import numpy as np
import onnx
import onnxruntime

_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], onnxruntime.InferenceSession] = {}
_SESSION_CACHE_LOCK = threading.Lock()


def shared_session(path: str, providers: list[str]) -> onnxruntime.InferenceSession:
    """Share read-only model sessions, never mutable episode data."""
    key = (path, tuple(providers))
    with _SESSION_CACHE_LOCK:
        session = _SESSION_CACHE.get(key)
        if session is None:
            session = onnxruntime.InferenceSession(path, providers=providers)
            _SESSION_CACHE[key] = session
        return session


def load_metadata(model_path: str) -> dict:
    model = onnx.load(model_path, load_external_data=False)
    return {prop.key: json.loads(prop.value) for prop in model.metadata_props}


def resolve_control_gains(robot_config, kp=None, kd=None):
    """Resolve the existing paired config override > model metadata precedence."""
    if robot_config.motor_kp is not None and robot_config.motor_kd is not None:
        kp, kd = robot_config.motor_kp, robot_config.motor_kd
    elif kp is None or kd is None:
        raise ValueError(
            "No KP/KD values found. Either provide them in robot config "
            "or ensure ONNX model has metadata attached during training."
        )
    kp, kd = np.asarray(kp), np.asarray(kd)
    for name, values in (("KP", kp), ("KD", kd)):
        if len(values) != robot_config.num_motors:
            raise ValueError(
                f"{name} array length ({len(values)}) does not match num_motors ({robot_config.num_motors})"
            )
    return replace(robot_config, motor_kp=tuple(kp.tolist()), motor_kd=tuple(kd.tolist()))


class OnnxActor:
    """Single-action ONNX adapter used by locomotion and waist locomotion."""

    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.metadata = load_metadata(model_path)

    def __call__(self, observations):
        return self.session.run(self.output_names, {name: observations[name] for name in self.input_names})[0]
