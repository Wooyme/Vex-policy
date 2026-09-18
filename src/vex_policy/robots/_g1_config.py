"""Load named G1 parameters and assemble lists in the SDK hardware order."""

from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter

# This order is part of the hardware interface, independent of YAML entry order.
DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class _JointParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    default_dof_angle: float
    action_scale: float
    stiff_startup_kp: float
    stiff_startup_kd: float
    lower: float
    upper: float
    velocity: float


def load_joint_parameters(path: Path | None = None) -> dict[str, list[float]]:
    """Validate a complete joint-name mapping and return hardware-ordered lists."""
    resource = path if path is not None else files(__package__).joinpath("g1.yaml")
    parameters = TypeAdapter(dict[str, _JointParameters]).validate_python(
        yaml.safe_load(resource.read_text(encoding="utf-8"))
    )
    missing = set(DOF_NAMES) - parameters.keys()
    unexpected = parameters.keys() - set(DOF_NAMES)
    if missing or unexpected:
        raise ValueError(f"Invalid G1 joint names: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    return {field: [getattr(parameters[name], field) for name in DOF_NAMES] for field in _JointParameters.model_fields}


JOINT_PARAMETERS = load_joint_parameters()
