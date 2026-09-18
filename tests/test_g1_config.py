from importlib.resources import files

import pytest
import yaml

from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER, G1_JOINT_VELOCITY
from vex_policy.robots._g1_config import DOF_NAMES, load_joint_parameters
from vex_policy.robots.g1 import (
    DEFAULT_DOF_ANGLES,
    DEFAULT_PER_JOINT_ACTION_SCALE,
    STIFF_STARTUP_KD,
    STIFF_STARTUP_KP,
)


@pytest.fixture
def joint_parameters():
    return yaml.safe_load(files("vex_policy.robots").joinpath("g1.yaml").read_text(encoding="utf-8"))


def test_yaml_order_does_not_change_hardware_order(tmp_path, joint_parameters):
    # Give each joint/field a distinct value to expose any positional mapping.
    for index, name in enumerate(DOF_NAMES):
        for field_index, field in enumerate(joint_parameters[name]):
            joint_parameters[name][field] = index + field_index / 10
    path = tmp_path / "g1.yaml"
    path.write_text(yaml.safe_dump(dict(reversed(joint_parameters.items())), sort_keys=False))

    loaded = load_joint_parameters(path)

    for field, values in loaded.items():
        assert isinstance(values, list)
        assert values == [joint_parameters[name][field] for name in DOF_NAMES]


@pytest.mark.parametrize("change", ["missing_joint", "unknown_joint", "missing_field", "unknown_field", "nan", "text"])
def test_invalid_joint_parameters_are_rejected(tmp_path, joint_parameters, change):
    name = DOF_NAMES[0]
    if change == "missing_joint":
        del joint_parameters[name]
    elif change == "unknown_joint":
        joint_parameters["left_hip_pitch_typo"] = joint_parameters.pop(name)
    elif change == "missing_field":
        del joint_parameters[name]["lower"]
    elif change == "unknown_field":
        joint_parameters[name]["lowre"] = joint_parameters[name].pop("lower")
    else:
        joint_parameters[name]["lower"] = float("nan") if change == "nan" else "invalid"
    path = tmp_path / "g1.yaml"
    path.write_text(yaml.safe_dump(joint_parameters))

    with pytest.raises(ValueError, match=name):
        load_joint_parameters(path)


def test_exported_constants_and_robot_config_are_lists(joint_parameters):
    constants = {
        "default_dof_angle": DEFAULT_DOF_ANGLES,
        "action_scale": DEFAULT_PER_JOINT_ACTION_SCALE,
        "stiff_startup_kp": STIFF_STARTUP_KP,
        "stiff_startup_kd": STIFF_STARTUP_KD,
        "lower": G1_JOINT_LOWER,
        "upper": G1_JOINT_UPPER,
        "velocity": G1_JOINT_VELOCITY,
    }
    for field, values in constants.items():
        assert isinstance(values, list)
        assert values == [joint_parameters[name][field] for name in DOF_NAMES]

    robot_fields = {
        "default_dof_angles": DEFAULT_DOF_ANGLES,
        "default_motor_angles": DEFAULT_DOF_ANGLES,
        "stiff_startup_pos": DEFAULT_DOF_ANGLES,
        "default_per_joint_action_scale": DEFAULT_PER_JOINT_ACTION_SCALE,
        "stiff_startup_kp": STIFF_STARTUP_KP,
        "stiff_startup_kd": STIFF_STARTUP_KD,
        "motor2joint": list(range(29)),
        "joint2motor": list(range(29)),
    }
    for field, expected in robot_fields.items():
        actual = getattr(G1_29DOF, field)
        assert isinstance(actual, list)
        assert actual == expected
