"""Stable Unitree G1 29-DOF hardware configuration."""

from vex_policy.config.config_types.robot import RobotConfig

from ._g1_config import DOF_NAMES, JOINT_PARAMETERS

DEFAULT_DOF_ANGLES = JOINT_PARAMETERS["default_dof_angle"]

UPPER_BODY_DOF_NAMES = DOF_NAMES[15:]
LOWER_BODY_DOF_NAMES = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

DEFAULT_PER_JOINT_ACTION_SCALE = JOINT_PARAMETERS["action_scale"]

STIFF_STARTUP_KP = JOINT_PARAMETERS["stiff_startup_kp"]

STIFF_STARTUP_KD = JOINT_PARAMETERS["stiff_startup_kd"]

_WEAK_MOTOR_ORDER = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *UPPER_BODY_DOF_NAMES,
)

G1_29DOF = RobotConfig(
    robot_type="g1_29dof",
    robot="g1",
    default_dof_angles=DEFAULT_DOF_ANGLES,
    default_motor_angles=DEFAULT_DOF_ANGLES,
    motor2joint=list(range(len(DOF_NAMES))),
    joint2motor=list(range(len(DOF_NAMES))),
    dof_names=DOF_NAMES,
    dof_names_upper_body=UPPER_BODY_DOF_NAMES,
    dof_names_lower_body=LOWER_BODY_DOF_NAMES,
    motor_kp=None,
    motor_kd=None,
    default_per_joint_action_scale=DEFAULT_PER_JOINT_ACTION_SCALE,
    joint_interpolation_slew_safety_factor=0.5,
    stiff_startup_pos=DEFAULT_DOF_ANGLES,
    stiff_startup_kp=STIFF_STARTUP_KP,
    stiff_startup_kd=STIFF_STARTUP_KD,
    sdk_type="unitree",
    motor_type="serial",
    message_type="HG",
    num_motors=29,
    num_joints=29,
    torso_link_name="torso_link",
    left_hand_link_name="left_rubber_hand",
    right_hand_link_name="right_rubber_hand",
    unitree_legged_const={
        "HIGHLEVEL": 238,
        "LOWLEVEL": 255,
        "TRIGERLEVEL": 240,
        "PosStopF": 2146000000.0,
        "VelStopF": 16000.0,
        "MODE_MACHINE": 5,
        "MODE_PR": 0,
    },
    weak_motor_joint_index=dict(zip(_WEAK_MOTOR_ORDER, range(29), strict=True)),
    motion={"body_name_ref": ["torso_link"]},
    dof_names_parallel_mech=(),
    use_sensor=False,
    num_upper_body_joints=14,
    joint_offsets_deg=None,
)

__all__ = ["G1_29DOF"]
