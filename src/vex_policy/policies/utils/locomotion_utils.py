"""Shared motion reference and right-ankle kinematics for locomotion adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from vex_policy.sdk.base.base_interface import LowState

from .initial_pose import InitialPose


def load_motion_last_pose(path: str | Path) -> InitialPose:
    """Load and validate the final root/joint pose from a Holosoma motion NPZ."""
    motion_path = Path(path)
    if not motion_path.is_file():
        raise ValueError(f"Locomotion motion file does not exist: {motion_path}")
    try:
        with np.load(motion_path, allow_pickle=False) as motion:
            missing = {"joint_names", "joint_pos"} - set(motion.files)
            if missing:
                raise ValueError(f"Locomotion motion is missing arrays: {sorted(missing)}")
            joint_names = tuple(str(name) for name in motion["joint_names"].tolist())
            joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Locomotion motion"):
            raise
        raise ValueError(f"Failed to load locomotion motion {motion_path}: {error}") from error

    expected_width = len(joint_names) + 7
    if joint_pos.ndim != 2 or joint_pos.shape[0] == 0 or joint_pos.shape[1] != expected_width:
        raise ValueError(
            f"Locomotion joint_pos must have shape (frames, 7 + {len(joint_names)}), got {joint_pos.shape}"
        )
    final_pose = joint_pos[-1]
    root_quat_wxyz = final_pose[3:7]
    return InitialPose(
        dof_names=joint_names,
        dof_pos=tuple(final_pose[7:]),
        root_quat_wxyz=tuple(root_quat_wxyz),
    )


class RightAnkleKinematics:
    """Fixed-base FK; unobserved auxiliary joints retain their neutral positions."""

    def __init__(self, robot_urdf: str, dof_names: tuple[str, ...]):
        if not isinstance(robot_urdf, str) or not robot_urdf.strip():
            raise ValueError("Locomotion ONNX must contain non-empty robot_urdf metadata")
        try:
            kinematics_model = pin.buildModelFromXML(robot_urdf)
        except Exception as error:
            raise ValueError(f"Failed to build locomotion kinematics from ONNX robot_urdf: {error}") from error

        q_indices: list[int] = []
        for name in dof_names:
            joint_id = kinematics_model.getJointId(name)
            if joint_id == 0 or joint_id >= kinematics_model.njoints or kinematics_model.names[joint_id] != name:
                raise ValueError(f"Locomotion robot_urdf is missing joint {name!r}")
            joint = kinematics_model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"Locomotion robot_urdf joint {name!r} must have one configuration value")
            q_indices.append(joint.idx_q)

        ankle_frame_id = kinematics_model.getFrameId("right_ankle_roll_link", pin.FrameType.BODY)
        if ankle_frame_id >= kinematics_model.nframes:
            raise ValueError("Locomotion robot_urdf is missing body frame 'right_ankle_roll_link'")
        self._kinematics_model = kinematics_model
        self._kinematics_data = kinematics_model.createData()
        self._kinematics_q_indices = np.asarray(q_indices, dtype=np.int64)
        self._right_ankle_frame_id = ankle_frame_id

    def height_difference(self, robot_state_data: LowState, projected_gravity: np.ndarray) -> np.ndarray:
        """Solve base-minus-right-ankle world height from joints and IMU gravity."""

        gravity_b = np.asarray(projected_gravity, dtype=np.float64)
        if gravity_b.shape != (1, 3) or not np.isfinite(gravity_b).all():
            raise ValueError("Cannot solve right-ankle height with invalid projected gravity")

        joint_pos = np.asarray(robot_state_data.joint_pos[0], dtype=np.float64)
        if not np.isfinite(joint_pos).all():
            raise ValueError("Cannot solve right-ankle height with invalid joint positions")
        q = pin.neutral(self._kinematics_model)
        q[self._kinematics_q_indices] = joint_pos
        pin.forwardKinematics(self._kinematics_model, self._kinematics_data, q)
        ankle_placement = pin.updateFramePlacement(
            self._kinematics_model, self._kinematics_data, self._right_ankle_frame_id
        )
        ankle_position_b = np.asarray(ankle_placement.translation, dtype=np.float64)

        # projected_gravity is world-down expressed in the base frame. Its dot
        # product with (ankle - base) is therefore base_z - ankle_z in world.
        return np.asarray([[np.dot(gravity_b[0], ankle_position_b)]], dtype=np.float64)
