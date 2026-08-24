import numpy as np
from numpy import bool

from vex_policy.policies.guard.base import BaseGuard
from vex_policy.utils.math import quat_rotate_inverse, xyzw_to_wxyz


class WbtGuard(BaseGuard):
    def start_check(self) -> tuple[bool, str | None]:
        robot_state_data = self.policy.interface.get_low_state()
        motion_command_t = self.policy.motion_command_0
        ref_quat_xyzw_t = self.policy.ref_quat_xyzw_0
        if self.bad_ref_ori(xyzw_to_wxyz(ref_quat_xyzw_t), robot_state_data):
            self.policy.logger.warning("start check failed, bad_ref_ori")
            return False, "start check failed, bad_ref_ori"
        if self.bad_lower_joint_pos(motion_command_t, robot_state_data):
            self.policy.logger.warning("start check failed, bad_joint_pos")
            return False, "start check failed, bad_joint_pos"
        return True, None

    def bad_ref_ori(self, ref_quat_wxyz, robot_state_data) -> bool:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            ref_quat_wxyz, np.array([0.0, 0.0, -1.0])
        )[0].astype(np.float32)
        robot_projected_gravity_b = quat_rotate_inverse(
            robot_state_data[:, 3:7], np.array([0.0, 0.0, -1.0])
        )[0].astype(np.float32)
        return abs(motion_projected_gravity_b[2] - robot_projected_gravity_b[2]) > self.config.bad_ref_ori_threshold

    def bad_lower_joint_pos(self, motion_command, robot_state_data):
        for i in self.policy.lower_dof_indices:
            command_pos = motion_command[0][i]
            joint_pos = robot_state_data[0][i + 7]
            if abs(command_pos - joint_pos) > self.config.bad_lower_joint_pos_threshold:
                return True
        return False
