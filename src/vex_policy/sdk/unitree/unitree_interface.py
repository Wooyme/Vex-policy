"""Unitree robot interface using C++/pybind11 binding."""

import numpy as np

from vex_policy.config.config_types import RobotConfig
from vex_policy.sdk.base.base_interface import BaseInterface, LowCommand, LowState


class UnitreeInterface(BaseInterface):
    """Interface for Unitree robots using C++/pybind11 binding."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        self._unitree_motor_order = None
        self._kp_level = 1.0
        self._kd_level = 1.0
        self._init_binding()

    def _init_binding(self):
        """Initialize C++/pybind11 binding."""
        try:
            import unitree_interface
        except ImportError as e:
            raise ImportError("unitree_interface python binding not found.") from e

        robot_type_map = {
            "G1": unitree_interface.RobotType.G1,
            "H1": unitree_interface.RobotType.H1,
            "H1_2": unitree_interface.RobotType.H1_2,
            "GO2": unitree_interface.RobotType.GO2,
        }
        message_type_map = {"HG": unitree_interface.MessageType.HG, "GO2": unitree_interface.MessageType.GO2}

        self.unitree_interface = unitree_interface.create_robot(
            self.interface_str,
            robot_type_map[self.robot_config.robot.upper()],
            message_type_map[self.robot_config.message_type.upper()],
        )
        self.unitree_interface.set_control_mode(unitree_interface.ControlMode.PR)

        # GO2 SDK motor order differs from joint order
        if self.robot_config.robot.lower() == "go2":
            self._unitree_motor_order = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

    def _get_low_state(self) -> LowState | None:
        """Get the latest SDK-cached robot state in configured joint order."""
        state = self.unitree_interface.read_low_state()
        if state is None:
            return None

        base_pos = np.zeros((1, 3))
        quat = np.asarray(state.imu.quat).reshape(1, 4)
        base_lin_vel = np.zeros((1, 3))
        base_ang_vel = np.asarray(state.imu.omega).reshape(1, 3)

        empty_joint_values = np.zeros((1, self.robot_config.num_joints))
        motor_order = self._unitree_motor_order or self.robot_config.joint2motor

        def read_motor_values(name: str) -> np.ndarray:
            motor_values = getattr(state.motor, name, None)
            if motor_values is None:
                return np.zeros_like(empty_joint_values)

            motor_values = np.asarray(motor_values).reshape(-1)
            if len(motor_values) <= max(motor_order):
                return np.zeros_like(empty_joint_values)

            joint_values = np.zeros_like(empty_joint_values)
            for j_id in range(self.robot_config.num_joints):
                joint_values[0, j_id] = float(motor_values[motor_order[j_id]])
            return joint_values

        q = read_motor_values("q")
        dq = read_motor_values("dq")
        ddq = read_motor_values("ddq")
        tau_est = read_motor_values("tau_est")

        return LowState(
            base_pos=base_pos,
            base_quat=quat,
            joint_pos=q,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            joint_vel=dq,
            q=q,
            dq=dq,
            ddq=ddq,
            tau_est=tau_est,
        )

    def _prepare_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray | None,
        kp_override: np.ndarray | None,
        kd_override: np.ndarray | None,
    ) -> LowCommand:
        """Build the final Unitree SDK command in motor order."""
        cmd_q_target = np.zeros(self.robot_config.num_motors)
        cmd_dq_target = np.zeros(self.robot_config.num_motors)
        cmd_tau_target = np.zeros(self.robot_config.num_motors)
        cmd_kp = np.zeros(self.robot_config.num_motors) if kp_override is not None else None
        cmd_kd = np.zeros(self.robot_config.num_motors) if kd_override is not None else None

        motor_order = self._unitree_motor_order or self.robot_config.joint2motor
        for j_id in range(self.robot_config.num_joints):
            m_id = motor_order[j_id]
            cmd_q_target[m_id] = float(cmd_q[j_id])
            cmd_dq_target[m_id] = float(cmd_dq[j_id])
            cmd_tau_target[m_id] = float(cmd_tau[j_id])
            if cmd_kp is not None:
                cmd_kp[m_id] = float(kp_override[j_id])
            if cmd_kd is not None:
                cmd_kd[m_id] = float(kd_override[j_id])

        cmd = self.unitree_interface.create_zero_command()
        cmd.q_target = list(cmd_q_target)
        cmd.dq_target = list(cmd_dq_target)
        cmd.tau_ff = list(cmd_tau_target)

        motor_kp = np.array(cmd_kp if cmd_kp is not None else self.robot_config.motor_kp)
        motor_kd = np.array(cmd_kd if cmd_kd is not None else self.robot_config.motor_kd)
        final_kp = motor_kp * self._kp_level
        final_kd = motor_kd * self._kd_level
        cmd.kp = list(final_kp)
        cmd.kd = list(final_kd)

        return LowCommand(
            payload=cmd,
            q_target=cmd_q_target,
            dq_target=cmd_dq_target,
            tau_ff=cmd_tau_target,
            kp=final_kp,
            kd=final_kd,
        )

    def _write_low_command(self, command: LowCommand) -> None:
        """Write one prepared Unitree command."""
        self.unitree_interface.write_low_command(command.payload)

    @property
    def kp_level(self):
        """Get proportional gain level."""
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        """Set proportional gain level."""
        self._kp_level = value

    @property
    def kd_level(self):
        """Get derivative gain level."""
        return self._kd_level

    @kd_level.setter
    def kd_level(self, value):
        """Set derivative gain level."""
        self._kd_level = value
