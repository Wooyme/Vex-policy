import numpy as np
from termcolor import colored

from vex_policy.config.config_types.control import input_parameters
from vex_policy.inputs.api.commands import VelCmd
from vex_policy.utils.latency import LatencyStage

from .base import BasePolicy
from .inference import OnnxActor, resolve_control_gains
from .joint_command import PositionAction, position_command
from .observations import ObservationHistory, robot_observation_terms


class LocomotionPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        self.observations = ObservationHistory(config.observation)
        self.actor = OnnxActor(config.task.model_path)
        self.robot_config = resolve_control_gains(
            config.robot, self.actor.metadata.get("kp"), self.actor.metadata.get("kd")
        )
        self.actions = PositionAction(
            self.num_dofs,
            self.action_mask,
            require_full_body=config.action_mask is not None,
            force_zero=config.task.debug.force_zero_action,
        )
        self.use_phase = config.task.use_phase
        self.gait_period = config.task.gait_period
        self._reset_commands()

    def _apply_velocity(self, vc: VelCmd) -> None:
        """Gate velocity by stand_command — zero when standing."""
        self._maybe_switch_to_walk_mode(vc)
        s = self.stand_command[0, 0]
        self.lin_vel_command[0] = (vc.lin_vel[0] * s, vc.lin_vel[1] * s)
        self.ang_vel_command[0, 0] = vc.ang_vel * s

    def _maybe_switch_to_walk_mode(self, vc: VelCmd) -> None:
        """Auto-enter walking mode when a non-zero velocity is received."""
        if not self.config.task.auto_walk_on_vel_cmd:
            return
        if self.stand_command[0, 0] == 1:
            return
        if abs(vc.lin_vel[0]) < 1e-3 and abs(vc.lin_vel[1]) < 1e-3 and abs(vc.ang_vel) < 1e-3:
            return
        self.stand_command[0, 0] = 1
        self.logger.info(colored("Auto-walk: non-zero velocity received", "blue"))

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = robot_observation_terms(
            robot_state_data, self.default_dof_angles, self.config.task.debug
        )
        current_obs_buffer_dict["actions"] = self.actions.last
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command

        # Add phase observations only if they are configured
        if "sin_phase" in self.observations.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["sin_phase"] = self._get_obs_sin_phase()
        if "cos_phase" in self.observations.obs_dict.get("actor_obs", []):
            current_obs_buffer_dict["cos_phase"] = self._get_obs_cos_phase()

        return current_obs_buffer_dict

    def _get_obs_sin_phase(self):
        """Calculate sin phase for gait."""
        return np.array([np.sin(self.phase[0, :])])

    def _get_obs_cos_phase(self):
        """Calculate cos phase for gait."""
        return np.array([np.cos(self.phase[0, :])])

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        if np.linalg.norm(self.lin_vel_command[0]) < 0.01 and np.linalg.norm(self.ang_vel_command[0]) < 0.01:
            # Robot should stand still - set both feet to same phase
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            # When the robot starts to move, reset the phase to initial state
            self.phase = np.array([[0.0, np.pi]])
            self.is_standing = False

    def _reset_commands(self):
        self.is_standing = False
        self.lin_vel_command = np.zeros((1, 2))
        self.ang_vel_command = np.zeros((1, 1))
        self.stand_command = np.zeros((1, 1), dtype=int)
        self.phase = np.array([[0.0, np.pi]])
        if self.use_phase:
            self.phase_dt = 2 * np.pi / (self.rl_rate * self.gait_period)

    def _on_deactivate(self):
        self.observations.reset()
        self.actions.reset()
        self._reset_commands()

    def _on_activate(self, robot_state_data):
        self._on_deactivate()
        self._apply_control({param.name: param.default for param in input_parameters(self.config.inputs)})

    def _apply_control(self, control):
        if control:
            self._apply_velocity(VelCmd((control["vy"], -control["vx"]), -control["yaw"]))

    def _compute_command(self, robot_state_data):
        with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
            if self.use_phase:
                self.update_phase_time()
            obs = self.observations.prepare(self.get_current_obs_buffer_dict(robot_state_data))
            if self.config.task.print_observations:
                self.observations.print_observations(obs, self.dof_names, self.actions.scaled)
        with self.latency_tracker.measure(LatencyStage.INFERENCE):
            action = self.actor({"actor_obs": obs["actor_obs"]})
        with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
            self.actions.process(action, self.config.task.policy_action_scale)
            return position_command(
                self.actions.target(self.default_dof_angles),
                self.robot_config.motor_kp,
                self.robot_config.motor_kd,
                self.controlled_joint_mask,
            )
