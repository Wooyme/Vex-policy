"""Shared runtime for Holosoma BFM latent controllers driving the frozen actor."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np

from vex_policy.config.config_types import InferenceConfig, input_parameters
from vex_policy.policies.base import BasePolicy, PolicyRuntimeFault
from vex_policy.policies.bfm.contract import (
    ACTOR_CONTRACT_VERSION,
    ACTOR_HISTORY_LENGTH,
    ACTOR_OBS_DIM,
    DEFAULT_DOF_ANGLES,
    FRAME_DIM,
    JOINT_NAMES,
    LATENT_DIM,
    pack_actor_observation,
    project_latent,
)
from vex_policy.policies.utils.inference import load_metadata, resolve_control_gains, shared_session
from vex_policy.policies.utils.initial_pose import normalize_quaternion_wxyz
from vex_policy.policies.utils.joint_command import position_command
from vex_policy.policies.utils.observations import robot_observation_terms
from vex_policy.policies.utils.sonic_planner import ort_providers
from vex_policy.robots import G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyStage


def _validate_interface(session, inputs: dict[str, int], output_dim: int, label: str) -> None:
    """Require the supported batch-one float interfaces, independent of ONNX input enumeration order."""
    actual_inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(actual_inputs) != len(inputs)
        or {item.name for item in actual_inputs} != inputs.keys()
        or any(item.type != "tensor(float)" or list(item.shape) != [1, inputs[item.name]] for item in actual_inputs)
    ):
        raise ValueError(f"BFM {label} ONNX inputs must be {inputs} with batch size 1 and float32")
    if (
        len(outputs) != 1
        or outputs[0].name != "action"
        or outputs[0].type != "tensor(float)"
        or list(outputs[0].shape) != [1, output_dim]
    ):
        raise ValueError(f"BFM {label} ONNX must expose action[1, {output_dim}] float32")


def _joint_vector(value, label: str, *, nonnegative: bool = False) -> np.ndarray:
    """Validate a hardware-order calibration vector before it can reach the runtime."""
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (29,) or not np.isfinite(values).all() or (nonnegative and np.any(values < 0)):
        raise ValueError(f"BFM {label} must contain 29 finite {'nonnegative ' if nonnegative else ''}values")
    return values


class BfmPolicy(BasePolicy):
    """Run a latent controller every four cycles and the frozen 465+256→29 actor every cycle.

    Subclasses declare their ``command_obs`` terms and inputs, and build those command terms;
    the frame/latent groups, model contract checks and actor runtime are shared.
    """

    task_type: ClassVar[type]
    # Controller group names in training order: frames, commands, then the executed latent.
    frame_group: ClassVar[str] = "actor_obs"
    command_group: ClassVar[str] = "command_obs"
    # Oldest-first controller-cadence frames; Holosoma zero-pads history after a reset.
    frame_history: ClassVar[int] = 1
    # command_obs term -> (dimension, Holosoma observation function relative to managers.observation.terms).
    command_obs: ClassVar[dict[str, tuple[int, str]]]
    input_names: ClassVar[frozenset[str]] = frozenset()
    # Holosoma repeats the first frame into the frozen actor history since its token training.
    repeat_first_actor_frame: ClassVar[bool] = False
    controller_output_dim: ClassVar[int] = LATENT_DIM

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, self.task_type):
            raise TypeError(f"{type(self).__name__} requires {self.task_type.__name__}")
        super().__init__(config)
        if tuple(self.dof_names) != JOINT_NAMES or self.num_dofs != 29:
            raise ValueError("BFM requires the supported G1 29-DOF hardware joint order")
        if config.action_mask is not None or not self.controlled_joint_mask.all():
            raise ValueError("BFM requires full-body control without action masks")
        self.task = config.task
        self._input_parameters = {p.name: p for p in input_parameters(config.inputs)}
        if self._input_parameters.keys() != self.input_names:
            raise ValueError(f"{type(self).__name__} inputs must be exactly {sorted(self.input_names)}")
        self._groups = {
            self.frame_group: {"frame": FRAME_DIM},
            self.command_group: {term: dimension for term, (dimension, _) in self.command_obs.items()},
            "latent_history_obs": {"latent": LATENT_DIM},
        }
        self._history = {group: self.frame_history if group == self.frame_group else 1 for group in self._groups}
        self._setup_task()
        self._validate_observations()
        providers = ort_providers(self.task.inference_provider)
        self.controller = shared_session(
            self.task.model_path, providers, intra_op_num_threads=self.task.inference_threads
        )
        self.actor = shared_session(
            self.task.actor_model_path, providers, intra_op_num_threads=self.task.inference_threads
        )
        metadata = load_metadata(self.task.model_path)
        actor_metadata = load_metadata(self.task.actor_model_path)
        self._validate_models(metadata, actor_metadata)
        self.default_dof_angles = _joint_vector(actor_metadata.get("default_dof_angles"), "motor zeros")
        self.action_scales = _joint_vector(metadata.get("action_scale"), "action_scale", nonnegative=True)
        self.robot_config = resolve_control_gains(config.robot, metadata.get("kp"), metadata.get("kd"))
        for name in ("motor_kp", "motor_kd"):
            _joint_vector(getattr(self.robot_config, name), name, nonnegative=True)
        self._reset_episode()

    def _setup_task(self) -> None:
        """Validate task-specific configuration before the models are loaded."""

    def _command_observation(self) -> dict[str, np.ndarray]:
        """Return the current command_obs terms."""
        raise NotImplementedError

    def _validate_observations(self) -> None:
        """Keep controller groups separate; alphabetical term sorting is internal to each group."""
        observation = self.config.observation
        if {name: sorted(terms) for name, terms in observation.obs_dict.items()} != {
            name: sorted(terms) for name, terms in self._groups.items()
        }:
            raise ValueError("BFM observation groups do not match the controller contract")
        for group, terms in self._groups.items():
            if observation.history_length_dict.get(group, 1) != self._history[group]:
                raise ValueError(f"BFM controller group {group} must use history length {self._history[group]}")
            for term, dimension in terms.items():
                if observation.obs_dims.get(term) != dimension or observation.obs_scales.get(term) != 1.0:
                    raise ValueError(f"BFM observation {term} must have dimension {dimension} and scale 1")

    def _validate_models(self, metadata: dict, actor_metadata: dict) -> None:
        """Reject legacy latent controllers and actor exports with a different normalization/action convention."""
        controller_dim = sum(sum(terms.values()) * self._history[group] for group, terms in self._groups.items())
        _validate_interface(self.controller, {"actor_obs": controller_dim}, self.controller_output_dim, "controller")
        _validate_interface(self.actor, {"actor_obs": ACTOR_OBS_DIM, "latent": LATENT_DIM}, 29, "actor")
        for label, model_metadata in (("controller", metadata), ("actor", actor_metadata)):
            if tuple(model_metadata.get("dof_names", ())) != JOINT_NAMES:
                raise ValueError(f"BFM {label} dof_names do not match G1")
        if actor_metadata.get("bfm_actor_contract") != ACTOR_CONTRACT_VERSION:
            raise ValueError("BFM actor must be converted with vex-bfm-export-actor")
        zeros = _joint_vector(actor_metadata.get("default_dof_angles"), "actor motor zeros")
        if not np.allclose(zeros, DEFAULT_DOF_ANGLES, rtol=0, atol=1e-7):
            raise ValueError("BFM actor motor zeros do not match the supported checkpoint")
        try:
            experiment = metadata["experiment_config"]
            if (
                experiment["env_class"] != "holosoma.envs.bfm_controller.environment.BFMControllerEnvironment"
                or experiment["control_decimation"] != self.task.controller_decimation
                or experiment["algo"]["config"]["module_dict"]["actor"]["input_dim"] != list(self._groups)
            ):
                raise ValueError("BFM controller experiment/cadence does not match the supported contract")
            simulator = experiment["simulator"]["config"]["sim"]
            if (
                simulator["control_decimation"] <= 0
                or simulator["fps"] != self.rl_rate * simulator["control_decimation"]
            ):
                raise ValueError("BFM controller was not trained with a 50 Hz frozen actor")
            # The training reset pose must not redefine the frozen actor's residual zeros.
            training_zeros = experiment["robot"]["init_state"]["default_joint_angles"]
            if not np.allclose([training_zeros[name] for name in JOINT_NAMES], zeros, rtol=0, atol=1e-7):
                raise ValueError("BFM controller motor zeros differ from the frozen actor")
            training_groups = experiment["observation"]["groups"]
            functions = {
                "frame": "bfm:proprioceptive_frame",
                "latent": "bfm:executed_latent",
                **{term: function for term, (_, function) in self.command_obs.items()},
            }
            for name, terms in self._groups.items():
                group = training_groups[name]
                if (
                    group.get("history_length", 1) != self._history[name]
                    or not group.get("concatenate", True)
                    or group["terms"].keys() != terms.keys()
                ):
                    raise ValueError(f"BFM controller training observation group {name} is incompatible")
                for term in terms:
                    cfg = group["terms"][term]
                    if (
                        cfg["func"] != "holosoma.managers.observation.terms." + functions[term]
                        or cfg.get("scale", 1.0) != 1.0
                        or cfg.get("clip") is not None
                        or cfg.get("params", {})
                    ):
                        raise ValueError(f"BFM controller training observation {term} is incompatible")
            self._validate_experiment(experiment)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("BFM controller requires complete Holosoma experiment_config metadata") from error
        for name in ("kp", "kd"):
            _joint_vector(metadata.get(name), name, nonnegative=True)

    def _validate_experiment(self, experiment: dict) -> None:
        """Task-specific training contract checks; missing keys are reported as incomplete metadata."""

    def _reset_episode(self) -> None:
        """An episode owns its history and executed latent, even when sessions are shared."""
        self.actor_history = np.zeros((1, ACTOR_HISTORY_LENGTH, FRAME_DIM), dtype=np.float32)
        self._actor_history_stale = self.repeat_first_actor_frame
        self.controller_frames = np.zeros((1, self.frame_history, FRAME_DIM), dtype=np.float32)
        self.latent = np.zeros((1, LATENT_DIM), dtype=np.float32)
        self.motor_actions = np.zeros((1, 29), dtype=np.float32)
        self._episode_step = 0
        self._control = {name: p.default for name, p in self._input_parameters.items()}

    def _validated_state(self, state: LowState) -> LowState:
        for name, shape in (
            ("joint_pos", (1, 29)),
            ("joint_vel", (1, 29)),
            ("base_ang_vel", (1, 3)),
            ("base_quat", (1, 4)),
        ):
            value = np.asarray(getattr(state, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise PolicyRuntimeFault(f"Invalid BFM state: {name}")
        try:
            quaternion = normalize_quaternion_wxyz(state.base_quat[0])
        except ValueError as error:
            raise PolicyRuntimeFault(f"Invalid BFM quaternion: {error}") from error
        return replace(state, base_quat=quaternion[None])

    def _on_activate(self, robot_state_data: LowState) -> None:
        self._validated_state(robot_state_data)
        self._reset_episode()

    def _on_deactivate(self) -> None:
        self._reset_episode()

    def _apply_control(self, control) -> None:
        if control.keys() != self._input_parameters.keys():
            raise PolicyRuntimeFault("BFM control must contain all configured inputs")
        for name, parameter in self._input_parameters.items():
            value = control[name]
            if isinstance(value, bool) or not np.isfinite(value) or not parameter.min <= value <= parameter.max:
                raise PolicyRuntimeFault(f"Invalid BFM input: {name}")
        self._control = dict(control)

    def _controller_observation(self, frame: np.ndarray) -> np.ndarray:
        self.controller_frames[:, :-1] = self.controller_frames[:, 1:].copy()
        self.controller_frames[:, -1] = frame
        frames = self.controller_frames.reshape(1, -1)
        terms = {"frame": frames, "latent": self.latent, **self._command_observation()}
        return np.concatenate(
            [terms[term] for group in self._groups.values() for term in sorted(group)], axis=1
        ).astype(np.float32)

    def _run_controller(self, frame: np.ndarray) -> np.ndarray:
        controller_obs = self._controller_observation(frame)
        if self.task.print_observations:
            print("BFM controller actor_obs:", controller_obs)
        return self._infer(self.controller, {"actor_obs": controller_obs}, self.controller_output_dim, "controller")

    def _decode_latent(self, action: np.ndarray) -> np.ndarray:
        """Map a controller action to the executed radius-16 latent."""
        return project_latent(action)

    def _clock_phase(self) -> np.ndarray:
        """Holosoma's evaluation gait clock: mean frequency, offsets [0, -pi], at controller cadence."""
        controller_step = self._episode_step // self.task.controller_decimation
        phase = controller_step * self.task.controller_decimation / self.rl_rate * (
            2 * np.pi / self.task.gait_period
        ) + np.array([[0.0, -np.pi]])
        return np.fmod(phase + np.pi, 2 * np.pi) - np.pi

    def _update_latent(self, frame: np.ndarray) -> None:
        """Refresh the executed latent at controller cadence."""
        if self._episode_step % self.task.controller_decimation == 0:
            self.latent = self._decode_latent(self._run_controller(frame))

    @staticmethod
    def _infer(session, feed: dict[str, np.ndarray], dimension: int, label: str) -> np.ndarray:
        if any(not np.isfinite(value).all() for value in feed.values()):
            raise PolicyRuntimeFault(f"Nonfinite BFM {label} observations")
        try:
            action = np.asarray(session.run(["action"], feed)[0], dtype=np.float32)
        except Exception as error:
            raise PolicyRuntimeFault(f"BFM {label} inference failed: {error}") from error
        if action.shape != (1, dimension) or not np.isfinite(action).all():
            raise PolicyRuntimeFault(f"BFM {label} must return {dimension} finite actions")
        return action

    def _compute_command(self, robot_state_data: LowState):
        with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
            state = self._validated_state(robot_state_data)
            terms = robot_observation_terms(state, self.default_dof_angles, self.task.debug)
            # BFM dq is unscaled; only body-frame omega uses the checkpoint's 0.25 factor.
            frame = np.concatenate(
                (
                    terms["dof_pos"],
                    terms["dof_vel"],
                    terms["projected_gravity"],
                    0.25 * terms["base_ang_vel"],
                    self.motor_actions,
                ),
                axis=1,
            ).astype(np.float32)
            self.actor_history[:, :-1] = self.actor_history[:, 1:].copy()
            self.actor_history[:, -1] = frame
            if self._actor_history_stale:
                # All-zero history is no reachable robot state; the first frame fills it instead.
                self.actor_history[:] = frame
                self._actor_history_stale = False
            actor_obs = pack_actor_observation(self.actor_history)
        with self.latency_tracker.measure(LatencyStage.INFERENCE):
            self._update_latent(frame)
            motor = self._infer(self.actor, {"actor_obs": actor_obs, "latent": self.latent}, 29, "actor")
        with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
            # The exported actor already includes the source tanh * 5 convention.
            self.motor_actions = np.clip(motor, -5.0, 5.0)
            if self.task.debug.force_zero_action:
                self.motor_actions.fill(0.0)
            target = np.clip(
                self.default_dof_angles + self.motor_actions[0] * self.action_scales, G1_JOINT_LOWER, G1_JOINT_UPPER
            )
            self._episode_step += 1
            return position_command(
                target, self.robot_config.motor_kp, self.robot_config.motor_kd, self.controlled_joint_mask
            )
