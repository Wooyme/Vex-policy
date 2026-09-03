from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from vex_policy.config.config_types import PolicySpec, WbtTaskConfig
from vex_policy.policies.wbt import WholeBodyTrackingPolicy
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.joint_interpolation import JointPositionInterpolator


def _state(joint_pos=(0.0, 0.0)) -> LowState:
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=np.asarray([joint_pos], dtype=np.float64),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
        joint_vel=np.zeros((1, len(joint_pos))),
    )


def _policy(startup_mode: str = "interpolate") -> WholeBodyTrackingPolicy:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(task=SimpleNamespace(startup_mode=startup_mode, init_duration_s=1.0, rl_rate=2.0))
    policy.guard = None
    policy.num_dofs = 2
    policy.motion_command_0 = np.asarray([[1.0, 2.0, 0.0, 0.0]])
    policy.ref_quat_xyzw_0 = np.asarray([[0.0, 0.0, 0.0, 1.0]])
    policy.motion_command_t = np.zeros_like(policy.motion_command_0)
    policy.ref_quat_xyzw_t = np.zeros_like(policy.ref_quat_xyzw_0)
    policy.last_policy_action = np.ones((1, 2))
    policy.scaled_policy_action = np.ones((1, 2))
    policy.obs_history_buffers = {"actor_obs": {"dof_pos": deque([np.ones((1, 2))], maxlen=2)}}
    policy.obs_buf_dict = {"actor_obs": np.ones((1, 4))}
    policy._activation_q = None
    policy._startup_interpolator = JointPositionInterpolator(
        joint_lower=(-10.0, -10.0),
        joint_upper=(10.0, 10.0),
        joint_velocity=(100.0, 100.0),
        rate_hz=2.0,
        duration_s=1.0,
        slew_safety_factor=1.0,
    )
    policy._stiff_hold_active = True
    policy.motion_clip_progressing = False
    policy.use_policy_action = False
    policy.get_ready_state = False
    policy.init_count = 0
    policy._init_phase_components = lambda: None
    policy.logger = SimpleNamespace(info=lambda message: None)
    return policy


def _policy_spec(startup_mode: str) -> dict:
    return {
        "name": "wbt-test",
        "implementation": "wbt",
        "type": "full_body",
        "inputs": [],
        "observation": {
            "obs_dict": {"actor_obs": ["dof_pos"]},
            "obs_dims": {"dof_pos": 2},
            "obs_scales": {"dof_pos": 1.0},
            "history_length_dict": {"actor_obs": 1},
        },
        "task": {"model_path": "wbt.onnx", "startup_mode": startup_mode},
    }


@pytest.mark.parametrize("startup_mode", ["interpolate", "immediate"])
def test_wbt_startup_mode_is_strictly_configured(startup_mode):
    spec = PolicySpec.model_validate(_policy_spec(startup_mode))

    assert isinstance(spec.task, WbtTaskConfig)
    assert spec.task.startup_mode == startup_mode


def test_wbt_startup_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="startup_mode"):
        PolicySpec.model_validate(_policy_spec("prefill"))


def test_wbt_immediate_start_enables_policy_without_initialization():
    policy = _policy("immediate")
    started = []
    policy._handle_start_policy = lambda state: started.append(state)

    assert policy.activate(_state()) is None

    assert len(started) == 1
    assert policy._activation_q is None
    np.testing.assert_allclose(policy.last_policy_action, 0.0)
    assert not policy.obs_history_buffers["actor_obs"]["dof_pos"]
    np.testing.assert_allclose(policy.obs_buf_dict["actor_obs"], 0.0)
    np.testing.assert_allclose(policy.motion_command_t, policy.motion_command_0)


def test_wbt_interpolates_from_fixed_activation_pose_then_starts_policy():
    policy = _policy("interpolate")
    started = []
    policy._handle_start_policy = lambda state: started.append(state)

    assert policy.activate(_state((0.0, 0.0))) is None
    assert policy.get_ready_state
    assert policy.use_policy_action
    assert not policy._stiff_hold_active

    first = policy.get_init_target(_state((0.25, 0.5)))
    second = policy.get_init_target(_state((0.75, 1.5)))

    np.testing.assert_allclose(first, [[0.5, 1.0]])
    np.testing.assert_allclose(second, [[1.0, 2.0]])
    assert len(started) == 1
    assert policy._activation_q is None


def test_wbt_interpolation_applies_position_and_velocity_limits():
    policy = _policy("interpolate")
    policy._startup_interpolator = JointPositionInterpolator(
        joint_lower=(-0.1, -0.2),
        joint_upper=(0.1, 0.2),
        joint_velocity=(0.2, 0.4),
        rate_hz=2.0,
        duration_s=1.0,
        slew_safety_factor=1.0,
    )
    policy._handle_start_policy = lambda state: None

    assert policy.activate(_state()) is None
    first = policy.get_init_target(_state())

    np.testing.assert_allclose(first, [[0.1, 0.2]])
