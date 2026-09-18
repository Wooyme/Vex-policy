from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vex_policy.config.config_types import (
    EmergencyStopConfig,
    HoldPositionTaskConfig,
    InferenceConfig,
    LimiterConfig,
    ObservationConfig,
)
from vex_policy.config.config_types.safety import Bounds
from vex_policy.policies.base import PolicyJointCommand, PolicyRuntimeFault
from vex_policy.policies.emergency_stop import EmergencyStop
from vex_policy.policies.hold_position import HoldPositionPolicy
from vex_policy.policies.limiter import JointCommandLimiter
from vex_policy.robots import G1_29DOF
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import rpy_to_quat

NAMES = G1_29DOF.dof_names


def state():
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.array([[1.0, 0, 0, 0]]),
        joint_pos=np.zeros((1, 29)),
        joint_vel=np.zeros((1, 29)),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
    )


def command():
    return PolicyJointCommand(
        q=np.full(29, 0.5),
        dq=np.full(29, -5.0),
        tau=np.ones(29),
        kp=np.full(29, 2.0),
        kd=np.full(29, 3.0),
        controlled_joints=np.ones(29, dtype=bool),
    )


def test_limiter_names_offsets_mask_and_no_slew():
    robot = replace(G1_29DOF, dof_names=NAMES[::-1], joint_offsets_deg=[10.0] * 29)
    config = LimiterConfig(joints={NAMES[0]: {"pos": {"min": -0.1, "max": 0.2}, "vel": 1.5}})
    limiter = JointCommandLimiter(config, robot)
    original = command()
    bounded = limiter.postprocess(original)
    assert bounded.q[-1] + np.deg2rad(10) == pytest.approx(0.2)
    assert bounded.dq[-1] == -1.5
    np.testing.assert_array_equal(bounded.q[:-1], original.q[:-1])
    np.testing.assert_array_equal(bounded.dq[:-1], original.dq[:-1])
    np.testing.assert_array_equal(original.q, 0.5)
    np.testing.assert_array_equal(original.dq, -5.0)
    for field in ("kp", "kd", "tau", "controlled_joints"):
        np.testing.assert_array_equal(getattr(bounded, field), getattr(original, field))
    masked = original.controlled_joints.copy()
    masked[-1] = False
    np.testing.assert_array_equal(limiter.postprocess(replace(original, controlled_joints=masked)).q, original.q)
    # Only dq is speed-limited: a large position jump within bounds passes immediately.
    changed = replace(original, q=np.full(29, -0.2), dq=np.full(29, 5.0))
    assert limiter.postprocess(changed).q[-1] == pytest.approx(-0.2)
    assert limiter.postprocess(changed).dq[-1] == 1.5


def test_hardware_intersection_and_velocity_only():
    limiter = JointCommandLimiter(
        LimiterConfig(joints={NAMES[0]: {"pos": {"min": -10, "max": 10}, "vel": 100}}), G1_29DOF
    )
    bounded = limiter.postprocess(replace(command(), q=np.full(29, 10.0), dq=np.full(29, 100.0)))
    assert bounded.q[0] == 2.8798
    assert bounded.dq[0] == 32.0
    with pytest.raises(ValueError, match="intersect"):
        JointCommandLimiter(LimiterConfig(joints={NAMES[0]: {"pos": {"min": 5, "max": 6}}}), G1_29DOF)
    limiter = JointCommandLimiter(LimiterConfig(joints={NAMES[0]: {"vel": 0}}), G1_29DOF)
    bounded = limiter.postprocess(command())
    np.testing.assert_array_equal(bounded.q, command().q)
    assert bounded.dq[0] == 0


@pytest.mark.parametrize(
    "field,value", [("q", np.full(29, np.nan)), ("dq", np.zeros(28)), ("controlled_joints", np.ones(29))]
)
def test_limiter_rejects_invalid_commands(field, value):
    limiter = JointCommandLimiter(LimiterConfig(), G1_29DOF)
    with pytest.raises(ValueError, match=field):
        limiter.postprocess(replace(command(), **{field: value}))


def test_base_policy_applies_limiter_and_converts_faults():
    config = InferenceConfig(
        robot=G1_29DOF,
        inputs=(),
        observation=ObservationConfig({}, {}, {}, {}),
        task=HoldPositionTaskConfig(),
        limiter=LimiterConfig(joints={NAMES[0]: {"pos": {"min": -0.1, "max": 0.1}}}),
    )
    policy = HoldPositionPolicy(config)
    robot_state = state()
    robot_state.joint_pos[0, 0] = 0.5
    assert policy.activate(robot_state) is None
    assert policy.step(robot_state).q[0] == 0.1
    assert policy.held_dof_pos[0] == 0.5
    policy._compute_command = lambda _: replace(command(), dq=np.full(29, np.inf))
    with pytest.raises(PolicyRuntimeFault, match="invalid dq"):
        policy.step(robot_state)
    policy.close()


def test_estop_independent_checks_boundaries_and_joint_order():
    config = EmergencyStopConfig(joints={NAMES[0]: {"pos": {"min": -0.2, "max": 0.3}, "vel": 2.0}})
    checker = EmergencyStop(config, NAMES[::-1])
    robot_state = state()
    robot_state.joint_pos[0, -1] = 0.3
    robot_state.joint_vel[0, -1] = -2.0
    assert checker.check(robot_state) == ()
    robot_state.joint_pos[0, -1] = -0.21
    robot_state.joint_vel[0, -1] = -2.01
    violations = checker.check(robot_state)
    assert len(violations) == 2
    assert violations[0].field == f"{NAMES[0]}.pos"
    assert violations[1].value == -2.01
    assert violations[1].lower == -2.0
    robot_state.base_quat.fill(0)  # Orientation isn't configured here.
    assert len(checker.check(robot_state)) == 2


@pytest.mark.parametrize("axis,index", [("roll", 0), ("pitch", 1), ("yaw", 2)])
def test_rpy_world_frame_and_normalization(axis, index):
    checker = EmergencyStop(EmergencyStopConfig(rpy={axis: {"min": -0.3, "max": 0.3}}), NAMES)
    robot_state = state()
    angle = np.zeros(3)
    angle[index] = 0.5
    robot_state.base_quat[:] = rpy_to_quat(angle) * 2
    before = robot_state.base_quat.copy()
    (violation,) = checker.check(robot_state)
    assert violation.field == f"rpy.{axis}"
    assert violation.value == pytest.approx(0.5)
    np.testing.assert_array_equal(robot_state.base_quat, before)
    robot_state.base_quat[:] = [1, 0, 0, 0]
    assert checker.check(robot_state) == ()


def test_estop_invalid_state_takes_precedence_over_finite_violation():
    checker = EmergencyStop(
        EmergencyStopConfig(
            joints={NAMES[0]: {"pos": {"min": -0.1, "max": 0.1}, "vel": 1}}, rpy={"roll": {"min": -0.5, "max": 0.5}}
        ),
        NAMES,
    )
    robot_state = state()
    robot_state.joint_pos[0, 0] = 1.0
    for field, value in (
        ("base_quat", np.zeros((1, 4))),
        ("joint_vel", np.full((1, 29), np.nan)),
        ("joint_pos", np.zeros((1, 28))),
    ):
        data = {name: getattr(robot_state, name) for name in ("base_quat", "joint_vel", "joint_pos")}
        data[field] = value
        (violation,) = checker.check(SimpleNamespace(**data))
        assert violation.invalid
        assert violation.field == field


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_velocity_config(value):
    with pytest.raises(ValueError):
        LimiterConfig(joints={NAMES[0]: {"vel": value}})


@pytest.mark.parametrize("bounds", [{"min": 1, "max": 0}, {"min": np.nan, "max": 1}, {"min": -1, "max": np.inf}])
def test_invalid_position_config(bounds):
    with pytest.raises(ValueError):
        Bounds(**bounds)


def test_unknown_names_and_orientation_range():
    for cls, config in (
        (JointCommandLimiter, LimiterConfig(joints={"typo": {"vel": 1}})),
        (EmergencyStop, EmergencyStopConfig(joints={"typo": {"vel": 1}})),
    ):
        with pytest.raises(ValueError, match="typo"):
            cls(config, G1_29DOF if cls is JointCommandLimiter else NAMES)
    with pytest.raises(ValueError, match="pitch"):
        EmergencyStopConfig(rpy={"pitch": {"min": -2, "max": 2}})
