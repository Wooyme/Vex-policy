from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER, G1_JOINT_VELOCITY
from vex_policy.robots.g1 import DOF_NAMES
from vex_policy.utils.joint_interpolation import JointPositionInterpolator, limit_joint_position_target


def _interpolator(*, velocity=(100.0, 100.0), factor=1.0):
    return JointPositionInterpolator(
        joint_lower=(-1.0, -2.0),
        joint_upper=(1.0, 2.0),
        joint_velocity=velocity,
        rate_hz=2.0,
        duration_s=1.0,
        slew_safety_factor=factor,
    )


def test_g1_joint_constraints_follow_hardware_order():
    assert len(G1_JOINT_LOWER) == len(DOF_NAMES) == 29
    assert len(G1_JOINT_UPPER) == len(DOF_NAMES)
    assert len(G1_JOINT_VELOCITY) == len(DOF_NAMES)
    assert np.all(np.asarray(G1_JOINT_LOWER) < np.asarray(G1_JOINT_UPPER))
    assert np.all(np.asarray(G1_JOINT_VELOCITY) > 0.0)


@pytest.mark.parametrize("factor", [0.0, 1.01, np.inf])
def test_robot_interpolation_slew_factor_is_strictly_bounded(factor):
    with pytest.raises(ValueError, match="joint_interpolation_slew_safety_factor"):
        replace(G1_29DOF, joint_interpolation_slew_safety_factor=factor)


def test_interpolator_uses_fixed_start_and_finishes_on_configured_step():
    interpolator = _interpolator()
    interpolator.reset((0.0, 0.0), (1.0, 2.0))

    first = interpolator.next((0.25, 0.25))
    second = interpolator.next((0.75, 1.5))

    np.testing.assert_allclose(first.q_target, (0.5, 1.0))
    assert not first.complete
    np.testing.assert_allclose(second.q_target, (1.0, 2.0))
    assert second.complete


def test_interpolator_clips_position_and_limits_each_command_delta():
    interpolator = _interpolator(velocity=(0.4, 0.8), factor=0.5)
    interpolator.reset((0.0, 0.0), (10.0, -10.0))

    first = interpolator.next((0.0, 0.0))
    second = interpolator.next((0.0, 0.0))

    np.testing.assert_allclose(first.q_target, (0.1, -0.2))
    np.testing.assert_allclose(second.q_target, (0.2, -0.4))
    assert second.complete


def test_interpolator_completes_on_time_when_slew_limit_prevents_arrival():
    interpolator = _interpolator(velocity=(0.1, 0.1), factor=0.5)
    interpolator.reset((0.0, 0.0), (1.0, 1.0))

    interpolator.next((0.0, 0.0))
    final = interpolator.next((0.0, 0.0))

    assert final.complete
    assert np.all(final.q_target < 1.0)


def test_interpolator_clear_requires_a_new_reset():
    interpolator = _interpolator()
    interpolator.reset((0.0, 0.0), (1.0, 1.0))
    interpolator.clear()

    with pytest.raises(RuntimeError, match="has not been reset"):
        interpolator.next((0.0, 0.0))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("q_target", (np.nan, 0.0), "non-finite"),
        ("current_q", (0.0,), "shape"),
        ("dt", 0.0, "positive"),
        ("slew_safety_factor", -0.1, "non-negative"),
    ],
)
def test_target_limiter_rejects_invalid_inputs(field, value, reason):
    arguments = {
        "q_target": (0.0, 0.0),
        "current_q": (0.0, 0.0),
        "previous_q": None,
        "joint_lower": (-1.0, -1.0),
        "joint_upper": (1.0, 1.0),
        "joint_velocity": (1.0, 1.0),
        "dt": 0.02,
        "slew_safety_factor": 0.5,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=reason):
        limit_joint_position_target(**arguments)
