from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
import yaml
from loguru import logger

from vex_policy.config.config_types import GuardConfig, PolicySpec
from vex_policy.policies.guard.initial_pose import InitialPoseGuard
from vex_policy.policies.utils.initial_pose import InitialPose
from vex_policy.policies.utils.locomotion_utils import load_motion_last_pose
from vex_policy.policies.utils.wbt_utils import PinocchioRobot
from vex_policy.policies.wbt import WholeBodyTrackingPolicy
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import rpy_to_quat


def _pose(**changes):
    return InitialPose(
        **dict(dof_names=("knee", "shoulder"), dof_pos=(0.0, 0.0), root_quat_wxyz=(1, 0, 0, 0)) | changes
    )


def _state(q=(0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.asarray([quat], dtype=float),
        joint_pos=np.asarray([q], dtype=float),
        joint_vel=np.zeros((1, len(q))),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
    )


def _guard(pose=None, names=("knee", "shoulder"), **config):
    return InitialPoseGuard(GuardConfig(**config), pose or _pose(), names, logger)


def test_joint_name_reordering_and_per_joint_thresholds():
    pose = _pose(dof_names=("shoulder", "knee"), dof_pos=(1.0, 0.0))
    guard = _guard(pose, startup_joint_tolerances_rad={"knee": 0.5})
    assert guard.start_check(_state((0.5, 1.0))) == (True, None)
    assert not guard.start_check(_state((0.5001, 1.0)))[0]
    # The largest absolute error (knee) passes, but the smaller shoulder error fails.
    accepted, reason = guard.start_check(_state((0.4, 1.3)))
    assert not accepted
    assert "shoulder error=0.300rad > 0.200rad" in reason


def test_tight_override_and_default_boundary():
    guard = _guard(startup_joint_tolerances_rad={"shoulder": 0.05})
    assert guard.start_check(_state((0.2, 0.05))) == (True, None)
    assert "shoulder" in guard.start_check(_state((0.1, 0.051)))[1]
    assert "knee" in guard.start_check(_state((0.201, 0.0)))[1]


@pytest.mark.parametrize("value", [0, -0.1, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "field", ["startup_joint_tolerance_rad", "startup_gravity_tolerance", "startup_joint_tolerances_rad"]
)
def test_invalid_thresholds(field, value):
    with pytest.raises(ValueError, match=field):
        GuardConfig(**{field: {"knee": value} if field.endswith("tolerances_rad") else value})


@pytest.mark.parametrize(
    "changes",
    [
        {"dof_names": ()},
        {"dof_names": ("knee", "knee")},
        {"dof_names": ("", "shoulder")},
        {"dof_pos": (0.0,)},
        {"dof_pos": (np.nan, 0.0)},
        {"root_quat_wxyz": (0, 0, 0, 0)},
        {"root_quat_wxyz": (np.inf, 0, 0, 0)},
    ],
)
def test_invalid_reference_pose(changes):
    with pytest.raises(ValueError):
        _pose(**changes)


def test_unknown_override_and_mismatched_joint_sets():
    with pytest.raises(ValueError, match="typo"):
        _guard(startup_joint_tolerances_rad={"typo": 0.1})
    with pytest.raises(ValueError, match=r"missing=.*ankle.*extra=.*shoulder"):
        _guard(names=("knee", "ankle"))
    with pytest.raises(ValueError, match="unique"):
        _guard(names=("knee", "knee"))


@pytest.mark.parametrize("field", ["joint_pos", "joint_vel", "base_ang_vel", "base_quat"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_invalid_state_values(field, value):
    state = _state()
    getattr(state, field)[0, 0] = value
    accepted, reason = _guard().start_check(state)
    assert not accepted
    assert f"invalid_state:{field}" in reason


def test_invalid_state_joint_count_and_zero_quaternion():
    assert "invalid_state:joint_pos" in _guard().start_check(_state(q=(0.0,)))[1]
    assert "invalid_state:base_quat" in _guard().start_check(_state(quat=(0, 0, 0, 0)))[1]


def test_normalization_yaw_invariance_and_no_input_mutation():
    reference = rpy_to_quat((0.4, -0.2, 0))
    pose = _pose(root_quat_wxyz=reference * 3)
    np.testing.assert_allclose(pose.root_quat_wxyz, reference)
    state = _state(quat=rpy_to_quat((0.4, -0.2, 1.2)) * 2)
    original = state.base_quat.copy()
    assert _guard(pose).start_check(state) == (True, None)
    np.testing.assert_array_equal(state.base_quat, original)


def test_full_gravity_vector_and_boundary():
    pose = _pose(root_quat_wxyz=rpy_to_quat((0.6, 0, 0)))
    guard = _guard(pose)
    opposite = _pose(root_quat_wxyz=rpy_to_quat((-0.6, 0, 0)))
    assert pose.projected_gravity[2] == opposite.projected_gravity[2]
    assert "projected_gravity" in guard.start_check(_state(quat=opposite.root_quat_wxyz))[1]
    inverted = _state(quat=(0, 1, 0, 0))
    assert _guard(startup_gravity_tolerance=2.0).start_check(inverted) == (True, None)
    assert not _guard(startup_gravity_tolerance=1.99).start_check(inverted)[0]


def test_motion_final_frame_and_fixed_reference(tmp_path):
    path = tmp_path / "motion.npz"
    q = rpy_to_quat((0.3, 0.1, 0.5))
    np.savez(
        path,
        joint_names=["shoulder", "knee"],
        joint_pos=np.asarray([[0, 0, 0, 1, 0, 0, 0, 9, 9], [0, 0, 0, *(q * 2), 1, 2]]),
    )
    pose = load_motion_last_pose(path)
    guard = _guard(pose)
    state = _state(q=(2, 1), quat=q)
    assert guard.start_check(state) == (True, None)
    state.joint_pos[0, 0] += 1
    assert not guard.start_check(state)[0]
    assert guard.start_check(_state(q=(2, 1), quat=q)) == (True, None)


def test_wbt_reference_link_to_base_with_real_fk():
    urdf = """<robot name="guard_test">
      <link name="base"/><link name="torso_link"/><link name="arm"/>
      <joint name="waist_pitch_joint" type="revolute">
        <parent link="base"/><child link="torso_link"/><axis xyz="0 1 0"/>
        <limit lower="-2" upper="2" effort="10" velocity="10"/>
      </joint>
      <joint name="arm_joint" type="revolute">
        <parent link="torso_link"/><child link="arm"/><axis xyz="1 0 0"/>
        <limit lower="-2" upper="2" effort="10" velocity="10"/>
      </joint>
    </robot>"""
    names = ("arm_joint", "waist_pitch_joint")
    model = PinocchioRobot(SimpleNamespace(dof_names=names, motion={"body_name_ref": ["torso_link"]}), urdf)
    # Noncommuting rotations catch inversion and multiplication-order mistakes.
    base_rotation = pin.rpy.rpyToMatrix(0.3, -0.2, 0.7)
    ref_rotation = base_rotation @ pin.rpy.rpyToMatrix(0, 0.6, 0)
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    policy.num_dofs = 2
    policy.dof_names = names
    policy.pinocchio_robot = model
    policy.motion_command_0 = np.array([[0.1, 0.6, 0, 0]])
    policy.ref_quat_xyzw_0 = pin.Quaternion(ref_rotation).coeffs()[None, :]
    pose = policy._load_initial_pose()
    np.testing.assert_allclose(pose.root_quat_wxyz, rpy_to_quat((0.3, -0.2, 0.7)), atol=1e-12)
    guard = _guard(pose, names=names)
    state = _state(q=(0.1, 0.6), quat=pose.root_quat_wxyz)
    assert guard.start_check(state) == (True, None)
    # The reference-link attitude itself must not be mistaken for the base attitude.
    assert not guard.start_check(
        replace(state, base_quat=pin.Quaternion(ref_rotation).coeffs()[[3, 0, 1, 2]][None, :])
    )[0]


@pytest.mark.parametrize("implementation", ["ufo", "waist_locomotion", "passive_locomotion", "wbt"])
def test_unified_yaml_guard_config(implementation):
    examples = {
        "ufo": "configs/examples/ufo/g1_ufo_goal.yaml",
        "waist_locomotion": "configs/examples/g1_waist_locomotion.yaml",
        "passive_locomotion": "configs/g1/g1_passive_locomotion.yaml",
        "wbt": "configs/examples/holosoma/g1_ppo_wbt_dancing.yaml",
    }
    data = yaml.safe_load((Path(__file__).resolve().parents[1] / examples[implementation]).read_text())
    data["guard"] = {"startup_joint_tolerances_rad": {"left_knee_joint": 0.3}}
    spec = PolicySpec.model_validate(data)
    assert type(spec.guard) is GuardConfig
    assert spec.guard.startup_joint_tolerance_rad == 0.2
    assert spec.guard.startup_joint_tolerances_rad == {"left_knee_joint": 0.3}
    for field in ("bad_lower_joint_pos_threshold", "bad_ref_ori_threshold"):
        with pytest.raises(ValueError, match=field):
            PolicySpec.model_validate({**data, "guard": {field: 0.2}})
