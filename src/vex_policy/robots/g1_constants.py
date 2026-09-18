"""Unitree G1 29-DOF hardware joint constraints.

All lists use the hardware joint order declared by ``vex_policy.robots.g1.DOF_NAMES``.
Values are loaded from the joint-name mapping in ``g1.yaml``.
"""

from ._g1_config import JOINT_PARAMETERS

G1_JOINT_LOWER = JOINT_PARAMETERS["lower"]
G1_JOINT_UPPER = JOINT_PARAMETERS["upper"]
G1_JOINT_VELOCITY = JOINT_PARAMETERS["velocity"]

__all__ = ["G1_JOINT_LOWER", "G1_JOINT_UPPER", "G1_JOINT_VELOCITY"]
