from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GuardConfig:
    bad_lower_joint_pos_threshold: float = 0.8
    bad_ref_ori_threshold: float = 0.0
