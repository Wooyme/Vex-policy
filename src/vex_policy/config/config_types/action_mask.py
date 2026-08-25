"""Per-policy model-output masking configuration."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ActionMaskConfig:
    """Joint names whose residual model outputs are forced to zero."""

    masked_joints: tuple[str, ...]
