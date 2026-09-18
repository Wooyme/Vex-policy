from abc import ABC, abstractmethod

from vex_policy.sdk.base.base_interface import LowState


class BaseGuard(ABC):
    """Policy-independent startup check interface."""

    @abstractmethod
    def start_check(self, robot_state_data: LowState) -> tuple[bool, str | None]:
        raise NotImplementedError
