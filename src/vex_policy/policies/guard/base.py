from vex_policy.config.config_types.GuardConfig import GuardConfig
from vex_policy.policies.base import BasePolicy
from vex_policy.sdk.base.base_interface import LowState


class BaseGuard:
    def __init__(self, config: GuardConfig, policy: BasePolicy):
        self.policy = policy
        self.config = config

    def start_check(self, robot_state_data: LowState) -> tuple[bool, str | None]:
        raise NotImplementedError
