from vex_policy.config.config_types.GuardConfig import GuardConfig
from vex_policy.policies import BasePolicy


class BaseGuard:
    def __init__(self, config: GuardConfig, policy: BasePolicy):
        self.policy = policy
        self.config = config

    def start_check(self)-> tuple[bool,str|None]:
        raise NotImplementedError
