"""Robot communication package."""

from vex_policy.sdk.high_frequency_logger import HighFrequencyLogConfig, HighFrequencyLogger

__all__ = ["HighFrequencyLogConfig", "HighFrequencyLogger", "InterfaceManager"]


def __getattr__(name: str):
    if name == "InterfaceManager":
        from vex_policy.sdk.interface_manager import InterfaceManager

        return InterfaceManager
    raise AttributeError(name)
