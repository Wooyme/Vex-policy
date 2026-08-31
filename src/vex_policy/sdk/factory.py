"""Internal SDK backend construction."""

from __future__ import annotations

from vex_policy.compat import entry_points

_entry_points = {ep.name: ep for ep in entry_points(group="vex_policy.sdk")}
_registry = {}


def _create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True):
    """Construct one backend interface for :class:`InterfaceManager`."""
    sdk_type = robot_config.sdk_type
    if sdk_type not in _registry:
        if sdk_type in _entry_points:
            _registry[sdk_type] = _entry_points[sdk_type].load()
        elif sdk_type == "unitree":
            from vex_policy.sdk.unitree.unitree_interface import UnitreeInterface

            _registry[sdk_type] = UnitreeInterface
        else:
            available = sorted({"unitree", *_entry_points})
            raise ValueError(f"Unknown sdk_type: {sdk_type}. Available: {available}")

    return _registry[sdk_type](robot_config, domain_id, interface_str, use_joystick)
