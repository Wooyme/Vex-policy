"""Robot communication package."""

from __future__ import annotations

from vex_policy.compat import entry_points

_entry_points = {ep.name: ep for ep in entry_points(group="vex_policy.sdk")}
_registry = {}


def create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True):
    """Create interface from registry.

    If *interface_str* is ``"auto"``, the network interface is resolved
    automatically via :func:`vex_policy.utils.network.detect_robot_interface`.
    """
    sdk_type = robot_config.sdk_type
    if sdk_type not in _registry:
        if sdk_type in _entry_points:
            _registry[sdk_type] = _entry_points[sdk_type].load()
        elif sdk_type == "unitree":
            from vex_policy.sdk.unitree.unitree_interface import UnitreeInterface

            _registry[sdk_type] = UnitreeInterface
        elif sdk_type == "unitree_mp":
            from vex_policy.sdk.unitree.unitree_interface_mp import UnitreeInterfaceMP

            _registry[sdk_type] = UnitreeInterfaceMP
        else:
            available = sorted({"unitree", "unitree_mp", *_entry_points})
            raise ValueError(f"Unknown sdk_type: {sdk_type}. Available: {available}")

    return _registry[sdk_type](robot_config, domain_id, interface_str, use_joystick)
