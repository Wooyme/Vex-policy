"""Runtime switching among MQTT-selected policy instances."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from vex_policy.compat import entry_points
from vex_policy.config import ResolvedPolicy
from vex_policy.config.config_types import RuntimeConfig
from vex_policy.mqtt import CommandInbox, MqttTransport, encode_robot_state
from vex_policy.policies.base import BasePolicy
from vex_policy.policies.locomotion import LocomotionPolicy
from vex_policy.policies.sonic import SonicPolicy
from vex_policy.policies.wbt import WholeBodyTrackingPolicy
from vex_policy.utils.rate import RateLimiter


def _policy_class(kind: str) -> type[BasePolicy]:
    for ep in entry_points(group="vex_policy.policies"):
        if ep.name == kind:
            return ep.load()
    if kind == "locomotion":
        return LocomotionPolicy
    if kind == "wbt":
        return WholeBodyTrackingPolicy
    if kind == "sonic":
        return SonicPolicy
    raise ValueError(f"Unknown policy kind: {kind}")


class SwitchModePolicy:
    """Own one interface and run either one full-body or an upper/lower pair."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        resolved: tuple[ResolvedPolicy, ...],
        *,
        instances: dict[str, BasePolicy] | None = None,
        inbox: CommandInbox | None = None,
        transport: MqttTransport | None = None,
        interface: Any | None = None,
        clock=time.monotonic,
    ):
        self.runtime = runtime
        self.resolved = resolved
        self._clock = clock
        self._started_at = clock()
        self._specs = {item.spec.name: item.spec for item in resolved}
        self.inbox = inbox or CommandInbox(self._specs, clock=clock)
        self.transport = transport or MqttTransport(runtime.mqtt, tuple(item.spec for item in resolved), self.inbox)
        self._injected_interface = interface
        self.policies = instances or self._build_policies()
        if set(self.policies) != set(self._specs):
            raise ValueError("Policy instances must exactly match configured policy names")
        owner = next(iter(self.policies.values()))
        self.interface = owner.interface
        self.dof_names = tuple(owner.dof_names)
        for policy in self.policies.values():
            policy._manager_owns_interface_lifecycle = True

        rate = resolved[0].config.task.rl_rate
        self.rate = RateLimiter(rate)
        self._state_period = 1.0 / runtime.mqtt.state_frequency_hz
        self._next_state_publish = self._started_at
        self.state = "startup_latched"
        self.active_policy: tuple[str, ...] = ()
        self.requested_policy: tuple[str, ...] = ()
        self.reason: str | None = "startup"
        self.last_command_seq: int | None = None
        self._last_status: tuple[Any, ...] | None = None

    def _build_policies(self) -> dict[str, BasePolicy]:
        instances: dict[str, BasePolicy] = {}
        owner: BasePolicy | None = None
        for item in self.resolved:
            cls = _policy_class(item.kind)
            policy = object.__new__(cls)
            if owner is None:
                policy._runtime_domain_id = self.runtime.robot.domain_id
                policy._runtime_interface = self.runtime.robot.interface
                if self._injected_interface is not None:
                    policy._injected_interface = self._injected_interface
            else:
                policy._shared_hardware_source = owner
            cls.__init__(policy, item.config)
            owner = owner or policy
            instances[item.spec.name] = policy
            logger.info(f"Preloaded policy {item.spec.name}: {cls.__name__}")
        return instances

    def _status_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "active_policy": list(self.active_policy),
            "requested_policy": list(self.requested_policy),
            "reason": self.reason,
            "last_command_seq": self.last_command_seq,
        }

    def _publish_status(self, *, force: bool = False) -> None:
        payload = self._status_payload()
        marker = tuple(payload.values())
        if force or marker != self._last_status:
            self.transport.publish_status(payload)
            self._last_status = marker

    def _deactivate(self) -> None:
        if not self.active_policy:
            return
        for name in self.active_policy:
            self.policies[name].deactivate()
        self.active_policy = ()
        stop_writer = getattr(self.interface, "stop_command_writer", None)
        if stop_writer is not None:
            stop_writer()
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 1

    def _latch(self, reason: str) -> None:
        self._deactivate()
        self.state = "latched"
        self.reason = reason

    def _selection_key(self, name: str) -> tuple[int, str]:
        order = {"full_body": 0, "lower_body": 0, "upper_body": 1}
        return order[self._specs[name].type], name

    def _canonical_selection(self, names) -> tuple[str, ...]:
        return tuple(sorted(names, key=self._selection_key))

    def _configure_writer(self, names: tuple[str, ...]) -> None:
        configure = getattr(self.interface, "configure_writer", None)
        if configure is None:
            return
        tasks = [self.policies[name].config.task for name in names]
        publish_rate = max(getattr(task, "lowcmd_publish_rate", 500.0) for task in tasks)
        timeout = min(getattr(task, "low_state_timeout_s", 0.1) for task in tasks)
        configure(publish_rate, timeout)

    def _activate(self, names: tuple[str, ...]) -> None:
        names = self._canonical_selection(names)
        self.state = "switching"
        self.reason = None
        self._publish_status()

        previous = set(self.active_policy)
        desired = set(names)
        for name in self.active_policy:
            if name not in desired:
                self.policies[name].deactivate()

        activated: list[str] = []
        for name in names:
            if name in previous:
                continue
            reason = self.policies[name].activate()
            if reason:
                for activated_name in activated:
                    self.policies[activated_name].deactivate()
                for retained_name in previous & desired:
                    self.policies[retained_name].deactivate()
                self.active_policy = ()
                stop_writer = getattr(self.interface, "stop_command_writer", None)
                if stop_writer is not None:
                    stop_writer()
                if hasattr(self.interface, "no_action"):
                    self.interface.no_action = 1
                self.state = "latched"
                self.reason = reason
                self._publish_status()
                return
            activated.append(name)

        self.active_policy = names
        self._configure_writer(names)
        if not previous:
            start_writer = getattr(self.interface, "start_command_writer", None)
            if start_writer is not None:
                start_writer()
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.state = "running"

    def _on_command_sent(self, cmd_q, robot_state) -> None:
        del cmd_q
        self._maybe_publish_state(robot_state)

    def _maybe_publish_state(self, robot_state, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        if current + 1e-9 < self._next_state_publish:
            return
        timestamp = time.time()
        payload = encode_robot_state(
            robot_state,
            self.dof_names,
            started_at=self._started_at,
            monotonic_now=current,
            timestamp=timestamp,
        )
        self.transport.publish_state(payload)
        if len(self.active_policy) == 1:
            policy = self.policies[self.active_policy[0]]
            get_reference_state = getattr(policy, "get_reference_state", None)
            reference_state = get_reference_state() if get_reference_state is not None else None
            if reference_state is not None:
                reference_payload = encode_robot_state(
                    reference_state,
                    self.dof_names,
                    started_at=self._started_at,
                    monotonic_now=current,
                    timestamp=timestamp,
                )
                self.transport.publish_reference_state(reference_payload)
        self._next_state_publish = current + self._state_period

    def _publish_idle_state(self, now: float) -> None:
        if now + 1e-9 < self._next_state_publish:
            return
        robot_state = self.interface.get_low_state()
        if robot_state is not None:
            self._maybe_publish_state(robot_state, now)

    def tick(self, now: float | None = None) -> None:
        """Run one deterministic state-machine/control iteration."""
        current = self._clock() if now is None else now
        received = self.inbox.snapshot()
        if received is None:
            self._publish_idle_state(current)
            self._publish_status()
            return

        packet = received.packet
        control = packet.control
        self.last_command_seq = packet.seq
        self.requested_policy = self._canonical_selection(control.policy)

        if current - received.received_at > self.runtime.mqtt.command_timeout_s:
            self._latch("command_timeout")
            self._publish_idle_state(current)
            self._publish_status()
            return
        if control.estop:
            self._latch("estop")
            self._publish_idle_state(current)
            self._publish_status()
            return

        if self.state in {"startup_latched", "latched"}:
            if not control.policy:
                self.state = "idle"
                self.reason = None
            self._publish_idle_state(current)
            self._publish_status()
            return

        if not control.policy:
            self._deactivate()
            self.state = "idle"
            self.reason = None
            self._publish_idle_state(current)
            self._publish_status()
            return

        desired = self._canonical_selection(control.policy)
        if desired != self.active_policy:
            self._activate(desired)
            self._publish_idle_state(current)
            self._publish_status()
            return  # deliberate one-cycle low-command gap during a switch

        for name in desired:
            policy = self.policies[name]
            spec = self._specs[name]
            policy.apply_control(control.values_for(spec.inputs))
        if not self._step_active_policies():
            self._publish_idle_state(current)
            self._publish_status()
            return
        self.state = "running"
        self.reason = None
        self._publish_status()

    def run(self) -> None:
        self.transport.start()
        self._publish_status(force=True)
        try:
            while True:
                self.tick()
                self.rate.sleep()
        except KeyboardInterrupt:
            logger.info("Policy runtime interrupted")
        finally:
            self._deactivate()
            for policy in self.policies.values():
                policy.close()
            close_interface = getattr(self.interface, "close", None)
            if close_interface is not None:
                close_interface()
            self.transport.close()

    def _step_active_policies(self) -> bool:
        """Infer all active policies from one state snapshot and publish one merged command."""
        robot_state = self.interface.get_low_state()
        if robot_state is None:
            self._latch("low_state_unavailable")
            return False

        commands = []
        for name in self.active_policy:
            policy = self.policies[name]
            policy.latency_tracker.start_cycle()
            if policy.use_phase:
                policy.update_phase_time()
            commands.append(policy.compute_joint_command(robot_state))
            policy.latency_tracker.end_cycle()

        owner = self.policies[self.active_policy[0]]
        num_dofs = owner.num_dofs
        q = np.asarray(owner.default_dof_angles, dtype=np.float64).copy()
        dq = np.zeros(num_dofs, dtype=np.float64)
        tau = np.zeros(num_dofs, dtype=np.float64)
        kp = commands[0].kp.copy()
        kd = commands[0].kd.copy()

        for command in commands:
            selected = command.controlled_joints
            q[selected] = command.q[selected]
            dq[selected] = command.dq[selected]
            tau[selected] = command.tau[selected]
            kp[selected] = command.kp[selected]
            kd[selected] = command.kd[selected]

        q += owner.joint_offsets
        self.interface.send_low_command(
            q,
            dq,
            tau,
            robot_state[0, 7 : 7 + num_dofs],
            kp_override=kp,
            kd_override=kd,
        )
        self._maybe_publish_state(robot_state)
        return True


__all__ = ["SwitchModePolicy"]
