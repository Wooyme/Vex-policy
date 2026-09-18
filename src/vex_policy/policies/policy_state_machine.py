"""MQTT-driven policy control state machine."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from enum import StrEnum
from typing import Any

import numpy as np
from loguru import logger

from vex_policy.compat import entry_points
from vex_policy.config import ResolvedPolicy
from vex_policy.config.config_types import RuntimeConfig
from vex_policy.mqtt import CommandInbox, MqttTransport, encode_robot_state
from vex_policy.policies.base import BasePolicy, PolicyRuntimeFault
from vex_policy.policies.hold_position import HoldPositionPolicy
from vex_policy.policies.interpolation import InterpolationPolicy
from vex_policy.policies.locomotion import LocomotionPolicy
from vex_policy.policies.passive_locomotion import PassiveLocomotionPolicy
from vex_policy.policies.sonic import SonicPolicy
from vex_policy.policies.ufo import UfoPolicy
from vex_policy.policies.waist_locomotion import WaistLocomotionPolicy
from vex_policy.policies.wbt import WholeBodyTrackingPolicy
from vex_policy.sdk import InterfaceManager
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.rate import RateLimiter


def _policy_class(kind: str) -> type[BasePolicy]:
    for ep in entry_points(group="vex_policy.policies"):
        if ep.name == kind:
            return ep.load()
    if kind == "locomotion":
        return LocomotionPolicy
    if kind == "hold_position":
        return HoldPositionPolicy
    if kind == "interpolation":
        return InterpolationPolicy
    if kind == "wbt":
        return WholeBodyTrackingPolicy
    if kind == "sonic":
        return SonicPolicy
    if kind == "ufo":
        return UfoPolicy
    if kind == "passive_locomotion":
        return PassiveLocomotionPolicy
    if kind == "waist_locomotion":
        return WaistLocomotionPolicy
    raise ValueError(f"Unknown policy kind: {kind}")


class PolicyState(StrEnum):
    STARTUP_LATCHED = "startup_latched"
    LATCHED = "latched"
    IDLE = "idle"
    SWITCHING = "switching"
    RUNNING = "running"
    FALLBACK = "fallback"


class PolicyStateMachine:
    """Run one full-body policy or an upper/lower pair through one interface."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        resolved: tuple[ResolvedPolicy, ...],
        *,
        instances: dict[str, BasePolicy] | None = None,
        inbox: CommandInbox | None = None,
        transport: MqttTransport | None = None,
        interface_manager: InterfaceManager | None = None,
        clock=time.monotonic,
    ):
        self.runtime = runtime
        self.resolved = resolved
        self._clock = clock
        self._started_at = clock()
        self._specs = {item.spec.name: item.spec for item in resolved}
        self.inbox = inbox or CommandInbox(self._specs, clock=clock)
        self.transport = transport or MqttTransport(runtime.mqtt, tuple(item.spec for item in resolved), self.inbox)
        self.policies: dict[str, BasePolicy] = {}
        self._policy_executor = None
        try:
            self.interface_manager = interface_manager or InterfaceManager.get()
            self.policies = instances if instances is not None else self._build_policies()
            if set(self.policies) != set(self._specs):
                raise ValueError("Policy instances must exactly match configured policy names")
            owner = next(iter(self.policies.values()))
            self.dof_names = tuple(owner.dof_names)
            self._policy_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="vex-policy",
            )

            rate = resolved[0].config.task.rl_rate
            self.rate = RateLimiter(rate)
            self._state_period = 1.0 / runtime.mqtt.state_frequency_hz
            self._next_state_publish = self._started_at
            self.state: PolicyState = PolicyState.STARTUP_LATCHED
            self.active_policy: tuple[str, ...] = ()
            self.requested_policy: tuple[str, ...] = ()
            self.reason: str | None = "startup"
            self._fallback_inputs: dict[str, float] | None = None
            self.last_command_seq: int | None = None
            self._last_status: tuple[Any, ...] | None = None
        except BaseException:
            if self._policy_executor is not None:
                self._policy_executor.shutdown(wait=True)
            self._close_policies(self.policies.values())
            try:
                self.transport.close()
            except Exception:
                logger.exception("Failed to close transport after runtime initialization failure")
            raise

    def _build_policies(self) -> dict[str, BasePolicy]:
        instances: dict[str, BasePolicy] = {}
        try:
            for item in self.resolved:
                cls = _policy_class(item.kind)
                policy = cls(item.config)
                instances[item.spec.name] = policy
                logger.info(f"Preloaded policy {item.spec.name}: {cls.__name__}")
        except BaseException:
            self._close_policies(instances.values())
            raise
        return instances

    @staticmethod
    def _close_policies(policies) -> None:
        for policy in policies:
            try:
                policy.close()
            except Exception:
                logger.exception("Failed to close policy")

    def _status_payload(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
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
        names = self.active_policy
        self.active_policy = ()
        self._fallback_inputs = None
        self._stop_policies(names)

    def _stop_policies(self, names) -> None:
        errors = []
        for name in names:
            try:
                self.policies[name].deactivate()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Policy deactivation failed", errors)

    def _latch(self, reason: str) -> None:
        try:
            self._deactivate()
        except Exception as error:
            logger.exception("Failed to deactivate policies while latching")
            reason = f"{reason}; deactivate_failed:{error}"
        self.state = PolicyState.LATCHED
        self.reason = reason

    def _selection_key(self, name: str) -> tuple[int, str]:
        order = {"full_body": 0, "lower_body": 0, "upper_body": 1}
        return order[self._specs[name].type], name

    def _canonical_selection(self, names) -> tuple[str, ...]:
        return tuple(sorted(names, key=self._selection_key))

    def _activate(self, names: tuple[str, ...], robot_state: LowState) -> None:
        names = self._canonical_selection(names)
        self.state = PolicyState.SWITCHING
        self.reason = None
        self._publish_status()

        previous = set(self.active_policy)
        desired = set(names)
        live = dict.fromkeys(self.active_policy)
        try:
            for name in self.active_policy:
                if name not in desired:
                    self.policies[name].deactivate()
                    live.pop(name)

            for name in names:
                if name in previous:
                    continue
                live[name] = None
                reason = self.policies[name].activate(robot_state)
                if reason is not None:
                    self.active_policy = ()
                    self.state = PolicyState.LATCHED
                    self.reason = reason
                    self._stop_policies(live)
                    self._publish_status()
                    return
        except BaseException as error:
            self.active_policy = ()
            self.state = PolicyState.LATCHED
            self.reason = f"policy_fault:{'+'.join(names)}:{error}"
            try:
                self._stop_policies(live)
            except Exception:
                logger.exception("Policy activation rollback failed")
            if isinstance(error, PolicyRuntimeFault):
                self._publish_status()
                return
            raise

        self.active_policy = names
        self.state = PolicyState.RUNNING

    def _maybe_publish_state(self, robot_state, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        if current + 1e-9 < self._next_state_publish:
            return
        timestamp = time.time()
        self._next_state_publish = current + self._state_period
        try:
            payload = encode_robot_state(
                robot_state,
                self.dof_names,
                started_at=self._started_at,
                monotonic_now=current,
                timestamp=timestamp,
            )
        except (TypeError, ValueError, IndexError) as error:
            logger.warning(f"Skipping invalid state telemetry: {error}")
            return
        self.transport.publish_state(payload)
        if len(self.active_policy) == 1:
            policy = self.policies[self.active_policy[0]]
            get_reference_state = getattr(policy, "get_reference_state", None)
            reference_state = get_reference_state() if get_reference_state is not None else None
            if reference_state is not None:
                try:
                    reference_payload = encode_robot_state(
                        reference_state,
                        self.dof_names,
                        started_at=self._started_at,
                        monotonic_now=current,
                        timestamp=timestamp,
                    )
                except (TypeError, ValueError, IndexError) as error:
                    logger.warning(f"Skipping invalid reference telemetry: {error}")
                else:
                    self.transport.publish_reference_state(reference_payload)

    def _publish_idle_state(self, robot_state: LowState | None, now: float) -> None:
        if robot_state is not None:
            self._maybe_publish_state(robot_state, now)

    def _estop_violations(self, names: tuple[str, ...], robot_state: LowState):
        violations = []
        for name in names:
            checker = self.policies[name].estop
            if checker is not None:
                found = checker.check(robot_state)
                if found:
                    violations.append((name, found))
        return violations

    def _handle_estops(self, names: tuple[str, ...], robot_state: LowState) -> bool:
        """Handle all violations before any policy in this group runs."""
        violations = self._estop_violations(names, robot_state)
        if not violations:
            return False
        detail = "; ".join(f"{name}: {', '.join(map(str, found))}" for name, found in violations)
        reason = f"estop:{detail}"
        if self._fallback_inputs is not None:
            self._latch(f"{self.reason}; fallback_{reason}")
            return True
        if any(item.invalid for _, found in violations for item in found):
            self._latch(reason)
            return True

        responses = []
        for name, _ in violations:
            fallback = self._specs[name].estop.fallback
            if fallback is None:
                self._latch(reason)
                return True
            target = self._specs[fallback.policy]
            inputs = {p.name: p.default for p in target.input_parameters}
            inputs.update(fallback.inputs)
            responses.append((target.name, inputs))
        if any(response != responses[0] for response in responses[1:]):
            self._latch(f"{reason}; fallback_conflict")
            return True

        target_name, inputs = responses[0]
        try:
            self._deactivate()
            target_violations = self._estop_violations((target_name,), robot_state)
            if target_violations:
                target_detail = ", ".join(str(item) for _, found in target_violations for item in found)
                self._latch(f"{reason}; fallback_rejected:{target_name}: {target_detail}")
                return True
            self._activate((target_name,), robot_state)
            if self.state != PolicyState.RUNNING:
                self.reason = f"{reason}; fallback_rejected:{target_name}: {self.reason}"
                return True
        except Exception as error:
            self._latch(f"{reason}; fallback_failed:{target_name}: {error}")
            return True
        self._fallback_inputs = inputs
        self.state = PolicyState.FALLBACK
        self.reason = reason
        return True

    def tick(self, now: float | None = None) -> None:
        """Run one deterministic state-machine/control iteration."""
        current = self._clock() if now is None else now
        robot_state = self.interface_manager.get_low_state()
        received = self.inbox.snapshot()
        if received is None:
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return

        packet = received.packet
        control = packet.control
        self.last_command_seq = packet.seq
        self.requested_policy = self._canonical_selection(control.policy)

        if current - received.received_at > self.runtime.mqtt.command_timeout_s:
            self._latch("command_timeout")
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return
        if control.estop:
            self._latch("estop")
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return

        if self.state in {PolicyState.STARTUP_LATCHED, PolicyState.LATCHED}:
            if not control.policy:
                self.state = PolicyState.IDLE
                self.reason = None
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return

        if not control.policy:
            self._deactivate()
            self.state = PolicyState.IDLE
            self.reason = None
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return

        in_fallback = self._fallback_inputs is not None
        desired = self.active_policy if in_fallback else self._canonical_selection(control.policy)
        if robot_state is None:
            self._latch("low_state_unavailable")
            self._publish_status()
            return
        if self._handle_estops(desired, robot_state):
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return
        if desired != self.active_policy:
            self._activate(desired, robot_state)
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return  # deliberate one-cycle low-command gap during a switch

        try:
            for name in desired:
                policy = self.policies[name]
                inputs = self._fallback_inputs if in_fallback else control.inputs[name]
                policy.apply_control(inputs)
            self._step_active_policies(robot_state)
        except Exception as error:
            if not in_fallback and not isinstance(error, PolicyRuntimeFault):
                raise
            reason = f"policy_fault:{'+'.join(desired)}:{error}"
            self._latch(f"{self.reason}; {reason}" if in_fallback else reason)
            self._publish_idle_state(robot_state, current)
            self._publish_status()
            return
        if not in_fallback:
            self.state = PolicyState.RUNNING
            self.reason = None
        self._publish_status()

    def run(self) -> None:
        try:
            self.transport.start()
            self._publish_status(force=True)
            while True:
                self.tick()
                self.rate.sleep()
        except KeyboardInterrupt:
            logger.info("Policy runtime interrupted")
        finally:
            try:
                self._policy_executor.shutdown(wait=True)
            finally:
                try:
                    self._deactivate()
                finally:
                    try:
                        self._close_policies(self.policies.values())
                    finally:
                        self.transport.close()

    def _step_active_policies(self, robot_state: LowState) -> None:
        """Infer all active policies from one state snapshot and publish one merged command."""
        if len(self.active_policy) == 1:
            commands = [self.policies[self.active_policy[0]].step(robot_state)]
        else:
            futures = {
                name: self._policy_executor.submit(self.policies[name].step, robot_state) for name in self.active_policy
            }
            # A sibling must not still be running when fault handling tears down
            # its episode resources. No partial command is published on failure.
            wait(futures.values())
            commands = [futures[name].result() for name in self.active_policy]

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
        self.interface_manager.send_low_command(
            q,
            dq,
            tau,
            robot_state.joint_pos[0],
            kp_override=kp,
            kd_override=kd,
        )
        self._maybe_publish_state(robot_state)


__all__ = ["PolicyState", "PolicyStateMachine"]
