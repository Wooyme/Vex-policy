"""Opt-in, term-major observation history used by the exported PPO policies."""

from __future__ import annotations

from collections import deque

import numpy as np

from vex_policy.config.config_types.observation import ObservationConfig
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse


class ObservationHistory:
    def __init__(self, config: ObservationConfig):
        self.obs_scales = config.obs_scales
        self.obs_dims = config.obs_dims
        self.obs_dict = config.obs_dict
        self.history_length_dict = config.history_length_dict
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self._initialize_history_state()

    def reset(self) -> None:
        for terms in self.obs_history_buffers.values():
            for history in terms.values():
                history.clear()
        for buffer in self.obs_buf_dict.values():
            buffer.fill(0.0)

    def prepare(self, terms: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self._update_obs_history(self.parse_current_obs_dict(terms))

    def _initialize_history_state(self):
        """Create per-term history deques and zero-initialized flattened buffers."""
        self.obs_history_buffers: dict[str, dict[str, deque[np.ndarray]]] = {}
        self.obs_terms_sorted: dict[str, list[str]] = {}
        self.obs_buf_dict: dict[str, np.ndarray] = {}

        for group, term_names in self.obs_dict.items():
            self.obs_terms_sorted[group] = sorted(term_names)
            history_len = self.history_length_dict.get(group, 1)
            self.obs_history_buffers[group] = {}
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                term_dim = self.obs_dims[term]
                self.obs_history_buffers[group][term] = deque(maxlen=history_len)
                flattened_terms.append(np.zeros((1, term_dim * history_len), dtype=np.float32))

            self.obs_buf_dict[group] = np.concatenate(flattened_terms, axis=1) if flattened_terms else np.zeros((1, 0))

    def _calculate_obs_dim_dict(self):
        """Calculate observation dimensions for each observation type."""
        obs_dim_dict = {}
        for key in self.obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in self.obs_dict[key]:
                obs_dim_dict[key] += self.obs_dims[obs_name]
        return obs_dim_dict

    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary with per-term scaling."""
        current_obs_dict: dict[str, dict[str, np.ndarray]] = {}
        for group, term_names in self.obs_terms_sorted.items():
            grouped_terms: dict[str, np.ndarray] = {}
            for term in term_names:
                if term not in current_obs_buffer_dict:
                    raise KeyError(f"Observation term '{term}' missing from current observation buffer.")
                term_obs = current_obs_buffer_dict[term]
                if term_obs.ndim == 1:
                    term_obs = term_obs.reshape(1, -1)
                scale = self.obs_scales[term]
                grouped_terms[term] = (term_obs * scale).astype(np.float32, copy=False)
            current_obs_dict[group] = grouped_terms
        return current_obs_dict

    def _update_obs_history(self, current_obs_dict: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Update observation history buffers and return flattened observations per group."""
        group_outputs: dict[str, np.ndarray] = {}

        for group, term_dict in current_obs_dict.items():
            history_len = self.history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)

                buffer = self.obs_history_buffers[group][term]
                buffer.append(obs.copy())

                history = list(buffer)
                if len(history) < history_len:
                    missing = history_len - len(history)
                    history = [np.zeros_like(obs)] * missing + history

                # Match training order: time dimension first, then flatten into [history_len * term_dim].
                stacked = np.stack(history[-history_len:], axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))

            group_outputs[group] = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )

        self.obs_buf_dict = {group: value.copy() for group, value in group_outputs.items()}
        return group_outputs

    def print_observations(self, obs: dict[str, np.ndarray], dof_names, scaled_action) -> None:
        """Print observation vector with term naming for debugging.

        Args:
            obs: Dictionary mapping observation group names to their flattened arrays.
        """
        np.set_printoptions(suppress=True, precision=3)
        print("\n========== Observation Vector ==========")
        for group_name, group_obs in obs.items():
            print(f"\n{group_name}:")
            if group_name in self.obs_dict:
                start_idx = 0
                for term_name in self.obs_terms_sorted.get(group_name, []):
                    term_dim = self.obs_dims[term_name]
                    history_len = self.history_length_dict.get(group_name, 1)
                    total_dim = term_dim * history_len
                    term_values = group_obs[0, start_idx : start_idx + total_dim]
                    print(f"  {term_name:20s} (dim={term_dim:2d}, hist={history_len}): {term_values}")
                    start_idx += total_dim

        # Joint table: dof_name | q (deg) | dq | action
        self._print_joint_table(obs, dof_names, scaled_action)
        print("========================================\n")

    def _print_joint_table(self, obs: dict[str, np.ndarray], dof_names, scaled_action) -> None:
        """Print a compact per-joint table: name | q(°) | dq(°/s) | act(°)."""
        # Walk obs_terms_sorted + obs_dims to locate dof_pos / dof_vel slices
        q = dq = None
        for grp, buf in obs.items():
            col = 0
            for term in self.obs_terms_sorted.get(grp, []):
                dim = self.obs_dims[term] * self.history_length_dict.get(grp, 1)
                if q is None and term == "dof_pos":
                    q = buf[0, col : col + dim] / self.obs_scales.get("dof_pos", 1.0)
                if dq is None and term == "dof_vel":
                    dq = buf[0, col : col + dim] / self.obs_scales.get("dof_vel", 1.0)
                col += dim
        act = scaled_action[0] if scaled_action is not None else None
        d = np.degrees
        w = max(len(n) for n in dof_names)
        print(f"\n  {'joint':<{w}}  {'q(°)':>7}  {'dq(°/s)':>8}  {'act(°)':>7}")
        print(f"  {'─' * (w + 29)}")
        for i, name in enumerate(dof_names):
            qi = f"{d(q[i]):7.1f}" if q is not None and i < len(q) else "    n/a"
            di = f"{d(dq[i]):8.1f}" if dq is not None and i < len(dq) else "     n/a"
            ai = f"{d(act[i]):7.1f}" if act is not None and i < len(act) else "    n/a"
            print(f"  {name:<{w}}  {qi}  {di}  {ai}")


def robot_observation_terms(robot_state_data: LowState, default_dof_angles, debug):
    """Extract current observation data from robot state."""
    current_obs_buffer_dict = {}

    # Extract base and joint data
    current_obs_buffer_dict["base_quat"] = robot_state_data.base_quat
    if debug.force_zero_angular_velocity:
        current_obs_buffer_dict["base_ang_vel"] = np.zeros((1, 3))
    else:
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data.base_ang_vel
    current_obs_buffer_dict["dof_pos"] = robot_state_data.joint_pos - default_dof_angles
    current_obs_buffer_dict["dof_vel"] = robot_state_data.joint_vel

    if debug.force_upright_imu:
        current_obs_buffer_dict["projected_gravity"] = np.array([[0.0, 0.0, -1.0]])
    else:
        v = np.array([[0, 0, -1]])
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(current_obs_buffer_dict["base_quat"], v)

    return current_obs_buffer_dict
