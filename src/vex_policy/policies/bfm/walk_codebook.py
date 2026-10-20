"""Holosoma bfm_walk_codebook: a velocity-commanded walk selecting learned codewords."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from vex_policy.policies.bfm.codebook import BfmCodebookPolicy


class BfmWalkCodebookPolicy(BfmCodebookPolicy):
    """The 356-D walk observation (frame, velocity command, gait phase, latent) selects a codeword."""

    input_names: ClassVar[frozenset[str]] = frozenset({"vx", "vy", "yaw"})
    command_obs: ClassVar[dict[str, tuple[int, str]]] = {
        "command": (3, "bfm:velocity_command"),
        "cos_phase": (2, "locomotion:cos_phase"),
        "sin_phase": (2, "locomotion:sin_phase"),
    }

    def _phase(self) -> np.ndarray:
        """The evaluation gait clock, pinned to pi while standing; the episode clock keeps advancing."""
        velocity = np.asarray([self._control["vy"], -self._control["vx"]])
        if np.linalg.norm(velocity) < 0.01 and abs(self._control["yaw"]) < 0.01:
            return np.full((1, 2), np.pi)
        return self._clock_phase()

    def _command_observation(self) -> dict[str, np.ndarray]:
        phase = self._phase()
        return {
            "command": np.asarray([[self._control["vy"], -self._control["vx"], -self._control["yaw"]]]),
            "cos_phase": np.cos(phase),
            "sin_phase": np.sin(phase),
        }
