"""Holosoma bfm_wrist_student: a goal-free codebook student following the operator's pull on a wrist."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from vex_policy.policies.bfm.codebook import BfmCodebookPolicy


class BfmWristStudentPolicy(BfmCodebookPolicy):
    """Six controller-step frames, the gait clock and the previous latent (6x93 + 4 + 256 = 818).

    The wrist height/velocity goals were teacher-only; the student feels them through the arm,
    so it takes no inputs. The gait clock always runs.
    """

    frame_group: ClassVar[str] = "student_obs"
    command_group: ClassVar[str] = "phase_obs"
    frame_history: ClassVar[int] = 6
    input_names: ClassVar[frozenset[str]] = frozenset()
    command_obs: ClassVar[dict[str, tuple[int, str]]] = {
        "cos_phase": (2, "locomotion:cos_phase"),
        "sin_phase": (2, "locomotion:sin_phase"),
    }

    def _command_observation(self) -> dict[str, np.ndarray]:
        phase = self._clock_phase()
        return {"cos_phase": np.cos(phase), "sin_phase": np.sin(phase)}
