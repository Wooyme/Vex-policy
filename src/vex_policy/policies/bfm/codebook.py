"""Holosoma BFM codebook controllers: [token, codeword] exports with learned codewords in the graph."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from vex_policy.config.config_types import BfmTaskConfig
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.bfm.base import BfmPolicy
from vex_policy.policies.bfm.contract import LATENT_DIM, categorical_choices, project_latent


def _token(action: np.ndarray, choices: int) -> int:
    """The integral token index a codebook export emits first."""
    value = float(action[0, 0])
    token = round(value)
    if abs(value - token) > 1e-3 or not 0 <= token < choices:
        raise PolicyRuntimeFault(f"BFM token controller returned invalid token {value}")
    return token


class BfmCodebookPolicy(BfmPolicy):
    """A controller whose export returns ``action[1, 257]``: a selected token, then its codeword.

    The token is chosen by in-graph top-k Gumbel sampling and the codeword is noiseless. Codewords
    were trained away from the initial vocabulary and are baked into the graph, so no vocabulary is
    read; the executed latent is the projected codeword, as BFMCodebookAction projects the sampled latent.
    """

    task_type: ClassVar[type] = BfmTaskConfig
    repeat_first_actor_frame: ClassVar[bool] = True
    controller_output_dim: ClassVar[int] = 1 + LATENT_DIM
    codebook_action = "holosoma.managers.action.terms.bfm:BFMCodebookAction"

    def _setup_task(self) -> None:
        self.num_choices = categorical_choices(self.task.model_path)

    def _validate_experiment(self, experiment: dict) -> None:
        terms = experiment["action"]["terms"]
        if len(terms) != 1 or next(iter(terms.values()))["func"] != self.codebook_action:
            raise ValueError("BFM codebook controller must be trained with one BFMCodebookAction term")
        if experiment["algo"]["config"]["module_dict"]["actor"].get("distribution") != "codebook":
            raise ValueError("BFM codebook controller must use a codebook actor")

    def _reset_episode(self) -> None:
        super()._reset_episode()
        self.token = -1

    def _decode_latent(self, action: np.ndarray) -> np.ndarray:
        token = _token(action, self.num_choices)
        codeword = action[:, 1:]
        # Finiteness is checked by _infer; a zero codeword has no direction to project.
        if not np.linalg.norm(codeword) > 1e-6:
            raise PolicyRuntimeFault("BFM codebook controller returned an invalid codeword")
        self.token = token
        return project_latent(codeword)
