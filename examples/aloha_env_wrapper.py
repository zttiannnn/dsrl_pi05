"""Utility wrappers for running Aloha real-world environments inside the DSRL pipeline.

This module provides a thin adapter around the OpenPI05 Aloha environment so that it
exposes the minimal interface expected by the existing training utilities
(reset, get_observation, step).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from openpi.examples.aloha_real import env as aloha_env


class AlohaRobotEnv:
    """Wraps the OpenPI Aloha real robot environment with a simple API."""

    def __init__(
        self,
        *,
        render_size: int = 224,
        reset_position: Optional[list[float]] = None,
    ) -> None:
        self._env = aloha_env.AlohaRealEnvironment(
            reset_position=reset_position,
            render_height=render_size,
            render_width=render_size,
        )

    def reset(self) -> None:
        """Resets both arms to the nominal pose."""
        self._env.reset()

    def get_observation(self) -> dict:
        """Returns the latest observation dict from the inner environment."""
        return self._env.get_observation()

    def step(self, action: np.ndarray) -> None:
        """Executes a single control command on the hardware."""
        self._env.apply_action({"actions": action})
