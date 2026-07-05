from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class MasterState:
    """Normalized state exposed by a master input device.

    `motion` is a relative 6D command, not an absolute pose:
    [x, y, z, roll, pitch, yaw].
    `buttons` is a generic indexed button vector used by controller-level
    commands such as gripper actions and joint reset.
    """

    motion: np.ndarray
    buttons: list[int] = field(default_factory=list)
    connected: bool = True
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def motion6(self) -> np.ndarray:
        motion = np.asarray(self.motion, dtype=np.float64)
        if motion.size < 6:
            padded = np.zeros(6, dtype=np.float64)
            padded[: motion.size] = motion
            return padded
        return motion[:6].copy()


class MasterDevice(Protocol):
    """Interface implemented by master devices such as SpaceMouse or a master arm."""

    def start(self) -> None:
        ...

    def close(self) -> None:
        ...

    def get_state(self) -> MasterState:
        ...
