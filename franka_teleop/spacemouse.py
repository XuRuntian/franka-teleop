from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

_pyspacemouse = None


def _driver():
    global _pyspacemouse
    if _pyspacemouse is None:
        from . import pyspacemouse as loaded_driver

        _pyspacemouse = loaded_driver
    return _pyspacemouse


@dataclass
class SpaceMouseState:
    motion: np.ndarray
    buttons: list[int]
    connected: bool


class SpaceMouseDevice:
    """Background reader for 3Dconnexion SpaceMouse devices."""

    def __init__(self, poll_hz: float = 100.0):
        self.poll_hz = poll_hz
        self._lock = threading.Lock()
        self._state = SpaceMouseState(np.zeros(6, dtype=np.float64), [0, 0], False)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        _driver().open()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if _pyspacemouse is not None:
            _pyspacemouse.close()

    def get_state(self) -> SpaceMouseState:
        with self._lock:
            return SpaceMouseState(
                motion=self._state.motion.copy(),
                buttons=list(self._state.buttons),
                connected=self._state.connected,
            )

    def _read_loop(self) -> None:
        period = 1.0 / self.poll_hz
        driver = _driver()
        while not self._stop_event.is_set():
            raw_state = driver.read_all()
            motion, buttons, connected = self._convert_state(raw_state)
            with self._lock:
                self._state = SpaceMouseState(motion, buttons, connected)
            time.sleep(period)

    @staticmethod
    def _convert_state(raw_state) -> tuple[np.ndarray, list[int], bool]:
        if not raw_state:
            return np.zeros(6, dtype=np.float64), [0, 0], False

        states = [state for state in raw_state if state is not None]
        if not states:
            return np.zeros(6, dtype=np.float64), [0, 0], False

        first = states[0]
        motion = np.array(
            [
                -first.y,
                first.x,
                first.z,
                -first.roll,
                -first.pitch,
                -first.yaw,
            ],
            dtype=np.float64,
        )
        buttons = list(first.buttons)

        if len(states) > 1:
            second = states[1]
            motion = np.concatenate(
                [
                    motion,
                    np.array(
                        [
                            -second.y,
                            second.x,
                            second.z,
                            -second.roll,
                            -second.pitch,
                            -second.yaw,
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
            buttons.extend(list(second.buttons))

        return motion, buttons, True
