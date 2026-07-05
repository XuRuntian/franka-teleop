from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ButtonHoldTrigger:
    """Generic multi-button long-press trigger for master devices."""

    button_indices: tuple[int, ...]
    hold_s: float
    cooldown_s: float = 0.0
    _pressed_since: float | None = None
    _last_triggered_at: float | None = None
    _armed: bool = True

    def pressed(self, buttons: list[int]) -> bool:
        if not self.button_indices:
            return False
        return all(self._button_pressed(buttons, index) for index in self.button_indices)

    def update(self, buttons: list[int], timestamp: float) -> bool:
        if not self.pressed(buttons):
            self._pressed_since = None
            self._armed = True
            return False

        if self._pressed_since is None:
            self._pressed_since = timestamp
            return False

        if self._last_triggered_at is not None:
            if timestamp - self._last_triggered_at < self.cooldown_s:
                return False

        if self._armed and timestamp - self._pressed_since >= self.hold_s:
            self._armed = False
            self._last_triggered_at = timestamp
            return True

        return False

    @staticmethod
    def _button_pressed(buttons: list[int], index: int) -> bool:
        try:
            return bool(buttons[index])
        except IndexError:
            return False
