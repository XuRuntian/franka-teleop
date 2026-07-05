from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .client import FrankaHttpClient, RobotState
from .config import TeleopConfig
from .master import MasterDevice, MasterState
from .spacemouse import SpaceMouseDevice
from .transforms import (
    apply_cartesian_delta,
    clip_rotation_delta,
    clip_translation_delta,
    ee_pose_to_tcp_pose,
    tcp_pose_to_ee_pose,
    transform_from_xyz_quat,
)
from .triggers import ButtonHoldTrigger


@dataclass
class TeleopCommand:
    master_state: MasterState
    raw_motion: np.ndarray
    buttons: list[int]
    translation_delta: np.ndarray
    rotation_delta: np.ndarray
    reference_frame: str
    current_ee_pose: np.ndarray
    current_tcp_pose: np.ndarray
    target_tcp_pose: np.ndarray
    target_ee_pose: np.ndarray
    active: bool


class CartesianTeleopController:
    """Standalone master-device to Cartesian impedance teleoperation loop."""

    def __init__(
        self,
        config: TeleopConfig,
        client: FrankaHttpClient | None = None,
        gripper_client: FrankaHttpClient | None = None,
        master: MasterDevice | None = None,
    ):
        config.validate()
        self.config = config
        self.client = client or FrankaHttpClient(config.server_url, config.request_timeout_s)
        if gripper_client is not None:
            self.gripper_client = gripper_client
        elif client is None:
            self.gripper_client = FrankaHttpClient(config.server_url, config.request_timeout_s)
        else:
            self.gripper_client = self.client
        self.master = master or SpaceMouseDevice()
        self.ee_t_tcp = transform_from_xyz_quat(config.tool.ee_tcp_xyz, config.tool.ee_tcp_quat)
        self._last_gripper_time = 0.0
        self._previous_buttons: list[int] = []
        self._gripper_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="franka-gripper")
        self._gripper_future: Future | None = None
        self._last_motion_active = False
        self._reset_trigger = (
            ButtonHoldTrigger(
                button_indices=config.reset.button_indices,
                hold_s=config.reset.hold_s,
                cooldown_s=config.reset.cooldown_s,
            )
            if config.reset.enabled
            else None
        )

    def run_forever(self) -> None:
        self.start()
        try:
            self._loop()
        finally:
            self.stop()

    def start(self) -> None:
        if self.config.clear_errors:
            self.client.clear_errors()
        if self.config.apply_load_on_start:
            self.client.set_load(self.config.tool.load.as_server_payload())
        self.master.start()

    def stop(self) -> None:
        self.master.close()
        self._gripper_executor.shutdown(wait=False, cancel_futures=True)

    def request_joint_reset(self) -> None:
        """Run the configured joint reset through the robot client."""
        self._last_motion_active = False
        print("Joint reset requested; teleop loop will pause until reset completes.")
        self.client.joint_reset(timeout_s=self.config.reset.timeout_s)

    def _loop(self) -> None:
        period = 1.0 / self.config.motion.rate_hz
        print(
            "Standalone SpaceMouse teleop running. "
            "Press Ctrl+C to stop."
        )
        while True:
            start_time = time.time()
            self.step(start_time)
            elapsed = time.time() - start_time
            time.sleep(max(0.0, period - elapsed))

    def step(self, timestamp: float | None = None) -> None:
        timestamp = time.time() if timestamp is None else timestamp
        state = self.client.get_state()
        master_state = self.master.get_state()
        command = self.compute_command(state, master_state)

        try:
            if self._maybe_trigger_reset(command.buttons, timestamp):
                return

            if self._reset_buttons_pressed(command.buttons):
                if self._last_motion_active:
                    self._send_idle_stop(command)
                self._last_motion_active = False
                return

            self._handle_gripper(command.buttons, timestamp)

            if command.active:
                self.client.send_pose(command.target_ee_pose)
                self._last_motion_active = True
                return

            if self._last_motion_active:
                self._send_idle_stop(command)
            self._last_motion_active = False
        finally:
            self._previous_buttons = list(command.buttons)

    def compute_command(self, robot_state: RobotState, master_state: MasterState) -> TeleopCommand:
        """Convert a master input state into a Cartesian command.

        This is the public interface for inspecting or reusing the master command
        without sending it to the robot.
        """
        raw_motion = master_state.motion6()
        current_ee_pose = np.asarray(robot_state.ee_pose, dtype=np.float64).copy()
        current_tcp_pose = ee_pose_to_tcp_pose(current_ee_pose, self.ee_t_tcp)
        translation_delta = raw_motion[:3] * self.config.motion.translation_scale
        rotation_delta = raw_motion[3:6] * self.config.motion.rotation_scale
        translation_delta = clip_translation_delta(
            translation_delta,
            self.config.motion.max_translation_step,
        )
        rotation_delta = clip_rotation_delta(
            rotation_delta,
            self.config.motion.max_rotation_step,
        )

        active = np.linalg.norm(raw_motion) > self.config.motion.deadband
        if active:
            target_tcp_pose = apply_cartesian_delta(
                current_tcp_pose,
                translation_delta,
                rotation_delta,
                self.config.motion.reference_frame,
            )
            target_tcp_pose[:3] = np.clip(
                target_tcp_pose[:3],
                self.config.workspace.low,
                self.config.workspace.high,
            )
        else:
            target_tcp_pose = current_tcp_pose.copy()

        target_ee_pose = tcp_pose_to_ee_pose(target_tcp_pose, self.ee_t_tcp)
        return TeleopCommand(
            master_state=master_state,
            raw_motion=raw_motion,
            buttons=list(master_state.buttons),
            translation_delta=translation_delta,
            rotation_delta=rotation_delta,
            reference_frame=self.config.motion.reference_frame,
            current_ee_pose=current_ee_pose,
            current_tcp_pose=current_tcp_pose,
            target_tcp_pose=target_tcp_pose,
            target_ee_pose=target_ee_pose,
            active=active,
        )

    def _handle_gripper(self, buttons: list[int], timestamp: float) -> None:
        if not self.config.gripper.enabled or not buttons:
            return
        if self._gripper_command_busy():
            return
        if timestamp - self._last_gripper_time < self.config.gripper.cooldown_s:
            return

        close_pressed = self._button_pressed(buttons, self.config.gripper.close_button)
        open_pressed = self._button_pressed(buttons, self.config.gripper.open_button)
        close_was_pressed = self._button_pressed(self._previous_buttons, self.config.gripper.close_button)
        open_was_pressed = self._button_pressed(self._previous_buttons, self.config.gripper.open_button)
        if close_pressed:
            if close_was_pressed:
                return
            self._submit_gripper_command("close", self.gripper_client.close_gripper)
            self._last_gripper_time = timestamp
        elif open_pressed:
            if open_was_pressed:
                return
            self._submit_gripper_command("open", self.gripper_client.open_gripper)
            self._last_gripper_time = timestamp

    @staticmethod
    def _button_pressed(buttons: list[int], index: int) -> bool:
        try:
            return bool(buttons[index])
        except IndexError:
            return False

    def _maybe_trigger_reset(self, buttons: list[int], timestamp: float) -> bool:
        if self._reset_trigger is None:
            return False
        if not self._reset_trigger.update(buttons, timestamp):
            return False
        self.request_joint_reset()
        return True

    def _reset_buttons_pressed(self, buttons: list[int]) -> bool:
        if self._reset_trigger is None:
            return False
        return self._reset_trigger.pressed(buttons)

    def _send_idle_stop(self, command: TeleopCommand) -> None:
        if not self.config.motion.stop_on_idle:
            return
        self.client.send_pose(command.current_ee_pose)

    def _gripper_command_busy(self) -> bool:
        if self._gripper_future is None:
            return False
        if not self._gripper_future.done():
            return True
        self._gripper_future = None
        return False

    def _submit_gripper_command(self, name: str, command: Callable[[], None]) -> None:
        self._gripper_future = self._gripper_executor.submit(self._run_gripper_command, name, command)

    @staticmethod
    def _run_gripper_command(name: str, command: Callable[[], None]) -> None:
        try:
            command()
        except Exception as exc:
            print(f"Gripper {name} command failed: {exc}")
