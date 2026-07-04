from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import requests


@dataclass
class RobotState:
    ee_pose: np.ndarray
    ee_vel: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    jacobian: np.ndarray
    gripper_pos: float | None


class FrankaHttpClient:
    """HTTP client for franka_robot_server/franka_server.py."""

    def __init__(self, server_url: str = "http://127.0.0.2:5000/", timeout_s: float = 2.0):
        self.server_url = server_url.rstrip("/") + "/"
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def _post(self, route: str, json: dict[str, Any] | None = None) -> requests.Response:
        response = self.session.post(
            self.server_url + route.lstrip("/"),
            json=json,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response

    def clear_errors(self) -> None:
        self._post("clearerr")

    def send_pose(self, ee_pose: np.ndarray) -> None:
        ee_pose = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose.shape != (7,):
            raise ValueError(f"ee_pose must have shape (7,), got {ee_pose.shape}")
        self._post("pose", json={"arr": ee_pose.tolist()})

    def get_pose(self) -> np.ndarray:
        return np.asarray(self._post("getpos").json()["pose"], dtype=np.float64)

    def get_q(self) -> np.ndarray:
        return np.asarray(self._post("getq").json()["q"], dtype=np.float64)

    def get_dq(self) -> np.ndarray:
        return np.asarray(self._post("getdq").json()["dq"], dtype=np.float64)

    def get_state(self) -> RobotState:
        try:
            payload = self._post("getstate").json()
        except requests.RequestException:
            payload = self._get_state_from_individual_routes()

        return RobotState(
            ee_pose=np.asarray(payload.get("pose", np.zeros(7)), dtype=np.float64),
            ee_vel=np.asarray(payload.get("vel", np.zeros(6)), dtype=np.float64),
            q=np.asarray(payload.get("q", np.zeros(7)), dtype=np.float64),
            dq=np.asarray(payload.get("dq", np.zeros(7)), dtype=np.float64),
            force=np.asarray(payload.get("force", np.zeros(3)), dtype=np.float64),
            torque=np.asarray(payload.get("torque", np.zeros(3)), dtype=np.float64),
            jacobian=np.asarray(payload.get("jacobian", np.zeros((6, 7))), dtype=np.float64).reshape(6, 7),
            gripper_pos=payload.get("gripper_pos"),
        )

    def _get_state_from_individual_routes(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for route, key, default in [
            ("getpos", "pose", np.zeros(7).tolist()),
            ("getvel", "vel", np.zeros(6).tolist()),
            ("getq", "q", np.zeros(7).tolist()),
            ("getdq", "dq", np.zeros(7).tolist()),
            ("getforce", "force", np.zeros(3).tolist()),
            ("gettorque", "torque", np.zeros(3).tolist()),
            ("getjacobian", "jacobian", np.zeros((6, 7)).tolist()),
        ]:
            try:
                payload[key] = self._post(route).json()[key]
            except requests.RequestException:
                payload[key] = default
        try:
            payload["gripper_pos"] = self._post("get_gripper").json()["gripper"]
        except requests.RequestException:
            payload["gripper_pos"] = None
        return payload

    def open_gripper(self) -> None:
        self._post("open_gripper")

    def close_gripper(self) -> None:
        self._post("close_gripper")

    def move_gripper(self, position: int) -> None:
        self._post("move_gripper", json={"gripper_pos": int(position)})

    def joint_reset(self, timeout_s: float = 30.0) -> None:
        old_timeout = self.timeout_s
        try:
            self.timeout_s = timeout_s
            self._post("jointreset")
        finally:
            self.timeout_s = old_timeout

    def set_load(self, payload: dict[str, Any]) -> None:
        self._post("set_load", json=payload)

    def update_param(self, payload: dict[str, Any]) -> None:
        self._post("update_param", json=payload)
