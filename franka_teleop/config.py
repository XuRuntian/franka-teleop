from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _nested_get(data: dict[str, Any], key: str, default: Any) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


@dataclass
class MotionConfig:
    rate_hz: float = 20.0
    translation_scale: float = 0.04
    rotation_scale: float = 0.10
    deadband: float = 0.001
    max_translation_step: float = 0.04
    max_rotation_step: float = 0.10
    reference_frame: str = "base"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MotionConfig":
        return cls(
            rate_hz=float(data.get("rate_hz", cls.rate_hz)),
            translation_scale=float(data.get("translation_scale", cls.translation_scale)),
            rotation_scale=float(data.get("rotation_scale", cls.rotation_scale)),
            deadband=float(data.get("deadband", cls.deadband)),
            max_translation_step=float(data.get("max_translation_step", cls.max_translation_step)),
            max_rotation_step=float(data.get("max_rotation_step", cls.max_rotation_step)),
            reference_frame=str(data.get("reference_frame", cls.reference_frame)),
        )

    def validate(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("motion.rate_hz must be positive")
        if self.reference_frame not in {"base", "tcp"}:
            raise ValueError("motion.reference_frame must be 'base' or 'tcp'")


@dataclass
class WorkspaceConfig:
    low: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0], dtype=np.float64))
    high: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float64))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "WorkspaceConfig":
        return cls(
            low=_array(data.get("low", cls().low), (3,), "workspace.low"),
            high=_array(data.get("high", cls().high), (3,), "workspace.high"),
        )

    def validate(self) -> None:
        if np.any(self.low >= self.high):
            raise ValueError("workspace.low must be strictly lower than workspace.high")


@dataclass
class LoadConfig:
    mass: float = 0.0
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float64))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LoadConfig":
        return cls(
            mass=float(data.get("mass", cls.mass)),
            center_of_mass=_array(
                data.get("center_of_mass", data.get("F_x_center_load", cls().center_of_mass)),
                (3,),
                "tool.load.center_of_mass",
            ),
            inertia=_array(
                data.get("inertia", data.get("load_inertia", cls().inertia)),
                (9,),
                "tool.load.inertia",
            ),
        )

    def as_server_payload(self) -> dict[str, Any]:
        return {
            "mass": self.mass,
            "F_x_center_load": self.center_of_mass.tolist(),
            "load_inertia": self.inertia.tolist(),
        }


@dataclass
class ToolConfig:
    name: str = "default"
    ee_tcp_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ee_tcp_quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    load: LoadConfig = field(default_factory=LoadConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ToolConfig":
        transform = data.get("ee_T_tcp", {})
        return cls(
            name=str(data.get("name", cls.name)),
            ee_tcp_xyz=_array(transform.get("xyz", cls().ee_tcp_xyz), (3,), "tool.ee_T_tcp.xyz"),
            ee_tcp_quat=_array(transform.get("quat", cls().ee_tcp_quat), (4,), "tool.ee_T_tcp.quat"),
            load=LoadConfig.from_mapping(data.get("load", {})),
        )

    def validate(self) -> None:
        norm = np.linalg.norm(self.ee_tcp_quat)
        if norm < 1e-8:
            raise ValueError("tool.ee_T_tcp.quat must be non-zero")
        self.ee_tcp_quat = self.ee_tcp_quat / norm


@dataclass
class GripperConfig:
    enabled: bool = True
    close_button: int = 0
    open_button: int = -1
    cooldown_s: float = 0.6

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "GripperConfig":
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            close_button=int(data.get("close_button", cls.close_button)),
            open_button=int(data.get("open_button", cls.open_button)),
            cooldown_s=float(data.get("cooldown_s", cls.cooldown_s)),
        )


@dataclass
class TeleopConfig:
    server_url: str = "http://127.0.0.2:5000/"
    request_timeout_s: float = 2.0
    clear_errors: bool = True
    apply_load_on_start: bool = False
    motion: MotionConfig = field(default_factory=MotionConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TeleopConfig":
        return cls(
            server_url=str(data.get("server_url", cls.server_url)),
            request_timeout_s=float(data.get("request_timeout_s", cls.request_timeout_s)),
            clear_errors=bool(data.get("clear_errors", cls.clear_errors)),
            apply_load_on_start=bool(data.get("apply_load_on_start", cls.apply_load_on_start)),
            motion=MotionConfig.from_mapping(data.get("motion", {})),
            workspace=WorkspaceConfig.from_mapping(data.get("workspace", {})),
            tool=ToolConfig.from_mapping(data.get("tool", {})),
            gripper=GripperConfig.from_mapping(data.get("gripper", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TeleopConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_mapping(json.load(f))

    def validate(self) -> None:
        self.motion.validate()
        self.workspace.validate()
        self.tool.validate()

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "server_url":
                self.server_url = str(value)
            elif key == "rate_hz":
                self.motion.rate_hz = float(value)
            elif key == "translation_scale":
                self.motion.translation_scale = float(value)
            elif key == "rotation_scale":
                self.motion.rotation_scale = float(value)
            elif key == "reference_frame":
                self.motion.reference_frame = str(value)
            elif key == "workspace_low":
                self.workspace.low = _array(value, (3,), "workspace.low")
            elif key == "workspace_high":
                self.workspace.high = _array(value, (3,), "workspace.high")
            elif key == "tool_xyz":
                self.tool.ee_tcp_xyz = _array(value, (3,), "tool.ee_T_tcp.xyz")
            elif key == "tool_quat":
                self.tool.ee_tcp_quat = _array(value, (4,), "tool.ee_T_tcp.quat")
            elif key == "gripper_enabled":
                self.gripper.enabled = bool(value)
            else:
                raise ValueError(f"Unknown override: {key}")
        self.validate()


def load_config(path: str | Path | None) -> TeleopConfig:
    config = TeleopConfig.from_json(path) if path else TeleopConfig()
    config.validate()
    return config
