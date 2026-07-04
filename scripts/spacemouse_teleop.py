#!/usr/bin/env python3
from __future__ import annotations

import argparse

from franka_teleop.config import TeleopConfig, load_config
from franka_teleop.controller import CartesianTeleopController


def _float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SpaceMouse teleop for franka_robot_server.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON teleop config.")
    parser.add_argument("--server-url", type=str, default=None, help="Override server URL.")
    parser.add_argument("--rate-hz", type=float, default=None, help="Override command rate.")
    parser.add_argument("--translation-scale", type=float, default=None, help="Meters per full SpaceMouse step.")
    parser.add_argument("--rotation-scale", type=float, default=None, help="Radians per full SpaceMouse step.")
    parser.add_argument(
        "--reference-frame",
        choices=["base", "tcp"],
        default=None,
        help="Interpret SpaceMouse delta in the robot base frame or TCP frame.",
    )
    parser.add_argument("--workspace-low", type=_float_list, default=None, help="Comma-separated xyz lower bound.")
    parser.add_argument("--workspace-high", type=_float_list, default=None, help="Comma-separated xyz upper bound.")
    parser.add_argument("--tool-xyz", type=_float_list, default=None, help="Comma-separated EE-to-TCP xyz offset.")
    parser.add_argument("--tool-quat", type=_float_list, default=None, help="Comma-separated EE-to-TCP xyzw quaternion.")
    parser.add_argument("--disable-gripper", action="store_true", help="Disable SpaceMouse button gripper commands.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TeleopConfig:
    config = load_config(args.config)
    config.apply_overrides(
        {
            "server_url": args.server_url,
            "rate_hz": args.rate_hz,
            "translation_scale": args.translation_scale,
            "rotation_scale": args.rotation_scale,
            "reference_frame": args.reference_frame,
            "workspace_low": args.workspace_low,
            "workspace_high": args.workspace_high,
            "tool_xyz": args.tool_xyz,
            "tool_quat": args.tool_quat,
            "gripper_enabled": False if args.disable_gripper else None,
        }
    )
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    controller = CartesianTeleopController(config=config)
    controller.run_forever()


if __name__ == "__main__":
    main()
