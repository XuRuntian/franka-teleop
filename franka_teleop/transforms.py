from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to a homogeneous transform."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"pose must have shape (7,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to [x, y, z, qx, qy, qz, qw]."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    return np.concatenate(
        [transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()]
    )


def transform_from_xyz_quat(xyz: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pose = np.concatenate([np.asarray(xyz, dtype=np.float64), np.asarray(quat, dtype=np.float64)])
    return pose_to_matrix(pose)


def clip_translation_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(delta)
    if norm <= max_norm or norm < 1e-12:
        return delta
    return delta * (max_norm / norm)


def clip_rotation_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(delta)
    if norm <= max_norm or norm < 1e-12:
        return delta
    return delta * (max_norm / norm)


def apply_cartesian_delta(
    tcp_pose: np.ndarray,
    translation_delta: np.ndarray,
    rotation_delta: np.ndarray,
    reference_frame: str,
) -> np.ndarray:
    """Apply a small Cartesian delta to a TCP pose.

    base: translation and rotation increments are interpreted in the robot base frame.
    tcp: translation and rotation increments are interpreted in the current TCP frame.
    """
    tcp_transform = pose_to_matrix(tcp_pose)
    translation_delta = np.asarray(translation_delta, dtype=np.float64)
    rotation_delta = np.asarray(rotation_delta, dtype=np.float64)

    if reference_frame == "base":
        next_pose = np.asarray(tcp_pose, dtype=np.float64).copy()
        next_pose[:3] += translation_delta
        current_rot = Rotation.from_quat(next_pose[3:])
        delta_rot = Rotation.from_euler("xyz", rotation_delta)
        next_pose[3:] = (delta_rot * current_rot).as_quat()
        return next_pose

    if reference_frame == "tcp":
        delta_transform = np.eye(4, dtype=np.float64)
        delta_transform[:3, :3] = Rotation.from_euler("xyz", rotation_delta).as_matrix()
        delta_transform[:3, 3] = translation_delta
        return matrix_to_pose(tcp_transform @ delta_transform)

    raise ValueError("reference_frame must be 'base' or 'tcp'")


def ee_pose_to_tcp_pose(ee_pose: np.ndarray, ee_t_tcp: np.ndarray) -> np.ndarray:
    return matrix_to_pose(pose_to_matrix(ee_pose) @ ee_t_tcp)


def tcp_pose_to_ee_pose(tcp_pose: np.ndarray, ee_t_tcp: np.ndarray) -> np.ndarray:
    return matrix_to_pose(pose_to_matrix(tcp_pose) @ np.linalg.inv(ee_t_tcp))
