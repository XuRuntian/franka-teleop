import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from experiments.spacemouse_teleop.wrapper import TELEOPEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    RESET_POSE = np.array([0.3057233259692392,0.0007003741368593972,0.48132603587955003,3.1405456287466036,0.012270296830370064,0.002389949492528771]) # 初始位姿 向下
    # RESET_POSE = np.array([0.3913628586687851,0.043325557670220866,0.602417345242733,3.12939075238998,0.30130763204561596,0.004790074951020358]) # 初始位姿 向前
    ACTION_SCALE = (0.04, 0.1, 1) # 映射scale
    ABS_POSE_LIMIT_LOW = RESET_POSE - np.array([2.0, 1.0, 1.0, 3.14, 3.14, 3.14])
    ABS_POSE_LIMIT_HIGH = RESET_POSE + np.array([2.0, 1.0, 1.0, 3.14, 3.14, 3.14])

    def get_environment(self):
        env = TELEOPEnv(
            config=EnvConfig(),
        )
        env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        return env