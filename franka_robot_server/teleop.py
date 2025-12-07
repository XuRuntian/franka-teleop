import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EXPERIMENTS_PATH = os.path.join(PROJECT_ROOT, "experiments")

sys.path.insert(0, PROJECT_ROOT)      
sys.path.insert(0, EXPERIMENTS_PATH)  

print("✅ 已添加路径：", PROJECT_ROOT)
print("✅ 已添加路径：", EXPERIMENTS_PATH)
print("🔍 Python搜索路径：", sys.path[:2])  

import numpy as np
from scipy.spatial.transform import Rotation as R
import requests
import time
import argparse
from absl import app, flags
import threading

from experiments.mappings import CONFIG_MAPPING  # 现在能找到
from franka_env.envs.wrappers import *           # 现在能找到

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "spacemouse_teleop", "Name of experiment corresponding to folder.")

FLAGS = flags.FLAGS
def get_pose_quat():
    url = "http://127.0.0.1:5000/getpos"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_pose_euler():
    url = "http://127.0.0.1:5000/getpos_euler"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_joint():
    url = "http://127.0.0.1:5000/getq"
    response = requests.post(url)
    cur_joint = response.json()['q']
    
    return cur_joint    

def get_gripper():
    url = "http://127.0.0.1:5000/get_gripper"
    response = requests.post(url)
    cur_gripper = response.json()['gripper']
    gripper_open = 1 if cur_gripper > 0.03 else 0
    
    return np.array((cur_gripper, gripper_open))

def normalize_euler_angles(roll, pitch, yaw):
    # 确保roll在[0, 2π]之间
    roll = roll % (2 * np.pi)
    if roll < 0:
        roll += 2 * np.pi
    
    return roll, pitch, yaw

def normalize_pose(pose):
    x, y, z, roll, pitch, yaw = pose
    roll, pitch, yaw = normalize_euler_angles(roll, pitch, yaw)
    
    return np.array((x, y, z, roll, pitch, yaw))

# 主程序
def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment()
    
    status = [0]
    
    env.reset()
    print("Reset done")

    # input_thread = threading.Thread(target=listen_for_input, args=(env))
    # input_thread.daemon = True  # 设置为守护线程，主线程退出时该线程也会退出
    # input_thread.start()
    

    
    while True:
        if status[0] == 0:
            actions = np.zeros(env.action_space.sample().shape) 
            env.step(actions)
            time.sleep(0.01)
        else:
            env.reset()
            status[0] = 0
            time.sleep(0.01)
            print("Reset Environment")

# 启动应用
if __name__ == '__main__':
    app.run(main)
    
    