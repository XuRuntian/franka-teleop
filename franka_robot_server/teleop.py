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
from pynput import keyboard  # 新增
def emergency_joint_reset(url="http://127.0.0.2:5000/"):
    """
    完全独立的关节复位函数。
    直接通过 HTTP 请求控制服务器，不依赖任何复杂的环境类或配置文件。
    """
    # 确保 URL 末尾有斜杠
    if not url.endswith("/"):
        url += "/"
        
    print(f"🔗 正在连接服务器: {url}")
    
    try:
        # 1. 先尝试清除机器人当前的错误锁定
        print("🛠️  正在清除错误状态 (clearerr)...")
        requests.post(url + "clearerr", timeout=5)
        
        # 2. 发送关节复位请求
        print("🤖 正在触发关节空间复位 (jointreset)...")
        # 注意：复位过程较长，timeout 设久一点
        response = requests.post(url + "jointreset", timeout=30)
        
        if response.status_code == 200:
            print("✅ 指令已送达！机器人正在复位，请注意安全。")
        else:
            print(f"⚠️  服务器响应异常，状态码: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器 {url}。请确认 franka_server.py 是否已启动？")
    except Exception as e:
        print(f"💥 复位过程中发生错误: {e}")

# 用法：
# emergency_reset(env)
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
    server_url = getattr(config, 'SERVER_URL', "http://127.0.0.2:5000/")
    status = [0]
    # --- 新增键盘监听逻辑 ---
    def on_press(key):
        try:
            # 监听 'r' 键触发关节复位
            if hasattr(key, 'char') and key.char == 'r':
                print("\n[KEYBOARD] 检测到按键 'r'，手动触发紧急关节复位...")
                emergency_joint_reset(url=server_url)
        except Exception as e:
            print(f"监听器错误: {e}")

    # 启动非阻塞监听器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("⌨️  键盘监听已启动：按下 'r' 键可随时触发关节复位。")
    # -----------------------
    env.reset()
    time.sleep(0.5)
    emergency_joint_reset()
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
    
    