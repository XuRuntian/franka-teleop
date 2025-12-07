#!/usr/bin/env python3
"""
测试 franka_server 的 REST 接口
"""
import requests
import numpy as np
import time

BASE_URL = "http://127.0.0.2:5000"  # 与 franka_server.py 的 flask_url 一致


# ----------------- 工具 -----------------
def post(route, json=None):
    resp = requests.post(f"{BASE_URL}/{route}", json=json)
    resp.raise_for_status()
    print(f"[POST] /{route} -> {resp.content}")
    return resp 


# ----------------- 1. 拿状态 -----------------
def test_get_state():
    print("→ 获取机器人完整状态")
    r = post("getstate").json()
    print("pose   :", np.round(r["pose"], 3))
    print("force  :", np.round(r["force"], 3))
    print("q      :", np.round(r["q"], 3))
    print("gripper:", r["gripper_pos"])


# ----------------- 2. 夹爪 -----------------
def test_gripper():
    print("\n→ 夹爪测试")
    post("open_gripper")
    time.sleep(1)
    post("close_gripper")
    time.sleep(1)
    print("当前开度:", post("get_gripper").json()["gripper"])


# ----------------- 3. 走点 -----------------
def test_move():
    print("\n→ 走点测试")
    curr = post("getpos").json()["pose"]
    print("当前 pose:", np.round(curr, 3))
    target = np.array(curr)
    target[2] += 0.1  # 抬升5cm
    print("目标 pose:", np.round(target, 3))
    
    # 循环发送10次目标位姿（确保控制器收到持续指令）
    for _ in range(10):
        post("pose", {"arr": target.tolist()})
        time.sleep(0.1)
    
    time.sleep(2)
    print("到位后 pose:", np.round(post("getpos").json()["pose"], 3))


# ----------------- 4. 关节复位 -----------------
def test_joint_reset():
    print("\n→ 关节复位（耗时 ~10 s）")
    post("jointreset")
    time.sleep(10)
    print("复位完成")


# ----------------- 5. 一键自检 -----------------
def full_test():
    test_get_state()
    test_gripper()
    test_move()
    # test_joint_reset()  # 需要时打开，时间较长


if __name__ == "__main__":
    full_test()
