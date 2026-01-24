import subprocess
import rospy
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg
import time
from gripper_server import GripperServer

import threading

class RobotiqGripperServer(GripperServer):
    def __init__(self, gripper_ip):
        super().__init__()
        self.gripper = subprocess.Popen(
            [
                "rosrun",
                "robotiq_2f_gripper_control",
                "Robotiq2FGripperRtuNode.py",
                gripper_ip,
            ],
            stdout=subprocess.PIPE,
        )
        self.gripper_state_sub = rospy.Subscriber(
            "Robotiq2FGripperRobotInput",
            inputMsg.Robotiq2FGripper_robot_input,
            self._update_gripper,
            queue_size=1,
        )
        self.gripperpub = rospy.Publisher(
            "Robotiq2FGripperRobotOutput",
            outputMsg.Robotiq2FGripper_robot_output,
            queue_size=1,
        )
        self.gripper_command = outputMsg.Robotiq2FGripper_robot_output()

    def activate_gripper(self):
        self.gripper_command = self._generate_gripper_command("a", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)
        rospy.loginfo("Waiting for gripper activation...")
        time.sleep(2.5) # 给硬件初始化留出时间

    def reset_gripper(self):
        # 1. 先彻底重置
        self.gripper_command = self._generate_gripper_command("r", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)
        time.sleep(0.5)
        # 2. 再重新激活
        self.activate_gripper()

    def open(self):
        # 开一个新线程去处理，不阻塞主逻辑
        threading.Thread(target=self._open_task).start()

    def _open_task(self):
        self.gripper_command = self._generate_gripper_command("o", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)
        # 如果需要等待动作完成，可以在这里 sleep，不会卡死主服务器

    def close(self):
        # 开一个新线程去处理，不阻塞主逻辑
        threading.Thread(target=self._close_task).start()

    def _close_task(self):
        self.gripper_command = self._generate_gripper_command("c", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def move(self, position):
        self.gripper_command = self._generate_gripper_command(position, self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def close_slow(self):
        self.gripper_command = self._generate_gripper_command("cs", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def _update_gripper(self, msg):
        """internal callback to get the latest gripper position."""
        self.gripper_pos = 1 - msg.gPO / 255

    def _generate_gripper_command(self, char, command):
        """
        修正后的 Robotiq 命令生成逻辑
        确保 rACT (激活) 和 rGTO (运行到位置) 始终为有效状态
        """
        # 预设基础状态：确保夹爪处于激活并允许移动的状态
        # 注意：这里不能每次都重置 command，要保留已有的配置
        command.rACT = 1  # 强制保持激活
        command.rGTO = 1  # 强制启用运动
        
        # 默认速度和力量（如果之前没设过）
        if command.rSP == 0: command.rSP = 255
        if command.rFR == 0: command.rFR = 150

        if char == "a":
            # 激活序列
            command.rACT = 1
            command.rGTO = 1
            command.rSP = 255
            command.rFR = 150
            print("Gripper Command: ACTIVATE")

        elif char == "r":
            # 重置序列：ACT 必须为 0
            command.rACT = 0
            print("Gripper Command: RESET")

        elif char == "c":
            command.rPR = 255 # 闭合
            print("Gripper Command: CLOSE")
        
        elif char == "o":
            command.rPR = 0   # 打开
            print("Gripper Command: OPEN")

        elif char == "cs":
            # 慢速闭合
            command.rPR = 255
            command.rSP = 50 # 降低速度
            print("Gripper Command: CLOSE SLOW")

        else:
            # 处理数值指令 (0-255)
            try:
                val = int(char)
                command.rPR = max(0, min(255, val))
                print(f"Gripper Command: MOVE TO {val}")
            except (ValueError, TypeError):
                pass

        return command