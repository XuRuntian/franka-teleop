import sys
sys.path.append('/home/franka/RealBench/hil-serl/serl_launcher')
sys.path.append('/home/franka/RealBench/hil-serl/serl_robot_infra')
import numpy as np
from scipy.spatial.transform import Rotation as R
import requests
import time
import pyzed.sl as sl
import cv2
import argparse
import os
from absl import app, flags
import threading
import sys
sys.path.append('../')
from experiments.mappings import CONFIG_MAPPING
import h5py

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "spacemouse_teleop", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("index",0, "record index")
flags.DEFINE_string("output", "/home/franka/test", "output home directory.")
flags.DEFINE_string("date", "Pick_and_Place_20251020(原色幕布任务二)", "date of the collection.")
flags.DEFINE_string("task", "pick_and_place", "name of the task.")
flags.DEFINE_integer("camera_num", 3, "number of the camera.")



class ZEDMultiRecorder:
    def __init__(self):
        # 固定三台 ZED 相机序列号
        self.camera_map = {
            14463309: "cam_high",
            33717827: "cam_left",
            38232451: "cam_right"
        }

        self.cameras = {}
        self.serials = list(self.camera_map.keys())

        device_list = sl.Camera.get_device_list()
        print(f"检测到 {len(device_list)} 台 ZED 相机")
        print("设备列表:")
        for dev in device_list:
            print(f"  - SN={dev.serial_number}, Model={dev.camera_model}")

        # 初始化每台目标相机
        for dev in device_list:
            sn = dev.serial_number
            if sn not in self.camera_map:
                print(f"跳过未知相机 SN={sn}")
                continue

            cam = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.VGA  # 640x480
            init_params.camera_fps = 30
            init_params.camera_image_flip = sl.FLIP_MODE.AUTO
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.set_from_serial_number(sn)
            
            

            status = cam.open(init_params)
            
            cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 25)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"❌ 无法打开相机 SN={sn}: {repr(status)}")

            self.cameras[self.camera_map[sn]] = cam
            print(f"✅ 成功连接 {self.camera_map[sn]} (SN={sn})")

        assert len(self.cameras) == 3, "未检测到完整的三台相机（cam_high, cam_left, cam_right）"

    def record_frame(self):
        """
        同步采集三台相机的彩色图像。
        返回:
          color_high, color_left, color_right
        """
        color_images = {}
        depth_images = {}

        for name, cam in self.cameras.items():
            if cam.grab() == sl.ERROR_CODE.SUCCESS:
                color = sl.Mat()
                depth = sl.Mat()
                cam.retrieve_image(color, sl.VIEW.LEFT)
                cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

                color_images[name] = np.asanyarray(color.get_data())[:, :, :3].copy()  # BGR 格式
                depth_images[name] = np.asanyarray(depth.get_data(), dtype=np.float32).copy()
            else:
                print(f"[⚠️] 相机 {name} 抓取失败")
                color_images[name] = None
                depth_images[name] = None

        return  color_images["cam_high"], depth_images["cam_high"],  color_images["cam_left"], depth_images["cam_left"],  color_images["cam_right"], depth_images["cam_right"]
        

    def close(self):
        for name, cam in self.cameras.items():
            cam.close()
            print(f"🟢 已关闭 {name}")
        print("✅ 所有相机已关闭")
        


def get_pose_quat():
    url = "http://127.0.0.2:5000/getpos"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_pose_euler():
    url = "http://127.0.0.2:5000/getpos_euler"
    response = requests.post(url)
    cur_pose = response.json()['pose']
    
    return cur_pose

def get_joint():
    url = "http://127.0.0.2:5000/getq"
    response = requests.post(url)
    cur_joint = response.json()['q']
    
    return cur_joint    

def get_gripper():
    url = "http://127.0.0.2:5000/get_gripper"
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

def compress_image(image):
    """将RGB图像压缩为JPEG格式"""
    ret, jpeg_data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ret:
        return None
    return jpeg_data

# 主程序
def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment()
    
    status = [0]
    # 0.29895484 -0.03381186  0.41025707  0.99752339  0.04248433 -0.05579133  0.0054318 
    env.reset()
    print("Reset done")

    # input_thread = threading.Thread(target=listen_for_input, args=(env))
    # input_thread.daemon = True  # 设置为守护线程，主线程退出时该线程也会退出
    # input_thread.start()
    
    record_thread = threading.Thread(target=record, args=(status,))
    record_thread.daemon = True
    record_thread.start()
    
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

        
def record(status):
    output_root = f"{FLAGS.output}/{FLAGS.task}_{FLAGS.date}"
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    poses = []
    joints = []
    grippers = []
    left_images = []
    left_depths = []
    high_images = []
    high_depths = []
    right_images = []
    right_depths = []
    
    record_flag = False
    
    if FLAGS.camera_num == 3:
        cameras = ZEDMultiRecorder()
        # cameras = Recorder()
        index = FLAGS.index
        
        while True:

            cam_high, depth_high, cam_left, depth_left,  cam_right, depth_right = cameras.record_frame()
            
            
            cv2.imshow("Wrist RGB", cam_high)
            cv2.imshow("Left RGB", cam_left)
            cv2.imshow("Right RGB", cam_right)

            key = cv2.waitKey(1)
            
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                # cameras.pipeline.stop()
                cameras.close()
                print("End Recording")
                break
            
            if key == ord('s') or key == ord('S'):
                record_flag = True
                n = 1
                print("Start Recording")
                
            if key == ord('w') or key == ord('W'):
                record_flag = False
                print("Stop Recording")
                
                # 创建HDF5文件
                h5_filename = os.path.join(output_root, f"{index}.hdf5")
                with h5py.File(h5_filename, 'w') as f:
                    # 创建groups
                    depths_group = f.create_group('depths')
                    images_group = f.create_group('images')
                    puppet_group = f.create_group('puppet')
                    
                    # 保存深度图像
                    # depths_group.create_dataset('left_depth', 
                    #                          data=np.array(left_depths, dtype=np.uint16),
                    #                          compression="gzip",
                    #                          compression_opts=4,
                    #                          chunks=True)
                    # depths_group.create_dataset('right_depth', 
                    #                          data=np.array(right_depths, dtype=np.uint16),
                    #                          compression="gzip",
                    #                          compression_opts=4,
                    #                          chunks=True)
                    # depths_group.create_dataset('high_depth', 
                    #                          data=np.array(high_depths, dtype=np.uint16),
                    #                          compression="gzip",
                    #                          compression_opts=4,
                    #                          chunks=True)
                    
                    # 保存压缩后的RGB图像
                    left_images_compressed = [compress_image(img) for img in left_images]
                    right_images_compressed = [compress_image(img) for img in right_images]
                    high_images_compressed = [compress_image(img) for img in high_images]
                    
                    # 创建变长数据类型以存储不同大小的JPEG数据
                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    
                    images_group.create_dataset('left_image', 
                                             data=np.array(left_images_compressed, dtype=dt),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    
                    images_group.create_dataset('right_image', 
                                             data=np.array(right_images_compressed, dtype=dt),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    
                    images_group.create_dataset('high_image', 
                                             data=np.array(high_images_compressed, dtype=dt),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    
                    # 保存机器人状态
                    puppet_group.create_dataset('pose', 
                                             data=np.array(poses, dtype=np.float32),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    puppet_group.create_dataset('joint', 
                                             data=np.array(joints, dtype=np.float32),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    puppet_group.create_dataset('gripper', 
                                             data=np.array(grippers, dtype=np.float32),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)

                print(f"Save {index} Done!")
                
                # 清空列表为下一次记录做准备
                left_images.clear()
                left_depths.clear()
                high_images.clear()
                high_depths.clear()
                right_images.clear()
                right_depths.clear()
                poses.clear()
                joints.clear()
                grippers.clear()
                index += 1
                
            if key == ord('r') or key == ord('R'):
                record_flag = False
                status[0] = 1
                
            if record_flag:
                print(f"Recording {n}")
                n += 1
                joint = get_joint()
                pose = get_pose_euler()
                gripper = get_gripper()
                
                # 收集数据
                high_images.append(cam_high.copy())
                high_depths.append(depth_high.copy())
                left_images.append(cam_left.copy())
                left_depths.append(depth_left.copy())
                right_images.append(cam_right.copy())
                right_depths.append(depth_right.copy())
                joints.append(joint)
                poses.append(pose)
                grippers.append(gripper)

        # 程序结束时保存最后的数据
        if len(high_images) > 0:
            h5_filename = os.path.join(output_root, f"{index}.hdf5")
            with h5py.File(h5_filename, 'w') as f:
                depths_group = f.create_group('depths')
                images_group = f.create_group('images')
                puppet_group = f.create_group('puppet')
                
                depths_group.create_dataset('left_depth', 
                                         data=np.array(left_depths, dtype=np.uint16),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                
                depths_group.create_dataset('right_depth', 
                                         data=np.array(right_depths, dtype=np.uint16),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)       
                
                depths_group.create_dataset('high_depth', 
                                         data=np.array(high_depths, dtype=np.uint16),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                
                # 保存压缩后的RGB图像
                left_images_compressed = [compress_image(img) for img in left_images]
                right_images_compressed = [compress_image(img) for img in right_images]
                high_images_compressed = [compress_image(img) for img in high_images]
                
                # 创建变长数据类型以存储不同大小的JPEG数据
                dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                
                images_group.create_dataset('left_image', 
                                         data=np.array(left_images_compressed, dtype=dt),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                
                images_group.create_dataset('right_image', 
                                         data=np.array(right_images_compressed, dtype=dt),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
            
                images_group.create_dataset('high_image', 
                                         data=np.array(high_images_compressed, dtype=dt),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                
                puppet_group.create_dataset('pose', 
                                         data=np.array(poses, dtype=np.float32),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                puppet_group.create_dataset('joint', 
                                         data=np.array(joints, dtype=np.float32),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                puppet_group.create_dataset('gripper', 
                                         data=np.array(grippers, dtype=np.float32),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
            
            print(f"Save {index} Done!")

# 启动应用
if __name__ == '__main__':
    app.run(main)
    
    
