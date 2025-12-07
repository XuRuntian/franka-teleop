import sys
sys.path.append('/home/franka/RealBench/hil-serl/serl_launcher')
sys.path.append('/home/franka/RealBench/hil-serl/serl_robot_infra')
import numpy as np
from scipy.spatial.transform import Rotation as R
import requests
import time
import pyrealsense2 as rs
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
flags.DEFINE_string("date", "251020_black", "date of the collection.")
flags.DEFINE_string("task", "pick_and_place", "name of the task.")
flags.DEFINE_integer("camera_num", 2, "number of the camera.")

class Recorder:
    def __init__(self):

        self.connect_device = []
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            self.connect_device.append(d.get_info(rs.camera_info.serial_number))
        assert len(self.connect_device) == 1
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.connect_device[0])
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def record_frame(self):
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale * 1000
        
        return color_image, depth_image

class MultiRecorder:
    def __init__(self):
        self.align = rs.align(rs.stream.color)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.connect_device = []
        # import pdb; pdb.set_trace()
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), 
          ' Serial Number: ', d.get_info(rs.camera_info.serial_number),
          ' Device Type: ', d.get_info(rs.camera_info.product_line))
            #print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            #if d.get_info(rs.camera_info.serial_number) == "f1420629":
            #    continue
            #if d.get_info(rs.camera_info.serial_number) == "f1422292":
            #     continue
            
            self.connect_device.append(d.get_info(rs.camera_info.serial_number))
        assert len(self.connect_device) == 2

        self.pipline_l515 = rs.pipeline()
        self.config.enable_device(self.connect_device[1])
        self.profile_l515 = self.pipline_l515.start(self.config)
        self.pipline_d435 = rs.pipeline()
        self.config.enable_device(self.connect_device[0])
        self.profile_d435 = self.pipline_d435.start(self.config)
        
        self.depth_scale_l515 = self.profile_l515.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale_d435 = self.profile_d435.get_device().first_depth_sensor().get_depth_scale()


        self.color_sensor_l515 = self.profile_l515.get_device().query_sensors()[1]  # [0]=depth, [1]=color
        self.color_sensor_d435 = self.profile_d435.get_device().query_sensors()[1]

        self.color_sensor_l515.set_option(rs.option.enable_auto_exposure, False)
        self.color_sensor_d435.set_option(rs.option.enable_auto_exposure, False)
        
        # self.color_sensor_d435.set_option(rs.option.enable_auto_white_balance, False)
        # self.color_sensor_d435.set_option(rs.option.white_balance, 5000)
        
        self.color_sensor_l515.set_option(rs.option.exposure, 500)  # L515 曝光时间
        self.color_sensor_l515.set_option(rs.option.gain, 32)        # L515 增益
        self.color_sensor_d435.set_option(rs.option.exposure, 150)  # D435 曝光时间
        self.color_sensor_d435.set_option(rs.option.gain, 32)        # D435 增益
        



        self.intr_l515 = None
        self.intr_d435 = None

    def record_frame(self):

        frames_front = self.pipline_l515.wait_for_frames()
        frames_wrist = self.pipline_d435.wait_for_frames()
        aligned_frames_front = self.align.process(frames_front)
        aligned_frames_wrist = self.align.process(frames_wrist)
        self.intr_l515 = aligned_frames_front.get_profile().as_video_stream_profile().get_intrinsics()
        self.intr_d435 = aligned_frames_wrist.get_profile().as_video_stream_profile().get_intrinsics()
        
        print(self.intr_l515)
    
        color_frame_front = aligned_frames_front.get_color_frame()
        color_frame_wrist = aligned_frames_wrist.get_color_frame()
        depth_frame_front = aligned_frames_front.get_depth_frame()
        depth_frame_wrist = aligned_frames_wrist.get_depth_frame()
        
        color_image_front = np.asanyarray(color_frame_front.get_data(), dtype=np.uint8)
        color_image_wrist = np.asanyarray(color_frame_wrist.get_data(), dtype=np.uint8)
        depth_image_front = np.asanyarray(depth_frame_front.get_data(), dtype=np.float32) * self.depth_scale_l515 * 1000
        depth_image_wrist = np.asanyarray(depth_frame_wrist.get_data(), dtype=np.float32) * self.depth_scale_d435 * 1000
        
        return color_image_front, color_image_wrist, depth_image_front, depth_image_wrist

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
    front_images = []
    front_depths = []
    wrist_images = []
    wrist_depths = []
    record_flag = False
    
    if FLAGS.camera_num == 2:
        cameras = MultiRecorder()
        # cameras = Recorder()
        index = FLAGS.index
        
        while True:

            front_image, wrist_image, front_depth, wrist_depth = cameras.record_frame()
            
            cv2.imshow("Front RGB", front_image)
            cv2.imshow("Wrist RGB", wrist_image)
            key = cv2.waitKey(1)
            
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                # cameras.pipeline.stop()
                cameras.pipline_l515.stop()
                cameras.pipline_d435.stop()
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
                    depths_group.create_dataset('front_depth', 
                                             data=np.array(front_depths, dtype=np.uint16),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    depths_group.create_dataset('wrist_depth', 
                                             data=np.array(wrist_depths, dtype=np.uint16),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    
                    # 保存压缩后的RGB图像
                    front_images_compressed = [compress_image(img) for img in front_images]
                    wrist_images_compressed = [compress_image(img) for img in wrist_images]
                    
                    # 创建变长数据类型以存储不同大小的JPEG数据
                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    
                    images_group.create_dataset('front_image', 
                                             data=np.array(front_images_compressed, dtype=dt),
                                             compression="gzip",
                                             compression_opts=4,
                                             chunks=True)
                    images_group.create_dataset('wrist_image', 
                                             data=np.array(wrist_images_compressed, dtype=dt),
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
                front_images.clear()
                front_depths.clear()
                wrist_images.clear()
                wrist_depths.clear()
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
                front_images.append(front_image.copy())
                front_depths.append(front_depth.copy())
                wrist_images.append(wrist_image.copy())
                wrist_depths.append(wrist_depth.copy())
                joints.append(joint)
                poses.append(pose)
                grippers.append(gripper)

        # 程序结束时保存最后的数据
        if len(front_images) > 0:
            h5_filename = os.path.join(output_root, f"{index}.hdf5")
            with h5py.File(h5_filename, 'w') as f:
                depths_group = f.create_group('depths')
                images_group = f.create_group('images')
                puppet_group = f.create_group('puppet')
                
                depths_group.create_dataset('front_depth', 
                                         data=np.array(front_depths, dtype=np.uint16),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                depths_group.create_dataset('wrist_depth', 
                                         data=np.array(wrist_depths, dtype=np.uint16),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                
                # 保存压缩后的RGB图像
                front_images_compressed = [compress_image(img) for img in front_images]
                wrist_images_compressed = [compress_image(img) for img in wrist_images]
                
                # 创建变长数据类型以存储不同大小的JPEG数据
                dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                
                images_group.create_dataset('front_image', 
                                         data=np.array(front_images_compressed, dtype=dt),
                                         compression="gzip",
                                         compression_opts=4,
                                         chunks=True)
                images_group.create_dataset('wrist_image', 
                                         data=np.array(wrist_images_compressed, dtype=dt),
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
    
    
