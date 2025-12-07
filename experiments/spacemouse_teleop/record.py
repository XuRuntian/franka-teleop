import numpy as np
from scipy.spatial.transform import Rotation as R
import requests
import time
import pyrealsense2 as rs
import cv2
import argparse
import os

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
        for d in rs.context().devices:
            print('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
            self.connect_device.append(d.get_info(rs.camera_info.serial_number))

        assert len(self.connect_device) == 2

        self.pipeline_front = rs.pipeline()
        self.config.enable_device(self.connect_device[1])
        self.pipeline_front.start(self.config)
        self.pipeline_side = rs.pipeline()
        self.config.enable_device(self.connect_device[0])
        self.pipeline_side.start(self.config)

        self.depth_scale = 0.0010000000474974513

    def record_frame(self):

        frames_front = self.pipeline_front.wait_for_frames()
        frames_side = self.pipeline_side.wait_for_frames()
        aligned_frames_front = self.align.process(frames_front)
        aligned_frames_side = self.align.process(frames_side)
    
        color_frame_front = aligned_frames_front.get_color_frame()
        color_frame_side = aligned_frames_side.get_color_frame()
        depth_frame_front = aligned_frames_front.get_depth_frame()
        depth_frame_side = aligned_frames_side.get_depth_frame()
        
        color_image_front = np.asanyarray(color_frame_front.get_data(), dtype=np.uint8)
        color_image_side = np.asanyarray(color_frame_side.get_data(), dtype=np.uint8)
        depth_image_front = np.asanyarray(depth_frame_front.get_data(), dtype=np.float32) * self.depth_scale * 1000
        depth_image_side = np.asanyarray(depth_frame_side.get_data(), dtype=np.float32) * self.depth_scale * 1000
        
        return color_image_front, color_image_side, depth_image_front, depth_image_side

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
    
    return np.array(cur_gripper, gripper_open)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, required=True)
    parser.add_argument('--task', type=str, default='pick_and_place', required=True)
    parser.add_argument('--camera_num', type=int, default=1)
    
    args = parser.parse_args()
    
    output_root = "/home/franka/RealBench/hil-serl/examples/experiments/spacemouse_teleop/data"
    output_dir = os.path.join(output_root, args.task, str(args.index))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    poses = []
    joints = []
    grippers = []
    record_flag = False

    if args.camera_num == 1:
        camera = Recorder()
        front_images = []
        front_depths = []
        front_images_dir = os.path.join(output_dir, "front_images")
        front_depths_dir = os.path.join(output_dir, "front_depths")
        os.makedirs(front_images_dir, exist_ok=True)
        os.makedirs(front_depths_dir, exist_ok=True)
        
        while True:
            front_image, front_depth = camera.record_frame()
            joint = get_joint()
            pose = get_pose_euler()
            gripper = get_gripper()

            cv2.imshow("RGB", front_image)
            key = cv2.waitKey(1)
            
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                camera.pipeline.stop()
                print("End Recording")
                break
            
            if key == ord('s') or key == ord('S'):
                record_flag = True
                n = 1
                print("Start Recording")
                
            if key == ord('w') or key == ord('W'):
                record_flag = False
                print("Stop Recording")
                
            if record_flag:
                print(f"Recording {n}")
                n += 1
                front_images.append(front_image.copy())
                front_depths.append(front_depth.copy())
                joints.append(joint)
                poses.append(pose)
                grippers.append(gripper)
        
        data = []
        
        for i in range(len(front_images)):
            frame_data = {
                'image': front_images[i],
                'depth': front_depths[i],
                'joint': joints[i],
                'pose': poses[i],
                'gripper': grippers[i],
            }
            data.append(frame_data)
            
            image_filename = os.path.join(front_images_dir, f"{i}.png")
            cv2.imwrite(image_filename, front_images[i])
            
            depth_filename = os.path.join(front_depths_dir, f"{i}.npy")
            np.save(depth_filename, front_depths[i])

        np.save(os.path.join(output_dir, f"{args.index}.npy"), data)
        
        print("Save Done!")

    elif args.camere_num == 2:
        cameras = MultiRecorder()        
        front_images = []
        front_depths = []
        front_images_dir = os.path.join(output_dir, "front_images")
        front_depths_dir = os.path.join(output_dir, "front_depths")
        os.makedirs(front_images_dir, exist_ok=True)
        os.makedirs(front_depths_dir, exist_ok=True)
        
        side_images = []
        side_depths = []
        side_images_dir = os.path.join(output_dir, "side_images")
        side_depths_dir = os.path.join(output_dir, "side_depths")
        os.makedirs(side_images_dir, exist_ok=True)
        os.makedirs(side_depths_dir, exist_ok=True)
        
        while True:
            front_image, front_depth, side_image, side_depth = cameras.record_frame()
            
            cv2.imshow("Front RGB", front_image)
            cv2.imshow("Side RGB", side_image)
            key = cv2.waitKey(1)
            
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                cameras.pipeline.stop()
                print("End Recording")
                break
            
            if key == ord('s') or key == ord('S'):
                record_flag = True
                print("Start Recording")
                
            if key == ord('w') or key == ord('W'):
                record_flag = False
                print("Stop Recording")
                
            if record_flag:
                front_image, front_depth, side_image, side_depth = cameras.record_frame()
                joint = get_joint()
                pose = get_pose_euler()
                gripper = get_gripper()
                front_images.append(front_image.copy())
                front_depths.append(front_depth.copy())
                side_images.append(side_image.copy())
                side_depths.append(side_depth.copy())
                joints.append(joint)
                poses.append(pose)
                grippers.append(gripper)

        data = []
        
        for i in range(len(front_images)):
            frame_data = {
                'front_image': front_images[i],
                'front_depth': front_depths[i],
                'side_image': side_images[i],
                'side_depth': side_depths[i],
                'joint': joints[i],
                'pose': poses[i],
                'gripper': grippers[i],
            }
            data.append(frame_data)
            
            image_filename = os.path.join(front_images_dir, f"{i}.png")
            cv2.imwrite(image_filename, front_images[i])
            depth_filename = os.path.join(front_depths_dir, f"{i}.npy")
            np.save(depth_filename, front_depths[i])
            side_image_filename = os.path.join(side_images_dir, f"{i}.png")
            cv2.imwrite(side_image_filename, side_images[i])
            side_depth_filename = os.path.join(side_depths_dir, f"{i}.npy")
            np.save(side_depth_filename, side_depths[i])

        np.save(os.path.join(output_dir, f"{args.index}.npy"), data)
        print("Save Done!")