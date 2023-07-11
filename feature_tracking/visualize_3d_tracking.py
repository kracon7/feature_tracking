import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from typing import Tuple
import numpy as np
import pickle
import pandas as pd
from PIL import Image as pil_img
from .utils import np_pcd_to_colored_ros_pcd, np_pcd_to_ros_pcd


def proj_to_3d(camera_k, depth, pix):
    '''
    Project from 2D pixel locations to 3D points
    Args:
        camera_k -- (np.array) 3 x 3, intrinsic matrix
        depth -- (np.array) h x w, depth array
        pix -- (np.array) N x 2, pixel locations
    '''
    z = depth[pix]
    pix = np.insert(pix, 1, -1, axis=1)
    rays = (np.linalg.inv(camera_k) @ pix.T).T

    points = rays * z
    return points


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.pcd_publisher = self.create_publisher(PointCloud2, 
                                                   'd405_points', 
                                                   qos_profile=qos_profile)
        self.track_publisher = self.create_publisher(PointCloud2, 
                                                   'track_points', 
                                                   qos_profile=qos_profile)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        log_dir = '/tmp/1689053159'
        self.camera_k = np.loadtxt(os.path.join(log_dir, 'camera_k.txt'))
        self.df = pd.read_csv(os.path.join(log_dir, 'dataframe.csv'))
        self.num_frame = len(self.df.iloc[:,0])

        # create random colours for visualization for all 100 max corners for RGB channels
        self.num_color = 200
        self.colors = np.random.randint(0, 255, (self.num_color, 3))

        self.frame_buffer = []

        self.i = 0

    def timer_callback(self):
        color_arr = np.asarray(pil_img.open(self.df.iloc[self.i,1]))
        depth_arr = np.load(self.df.iloc[self.i, 2]).astype('float') / 1000
        depth_arr[depth_arr > 2] = 0
        pcd_msg = self.rgbd_to_ros_pcd(color_arr, depth_arr)
        self.pcd_publisher.publish(pcd_msg)

        frame = pickle.load(open(self.df.iloc[self.i, 3], 'rb'))
        # import pdb
        # pdb.set_trace()
        points, color = self.extract_frame(frame, depth_arr)
        self.frame_buffer.append([points, color])
        track_msg = self.frame_buffer_to_ros_pcd()
        self.track_publisher.publish(track_msg)

        while len(self.frame_buffer) > 30:
            self.frame_buffer.pop(0)

        self.get_logger().info('Publishing frame %d'%self.i)
        self.i += 1

    
    def rgbd_to_ros_pcd(self, color, depth):
        im_h, im_w = depth.shape[:2]
        x, y = np.arange(im_w), np.arange(im_h)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=2).reshape(-1,2)

        # project pixels
        rays = np.insert(points, 2, 1, axis=1) @ np.linalg.inv(self.camera_k).T
        points = rays.reshape(im_h, im_w, 3) * depth  # im_h x im_w x 3

        mask = depth != 0
        mask = mask.reshape(im_h, im_w)
        points = points[mask]
        colors = color[mask]

        pcd = np_pcd_to_colored_ros_pcd(points, colors, '/camera_link')
        return pcd

    def frame_buffer_to_ros_pcd(self):
        points = np.concatenate([item[0] for item in self.frame_buffer], axis=0)
        colors = np.concatenate([item[1] for item in self.frame_buffer], axis=0)
        pcd = np_pcd_to_colored_ros_pcd(points, colors, '/camera_link')
        return pcd

    def extract_frame(self, frame, depth):
        pix = list(frame.keys())
        pix = [list(i) for i in pix]
        pix = np.asarray(pix).astype('int')

        idx = np.fromiter(frame.values(), dtype=int)
        idx = idx % self.num_color

        z = depth[pix[:,1], pix[:,0]]
        mask = (z != 0).reshape(-1)

        pix, z, idx = pix[mask], z[mask], idx[mask]

        pix = np.insert(pix, 2, 1, axis=1)
        rays = pix @ np.linalg.inv(self.camera_k).T
        points = rays * z.reshape(-1, 1)
        colors = self.colors[idx]
        
        return points, colors


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
