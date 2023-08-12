import os
import rclpy
from rclpy.node import Node
import numpy as np
from numpy.lib import recfunctions as rfn
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import message_filters
import cv2
from .utils import wait_for_user_input
import time
from pathlib import Path

HOME = Path.home()
 
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.camera_k = None
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.camera_k_subscription = self.create_subscription(
                                        CameraInfo,
                                        "/camera/color/camera_info",
                                        self.camera_k_callback,
                                        qos_profile=qos_profile)

        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            "/camera/color/image_rect_raw",
            qos_profile=qos_profile,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            "/camera/depth/image_rect_raw",
            qos_profile=qos_profile,
        )

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, slop=0.2,
        )
        self.time_synchronizer.registerCallback(self.image_callback)

        self.output_dir = os.path.join(HOME, 'tmp')

    def image_callback(self, rgb_msg, depth_msg):
        # Wait until we have a camera k
        if self.camera_k is None:
            return

        x = wait_for_user_input()
        if x == 's':
            rgb = self.numpy_image_from_ros_image(rgb_msg, np.uint8, 3)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = self.numpy_image_from_ros_image(depth_msg, np.uint16, 1)
            depth = depth.astype('float32') / 1000
            pcd = self.rgbd_to_pcd(rgb, depth)
            pcd_path = os.path.join(self.output_dir, 'pcd_%.3f.npy'%(time.time()))
            np.save(pcd_path, pcd)
        elif x == 'q':
            self.destroy_subscription(self.time_synchronizer)
            rclpy.shutdown()

    def numpy_image_from_ros_image(self, ros_img, dtype, channels):
        np_img = np.frombuffer(ros_img.data, dtype=dtype)
        np_img = np_img.reshape(ros_img.height, ros_img.width, channels)
        return np_img
       
    def rgbd_to_pcd(self, color, depth):
        im_h, im_w = depth.shape[:2]
        u, v = np.arange(im_w), np.arange(im_h)
        uu, vv = np.meshgrid(v, u, indexing='ij')
        pixels = np.stack([vv.reshape(-1), uu.reshape(-1), np.ones(im_h*im_w)])

        # project pixels
        rays = np.linalg.inv(self.camera_k) @ pixels
        points = rays.T.reshape(im_h, im_w, 3) * depth  # im_h x im_w x 3

        mask = depth != 0
        mask = mask.reshape(im_h, im_w)
        points = points[mask]
        colors = color[mask]
        colors = colors[:, [2,1,0]]
        pcd = np.concatenate([points, colors], axis=1)
        return pcd

    def camera_k_callback(self, m):
        self.camera_k = m.k.reshape((3, 3))
        self.destroy_subscription(self.camera_k_subscription)
        self.camera_k_subscription = None
        self.get_logger().info(f'Camera K received:\n{self.camera_k}')


def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_subscriber = ImageSubscriber()

    # Spin the node so the callback function is called.
    rclpy.spin(image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()