import os
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import message_filters
import numpy as np
import yaml
from typing import Tuple
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import time
import pickle
import csv
 
def numpy_image_from_ros_image(ros_img, dtype, channels):
    np_img = np.frombuffer(ros_img.data, dtype=dtype)
    np_img = np_img.reshape(ros_img.height, ros_img.width, channels)
    return np_img


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        param_file = os.path.join(
                        get_package_share_directory('feature_tracking'),
                        'config',
                        'config.yaml'
                    )
        with open(param_file, 'r') as f:
            self.params = yaml.load(f, Loader=yaml.Loader)
        
        self.log_dir = os.path.join(self.params['log_dir'], 
                                    '%d'%(int(time.time())))
        self.get_logger().info("Logging tracks to %s"%(self.log_dir))
        self.color_dir = os.path.join(self.log_dir, 'color')
        self.depth_dir = os.path.join(self.log_dir, 'depth')                            
        self.feature_track_dir = os.path.join(self.log_dir, 'feature_3d')
        for d in [self.log_dir, self.color_dir, self.depth_dir, self.feature_track_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)
        csv_fname = os.path.join(self.log_dir, 'dataframe.csv')
        csvfile = open(csv_fname, 'w+')
        self.csvwriter = csv.writer(csvfile)

        self.camera_k = None
        self.im_h = None
        self.im_w = None
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.camera_k_subscription = self.create_subscription(
                                        CameraInfo,
                                        "/camera/color/camera_info",
                                        self.camera_k_callback,
                                        qos_profile=qos_profile)

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/depth/image_rect_raw',
            qos_profile=qos_profile,
        )
        self.color_sub = message_filters.Subscriber(
            self,
            Image, 
            '/camera/color/image_rect_raw', 
            qos_profile=qos_profile,
        )
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.color_sub], 10, slop=0.5
        )
        self.time_synchronizer.registerCallback(self.rgbd_callback)

        self.br = CvBridge()

        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize = (15, 15),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.win_size = 20
        self.ft_track = {}
        self.tmax = 2
        self.track_cnt = 0
        self.frame_count = 0

        # create random colours for visualization for all 100 max corners for RGB channels
        self.num_color = 100
        self.colors = np.random.randint(0, 255, (self.num_color, 3))

    def rgbd_callback(self, depth_msg, color_msg):
        color_img = self.br.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_img = numpy_image_from_ros_image(depth_msg, np.uint16, 1)
        canvas = np.zeros_like(color_img)
        
        # Wait until we have a camera k
        if self.camera_k is None:
            return

        print("Camera intrinsics: \n", self.camera_k)

        # Display the message on the console
        self.get_logger().info('Receiving video frame %d'%self.frame_count)

        # Convert ROS Image message to OpenCV image
        curr_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        curr_pts = cv2.goodFeaturesToTrack(curr_img, mask = None, **self.feature_params)
        curr_pts = curr_pts.reshape(-1, 2)

        curr_time = time.time()

        if self.frame_count == 0:
            self.ft_track[curr_time] = {}
            for pt in curr_pts:
                pt = tuple(pt)
                self.ft_track[curr_time][pt] = self.track_cnt
                self.track_cnt += 1

            forward_pts = curr_pts

        else:
            # update object corners by comparing with found edges in initial frame
            forward_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_img,
                                                            curr_img, 
                                                            self.prev_pts, 
                                                            None,
                                                            **self.lk_params)
            inbound = self.inbound_check(forward_pts, self.prev_pts)
            status = status.reshape(-1) & inbound

            valid_prev_pts = self.prev_pts[status==1]
            valid_forward_pts = forward_pts[status==1]
            self.add_frame(valid_prev_pts,
                           valid_forward_pts, 
                           curr_time)
            
            curr_pts = self.trim_pts(curr_pts, valid_forward_pts)
            forward_pts = np.concatenate([curr_pts, valid_forward_pts], axis=0)
            
            # Visualize tracks
            for f in self.ft_track.values():
                for pt, idx in f.items():
                    a, b = pt
                    color = self.colors[idx % self.num_color].tolist()
                    canvas = cv2.circle(canvas, (int(a), int(b)), 5, color, -1)

            # Log frame
            name_base = '%.4f'%curr_time
            row = ['%.4f'%curr_time,
                   os.path.join(self.color_dir, '%s.png'%name_base),
                   os.path.join(self.depth_dir, '%s.npy'%name_base),
                   os.path.join(self.feature_track_dir, '%s.pkl'%name_base)]
            self.csvwriter.writerow(row)
            cv2.imwrite(row[1], color_img)
            np.save(row[2], depth_img)
            with open(row[3], 'wb') as f:
                pickle.dump(self.ft_track[curr_time], f)
        
        self.pop_front(curr_time)
        self.prev_img = curr_img
        self.prev_pts = forward_pts
        self.prev_time = curr_time

        self.frame_count += 1

        result = cv2.add(color_img, canvas)
        cv2.imshow('Optical Flow (sparse)', result)

        cv2.waitKey(1)


    def add_frame(self, valid_prev, valid_forward, t):
        '''
        Args:
            curr_pts -- array of 2D points from goodFeaturesToTrack()
            valid_prev -- array of 2D points from calcOpticalFlowPyrLK()
                            only valid ones from previous frame
            valid_forward -- array of 2D points from calcOpticalFlowPyrLK()
                            only valid ones  from current frame
            t -- time of the frame to be added
        '''

        self.get_logger().info("Adding points in frame %d"%self.frame_count)
        self.ft_track[t] = {}     
        
        for pt1, pt2 in zip(valid_prev, valid_forward):
            pt1, pt2 = tuple(pt1), tuple(pt2)
            # If feature is already tracked
            if pt1 in self.ft_track[self.prev_time]:
                self.ft_track[t][pt2] = self.ft_track[self.prev_time][pt1]
            # Start a new feature track
            else:
                self.track_cnt += 1
                self.ft_track[self.prev_time][pt1] = self.track_cnt
                self.ft_track[t][pt2] = self.track_cnt

        self.get_logger().info("Number of feature tracks: %d"%self.track_cnt)

    def trim_pts(self, curr_pts, valid_forward_pts):
        '''
        Reject feature points that are too close to the tracked points
        '''
        m, n = curr_pts.shape[0], valid_forward_pts.shape[0]
        A = np.tile(curr_pts.reshape(m, 1, 2), (1, n, 1))
        B = np.tile(valid_forward_pts.reshape(1, n, 2), (m, 1, 1))
        dist = np.linalg.norm(A - B, axis=2)
        valid = np.any(dist < 2, axis=1)
        return curr_pts[~valid]

    def pop_front(self, t_end):
        '''
        Pop the old frames in the feature track
        '''
        to_del = [t for t in self.ft_track.keys() if t_end - t > self.tmax]
        for t in to_del:
            del self.ft_track[t]

        self.get_logger().info("Old frames, ft_track size: %d"%len(self.ft_track))

    def inbound_check(self, *args):
        mask = np.ones(args[0].shape[0]).astype(bool)
        for item in args:
            mask = mask & (item[:,0] >= 0) \
                        & (item[:,0] <= self.im_w) \
                        & (item[:,1] >= 0) \
                        & (item[:,1] <= self.im_h)
        return mask

    def camera_k_callback(self, m):
        self.camera_k = m.k.reshape((3, 3))
        self.im_h = m.height
        self.im_w = m.width
        np.savetxt(os.path.join(self.log_dir, 'camera_k.txt'), self.camera_k)
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