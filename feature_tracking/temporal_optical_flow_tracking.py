import rclpy
from rclpy.node import Node
import numpy as np
from numpy.lib import recfunctions as rfn
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import message_filters
from cv_bridge import CvBridge
import cv2
from .feature_manager import FeaturePoint, FeatureTrack, FrameNode, FrameTrack, FeatureManager
import time
import cProfile
from .utils import MotionFieldVisualizer, np_pcd_to_colored_ros_pcd, np_pcd_to_ros_pcd
 
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

        self.pcd_publisher = self.create_publisher(PointCloud2, 
                                                   'd405_points', 
                                                   qos_profile=qos_profile)

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, slop=0.2,
        )
        self.time_synchronizer.registerCallback(self.image_callback)

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
        self.track_cnt = 0
        self.frame_count = 0
        self.frame_skip = 2

        # create random colours for visualization for all 100 max corners for RGB channels
        self.num_color = 100
        self.colors = np.random.randint(0, 255, (self.num_color, 3))

        self.feature_manager = FeatureManager()

        self.motion_field_visualizer = MotionFieldVisualizer()

    def image_callback(self, rgb_msg, depth_msg):
        # Wait until we have a camera k
        if self.camera_k is None:
            return
        
        # # Profiling 
        # cProfile.runctx('self.process_images(rgb_msg, depth_msg)', globals(), locals(), 
        #                 'profiling_result.prof')

        self.process_images(rgb_msg, depth_msg)
        ids, locs, vecs = self.compute_track_velocity()
        self.motion_field_visualizer.visualize_mfield(ids, locs, vecs)
        
    def process_images(self, rgb_msg, depth_msg):
        
        self.get_logger().info('Receiving video frame %d'%self.frame_count)
        self.frame_count += 1

        if self.frame_count % self.frame_skip == 0:
            return
        
        now = time.time()
    
        rgb = self.numpy_image_from_ros_image(rgb_msg, np.uint8, 3)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = self.numpy_image_from_ros_image(depth_msg, np.uint16, 1)
        depth = depth.astype('float32') / 1000

        pcd_msg = self.rgbd_to_ros_pcd(rgb, depth)
        self.pcd_publisher.publish(pcd_msg)

        canvas = np.zeros_like(rgb)
        curr_time = time.time()
        frame_node = FrameNode(curr_time, rgb, depth)

        # OpenCV uses (n, 1, 2) for pixel shapes
        curr_pix = cv2.goodFeaturesToTrack(frame_node.gray, 
                                        mask = None, 
                                        **self.feature_params)
        if curr_pix is not None:
            curr_pix = curr_pix.reshape(-1, 2)
        else:
            curr_pix = np.zeros((1,2))
        
        # Project pixel coordinates to 3D
        curr_pos = self.proj_to_3d(depth, curr_pix)

        # Insert the first frame
        if self.feature_manager.frame_track.length == 0:
            self.feature_manager.frame_track.push(frame_node)
            feature_points = [FeaturePoint(-1,
                                           curr_time,
                                         pos,
                                         pix,
                                         np.zeros(128))
                              for pos, pix in zip(curr_pos, curr_pix)]
            self.feature_manager.push_feature_points(feature_points)

        else:
            prev_frame = self.feature_manager.frame_track.latest
            prev_gray = prev_frame.gray
            prev_track_ids, _, prev_pix = prev_frame.get_pts_info()

            # Optical Flow tracking
            # OpenCV uses (n, 1, 2) for pixel shapes
            forward_pix, status, _ = cv2.calcOpticalFlowPyrLK(
                                prev_gray,
                                frame_node.gray, 
                                prev_pix.reshape(-1,1,2).astype('float32'), 
                                None,
                                **self.lk_params)
            forward_pix = forward_pix.reshape(-1, 2)
            
            # In bound check
            status = status.reshape(-1)
            inbound = (forward_pix[:,0] <= depth.shape[1]) \
                    & (forward_pix[:,1] <= depth.shape[0]) \
                    & (forward_pix[:,0] >= 0) \
                    & (forward_pix[:,1] >= 0)
            status = status & inbound      

            self.get_logger().info("%d features matched with last frame"%(np.sum(status)))

            valid_forward_pix = forward_pix[status==1]
            curr_pix = self.trim_pts(curr_pix, valid_forward_pix)
            forward_pix = np.concatenate([valid_forward_pix, curr_pix], axis=0)

            ids = -np.ones(forward_pix.shape[0])

            # Number of points successfully tracked across two frames
            num_tracked = np.sum(status)
            ids[:num_tracked] = prev_track_ids[status==1]

            # Project pixel coordinates to 3D
            forward_pos = self.proj_to_3d(depth, forward_pix)
            
            # Push FrameNode to FeatureManager
            self.feature_manager.frame_track.push(frame_node)

            feature_points = [FeaturePoint(ids[i],
                                           curr_time,
                                         forward_pos[i],
                                         pix,
                                         np.zeros(128))
                              for i, pix in enumerate(forward_pix)]

            self.feature_manager.push_feature_points(feature_points)
            
            # Visualize tracks
            for track_id, track in self.feature_manager.feature_tracks.items():
                color = self.colors[track_id % self.num_color].tolist()
                node = track.latest
                while node is not None:
                    a, b = node.pix
                    canvas = cv2.circle(canvas, (int(a), int(b)), 5, color, -1)
                    node = node.prev

        result = cv2.add(rgb, canvas)
        cv2.imshow('Optical Flow (sparse)', result)

        self.get_logger().info('Frame processing took %.3f ms'%(1e3 * (time.time()-now)))
        
        cv2.waitKey(1)

    def compute_track_velocity(self):
        track_ids, _, _ = self.feature_manager.frame_track.latest.get_pts_info()
        valid_ids, locs, vecs = [], [], []
        for i in track_ids:
            track = self.feature_manager.feature_tracks[i]
            node = track.latest
            xyz, tt = [], []
            while node is not None:
                # Only extract valid depth value
                if node.pos[2] != 0:
                    xyz.append(node.pos)
                    tt.append(node.timestamp)
                node = node.prev
            if len(xyz) >= 3:
                valid_ids.append(i)
                locs.append(xyz[0])
                coeff = np.polyfit(np.array(tt), np.stack(xyz), 1)
                vecs.append(coeff[0])
        return valid_ids, locs, vecs
    
    def numpy_image_from_ros_image(self, ros_img, dtype, channels):
        np_img = np.frombuffer(ros_img.data, dtype=dtype)
        np_img = np_img.reshape(ros_img.height, ros_img.width, channels)
        return np_img

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
        colors = colors[:, [2,1,0]]
        pcd = np_pcd_to_colored_ros_pcd(points, colors, '/camera_link')
        return pcd


    def proj_to_3d(self, depth, pix):
        '''
        Project from 2D pixel locations to 3D points
        Args:
            camera_k -- (np.array) 3 x 3, intrinsic matrix
            depth -- (np.array) h x w, depth array
            pix -- (np.array) N x 2, pixel locations
        '''
        pix = np.floor(pix).astype('int')
        z = depth[pix[:,1], pix[:,0]]
        pix = np.insert(pix, 2, 1, axis=1)
        rays = (np.linalg.inv(self.camera_k) @ pix.T).T
        points = rays * z
        return points

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