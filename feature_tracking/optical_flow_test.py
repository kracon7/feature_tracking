import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
 
class ImageSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_subscriber')

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
                                        Image, 
                                        '/camera/color/image_rect_raw', 
                                        self.listener_callback, 
                                        10)
        self.subscription # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        self.frame_count = 1

        self.parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)
        self.parameter_lucas_kanade = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def listener_callback(self, msg):

        # Convert ROS Image message to OpenCV image
        frame = self.br.imgmsg_to_cv2(msg, 'bgr8')

        if self.frame_count == 1:
            # convert to grayscale
            self.frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use Shi-Tomasi to detect object corners / edges from initial frame
            self.edges = cv2.goodFeaturesToTrack(self.frame_gray_init, mask = None, **self.parameters_shitomasi)
            # create a black canvas the size of the initial frame
            self.canvas = np.zeros_like(frame)
            # create random colours for visualization for all 100 max corners for RGB channels
            self.colours = np.random.randint(0, 255, (100, 3))

        self.frame_count += 1

        # Display the message on the console
        self.get_logger().info('Receiving video frame %d'%self.frame_count)

        # prepare grayscale image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # update object corners by comparing with found edges in initial frame
        update_edges, status, errors = cv2.calcOpticalFlowPyrLK(self.frame_gray_init, frame_gray, self.edges, None,
                                                            **self.parameter_lucas_kanade)
        # only update edges if algorithm successfully tracked
        self.new_edges = update_edges[status == 1]
        # to calculate directional flow we need to compare with previous position
        self.old_edges = self.edges[status == 1]

        for i, (new, old) in enumerate(zip(self.new_edges, self.old_edges)):
            a, b = new.ravel()
            c, d = old.ravel()

            # draw line between old and new corner point with random colour
            mask = cv2.line(self.canvas, (int(a), int(b)), (int(c), int(d)), self.colours[i].tolist(), 2)
            # draw circle around new position
            frame = cv2.circle(frame, (int(a), int(b)), 5, self.colours[i].tolist(), -1)

        result = cv2.add(frame, mask)
        cv2.imshow('Optical Flow (sparse)', result)

        # overwrite initial frame with current before restarting the loop
        self.frame_gray_init = frame_gray.copy()
        # update to new edges before restarting the loop
        self.edges = self.new_edges.reshape(-1, 1, 2)

        # # Display image
        # cv2.imshow("camera", current_frame)

        cv2.waitKey(1)

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