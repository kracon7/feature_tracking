from typing import List
from rclpy.node import Node
from rclpy.qos import QoSProfile
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np

FIELDS_XYZRGB = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
]

def np_pcd_to_colored_ros_pcd(
    points: np.ndarray, # shape (N, 3) dtype = float32
    colors: np.ndarray, # shape (N, 3) dtype = float32
    frame_id: str,
) -> PointCloud2:

    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    r = np.asarray(colors[:,0], dtype=np.uint32)
    g = np.asarray(colors[:,1], dtype=np.uint32)
    b = np.asarray(colors[:,2], dtype=np.uint32)
    int_rgb = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
    int_rgb.dtype = np.float32
    rgb_arr = int_rgb.reshape((int_rgb.shape[0], 1))

    points = points.astype(dtype)
    pc = np.hstack([points, rgb_arr])
    
    return PointCloud2(
            header=Header(frame_id=frame_id),
            height=1,
            width=pc.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=FIELDS_XYZRGB,
            point_step=(itemsize * len(FIELDS_XYZRGB)),
            row_step=(itemsize * len(FIELDS_XYZRGB) * pc.shape[0]),
            data=pc.tobytes(),
        )


def np_pcd_to_ros_pcd(pcd: np.ndarray, frame_id: str) -> np.ndarray:
    columns = 'xyz'
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    data = pcd.astype(dtype).tobytes()
    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(columns)]
    header = Header(frame_id=frame_id)
    return PointCloud2(
        header=header,
        height=1,
        width=pcd.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * len(columns)),
        row_step=(itemsize * len(columns) * pcd.shape[0]),
        data=data
    )



class MotionFieldVisualizer(Node):
    def __init__(self) -> None:
        super().__init__('MotionFieldVisualizer')
        qos_profile = QoSProfile(depth=10)
        self.mfield_publisher = self.create_publisher(
            MarkerArray, f"/mfield_markers", qos_profile
        )

        self.num_color = 100
        self.colors = np.random.rand(self.num_color, 3)

    def visualize_mfield(self, 
                         track_ids: List[int],
                         mfield_locs: List[np.ndarray],
                         mfield_vecs: List[np.ndarray],
                         frame_id: str = 'camera_link',
                        ):
        add_marker_array = MarkerArray()
        delete_marker_array = MarkerArray()

        for idx in range(len(track_ids)):
            loc = mfield_locs[idx]
            vec = mfield_vecs[idx]
            color = self.colors[idx%self.num_color]
            arrow_marker = get_arrow_marker(idx, frame_id, loc, vec, rgb=color)
            add_marker_array.markers.append(arrow_marker)

        deletion_marker = Marker()
        deletion_marker.header.frame_id = frame_id
        deletion_marker.action = Marker.DELETEALL
        delete_marker_array.markers.append(deletion_marker)
        
        self.mfield_publisher.publish(delete_marker_array)
        self.mfield_publisher.publish(add_marker_array)

    
def get_arrow_marker(
        marker_id: int,
        frame_id: str,
        loc: np.ndarray,
        vec: np.ndarray,
        rgb: List[float] = [200.0, 200.0, 200.0],
        alpha=1.0
    ):
        marker = Marker()
        marker.type = marker.ARROW
        marker.header.frame_id = frame_id
        marker.id = marker_id
        tail = Point(x=loc[0], 
                     y=loc[1], 
                     z=loc[2])
        tip = Point(x=loc[0]+vec[0], 
                    y=loc[1]+vec[1], 
                    z=loc[2]+vec[2])
        marker.points = [ tail, tip ]
        marker.scale.x = 0.005
        marker.scale.y = 0.01
        marker.scale.z = 0.02
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = alpha
        return marker
