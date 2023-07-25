from std_msgs.msg import Header
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

class FeaturePoint:
    def __init__(self, track_id: int, timestamp: float):
        self.track_id = track_id
        self.timestamp = timestamp
        self.next = None
        self.prev = None

    def __repr__(self):
        return "%d %.2f"%(self.track_id, self.timestamp)

class FeatureTrack:
    def __init__(self):
        self.latest = None
        self.oldest = None
        self.length = 0

    # Adding a new node to the front		
    def push(self, new_node: FeaturePoint):
        self.length += 1

        if self.latest is None:
            assert self.oldest is None, "Cannot push node. "\
                "Head and Tail should be None at the same time!"
            self.latest = new_node
            self.oldest = new_node
            return
        
        new_node.prev = self.latest
        self.latest.next = new_node
        self.latest = new_node

    def delete(self, node: FeaturePoint):
        if self.length <= 1:
            self.clear()
            return
        
        if node.next is None:
            self.delete_latest()
            return
        
        if node.prev is None:
            self.delete_oldest()
            return

        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
        self.length -= 1

    def delete_latest(self):
        if self.length <= 1:
            self.clear()
        else:
            self.latest = self.latest.prev
            self.latest.next = None
            self.length -= 1

    def delete_oldest(self):
        if self.length <= 1:
            self.clear()
        else:
            self.oldest = self.oldest.next
            self.oldest.prev = None
            self.length -= 1
    
    def clear(self):
        self.latest = None
        self.oldest = None
        self.length = 0

    def __repr__(self):
        node = self.latest
        nodes = []
        while node is not None:
            nodes.append(repr(node))
            node = node.prev
        nodes.append("None")
        return " -> ".join(nodes)

class FrameOfFeatures:
    '''
    All features in a certain image frame
    '''
    def __init__(self, timestamp: float):
        self.timestamp = timestamp
        self.pts = {}

    def add_pt(self, point_feature: FeaturePoint):
        if point_feature.timestamp != self.timestamp:
            raise Exception("Mismatch between FeatureFrame time and FeaturePoint")
        
        self.pts[point_feature.track_id] = point_feature

class FeatureManager:
    def __init__(self) -> None:
        self.feature_tracks = {}

    def add_frame(self, new_frame: FrameOfFeatures):
        