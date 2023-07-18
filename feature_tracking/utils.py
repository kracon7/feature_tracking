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
