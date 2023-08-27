import rclpy
from rclpy.node import Node
import numpy as np
import os, time
from sensor_msgs.msg import PointCloud2, PointField


DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}


FIELDS_XYZRGB = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
]

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
        
    return np_dtype_list


def pointcloud2_to_array(cloud_msg):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Reshapes the returned array to have shape (height, width), even if the height is 1.

    The reason for using np.frombuffer rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    return np.stack([cloud_arr['x'], cloud_arr['y'], cloud_arr['z']]).T



class WristCamPCDSub(Node):

    def __init__(self):
        super().__init__(f"wrist_cam_pcd_subscriber")
        
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.pcd_subscription = self.create_subscription(
            PointCloud2,
            "/filtered_wrist_pcd",
            self.pcd_callback,
            qos_profile=qos_profile,
        )

    def pcd_callback(self, pcd_msg):
        now = time.time()
        np_pcd = pointcloud2_to_array(pcd_msg)
        np.save('/home/jiacheng/tmp/pringles_no_table.npy', np_pcd)

        self.destroy_subscription(self.pcd_subscription)
        self.pcd_subscription = None
        self.get_logger().info('Point cloud saved')

def main(args=None):
    rclpy.init(args=args)
    wrist_cam_pcd_sub = WristCamPCDSub()
    rclpy.spin(wrist_cam_pcd_sub)
    wrist_cam_pcd_sub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
