import open3d as o3d
import numpy as np

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=0.2, origin=[0,0,0])
pcd_path = '/home/jiacheng/tmp/pcd_1691618413.665.npy'

np_pcd = np.load(pcd_path)

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np_pcd[:,:3]))
pcd.colors = o3d.utility.Vector3dVector(np_pcd[:,3:].astype('float')/255)
o3d.visualization.draw_geometries([pcd, mesh_frame])