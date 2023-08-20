import os
import cv2
import argparse
from PIL import Image
import numpy as np
from gmflow_utils.flow_viz import save_vis_flow_tofile
from gmflow_evaluate import MplColorHelper

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--depth_dir', '-d', default='tmp', type=str,
                        help='where to load depth images')
    parser.add_argument('--pose_dir', '-p', default='tmp', type=str,
                        help='where to load camera poses')
    parser.add_argument('--camera_k', '-k', default='tmp', type=str,
                        help='where to load camera intrinsics')
    parser.add_argument('--finger_mask', '-m', default='tmp', type=str,
                        help='where to load finger mask')
    parser.add_argument('--output_dir', '-o', default='tmp', type=str,
                        help='where to save flow images')
    parser.add_argument('--save_raw', action='store_true',
                        help='save the raw flow')
    
    return parser.parse_args()


def lift(depth, camera_k):
    im_h, im_w = depth.shape[:2]
    u, v = np.arange(im_w), np.arange(im_h)
    uu, vv = np.meshgrid(v, u, indexing='ij')
    pixels = np.stack([vv.reshape(-1), uu.reshape(-1), np.ones(im_h*im_w)])

    # project pixels
    rays = np.linalg.inv(camera_k) @ pixels  # 3 x (im_h x im_w)
    points = rays.T.reshape(im_h, im_w, 3) * depth  # im_h x im_w x 3

    # Reshape for output
    pixels = pixels.T
    return pixels[:,:2], points.reshape(im_h*im_w, 3)

def transform_points(points, T):
    n_pts = points.shape[0]
    homo_points = np.concatenate([points, np.ones((n_pts, 1))], axis=1)
    transformed = (T @ homo_points.T).T
    transformed = transformed[:,:3]
    return transformed

def project(points, camera_k):
    points_T = points.T
    homo_points = (points_T / points_T[2])
    pixels = (camera_k @ homo_points).T
    return pixels[:,:2]

def main():
    args = get_args_parser()
    depth_flist = os.listdir(args.depth_dir)
    pose_flist = os.listdir(args.pose_dir)
    depth_flist.sort()
    pose_flist.sort()
    camera_k = np.loadtxt(args.camera_k)
    finger_mask = np.asarray(Image.open(args.finger_mask))
    finger_mask = finger_mask[:,:,0] > 0

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    rad_colormap = MplColorHelper('twilight', -np.pi, np.pi)
    mag_colormap = MplColorHelper('gray', 0, 20)

    flow_mag_dir = os.path.join(args.output_dir, 'flow_mag_gt')
    flow_rad_dir = os.path.join(args.output_dir, 'flow_rad_gt')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(flow_mag_dir, exist_ok=True)
    os.makedirs(flow_rad_dir, exist_ok=True)
    if args.save_raw:
        raw_dir = os.path.join(args.output_dir, 'flow_raw_gt') 
        os.makedirs(raw_dir, exist_ok=True)

    num_frames = len(depth_flist)
    for i in range(num_frames-1):
        num = depth_flist[i].split('.')[0].split('_')[1]
        print(depth_flist[i], '  ', num)
        b_T_cam1 = np.loadtxt(os.path.join(args.pose_dir, pose_flist[i]))
        b_T_cam2 = np.loadtxt(os.path.join(args.pose_dir, pose_flist[i+1]))
        depth_1 = np.load(os.path.join(args.depth_dir, depth_flist[i])).astype('float')
        depth_1 = depth_1 / 1e3

        depth_mask = (depth_1 == 0).reshape(depth_1.shape[0], depth_1.shape[1])
        mask = depth_mask | finger_mask

        pixels_1, points_1 = lift(depth_1, camera_k)

        cam2_T_cam1 = np.linalg.inv(b_T_cam2) @ b_T_cam1
        points_2 = transform_points(points_1, cam2_T_cam1)
        pixels_2 = project(points_2, camera_k)

        flow = (pixels_2 - pixels_1).reshape(depth_1.shape[0], depth_1.shape[1], 2)
        flow[mask] = 0

        # # Flip the u v channel
        # flow = flow[:,:,[1,0]]

        flow_mag = np.linalg.norm(flow, axis=2)
        flow_rad = np.arctan2(flow[:,:,0], flow[:,:,1])
        print("rad: %.3f, %.3f mag: %.3f, %.3f"%(flow_rad.min(), flow_rad.max(), 
                                                 flow_mag.min(), flow_mag.max()))

        flow_mag_vis = (mag_colormap.get_rgb(flow_mag)[:,:,:3] * 255).astype('uint8')
        flow_rad_vis = (rad_colormap.get_rgb(flow_rad)[:,:,:3] * 255).astype('uint8')
        
        mag_output_path = os.path.join(flow_mag_dir, '%s.png'%num)
        rad_output_path = os.path.join(flow_rad_dir, '%s.png'%num)

        cv2.imwrite(mag_output_path, flow_mag_vis)
        cv2.imwrite(rad_output_path, flow_rad_vis)
        
if __name__ == "__main__":
    main()