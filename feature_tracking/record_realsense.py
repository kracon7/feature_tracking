import os
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse

def record_video(args, rs_pipeline):
    color_path = os.path.join(args.output_dir, 'rs_rgb.avi')
    depth_path = os.path.join(args.output_dir, 'rs_depth.npy')
    colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (args.width, args.height), 1)
    depth_frames = []
    
    try:
        while True:
            frames = rs_pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            #convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_frames.append(depth_image)
            
            colorwriter.write(color_image)
            
            cv2.imshow('Stream', color_image)
            
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        colorwriter.release()
        np.save(depth_path, np.stack(depth_frames))


def main(args):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    pipeline.start(config)

    record_video(args, pipeline)
    pipeline.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output_dir", "-o", type=str,
                        default="resource")
    args = parser.parse_args()

    main(args)