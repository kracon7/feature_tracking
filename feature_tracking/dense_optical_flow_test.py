import argparse
import cv2 as cv
import numpy as np

def main(args):
    cap = cv.VideoCapture(args.video)

    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        
        cv.imshow("input", frame)
        
        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        
        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        
        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", rgb)
        
        # Updates previous frame
        prev_gray = gray
        
        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str,
                        default="")

    args = parser.parse_args()
    main(args)