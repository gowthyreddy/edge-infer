"""Builds a test video that stands in for the robot's camera: the robot-camera images
(robot_*.png from degrade.py), each shown for 1 second at 30 fps and drifting slowly
sideways so faces move a little between frames, like children in front of the robot.

Usage: python reference/make_video.py [images_dir] [out_file]
"""

import os
import sys

import cv2
import numpy as np

FPS = 30
SECONDS_PER_IMAGE = 1
DRIFT_PX_PER_FRAME = 1


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    images_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "images")
    out_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "data", "robot_camera.avi")

    names = sorted(n for n in os.listdir(images_dir) if n.startswith("robot_"))
    if not names:
        sys.exit("no robot_*.png images found; run reference/degrade.py first")

    # MJPG in AVI: OpenCV can write and read it without extra codecs
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (640, 480))
    frames = 0
    for name in names:
        img = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 480))
        for i in range(FPS * SECONDS_PER_IMAGE):
            shift = np.float32([[1, 0, i * DRIFT_PX_PER_FRAME - 15], [0, 1, 0]])
            frame = cv2.warpAffine(img, shift, (640, 480), borderMode=cv2.BORDER_REPLICATE)
            writer.write(frame)
            frames += 1
    writer.release()
    print(f"wrote {out_file}: {frames} frames, {frames / FPS:.0f} s at {FPS} fps")


if __name__ == "__main__":
    main()
