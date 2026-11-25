"""Makes a "robot camera" copy of every test image, so tests and benchmarks also cover
the low-quality input the robot actually sees.

The real camera's settings are not available, so these values are approximations:
640x480 frames, dim classroom lighting, sensor noise, slight blur, strong JPEG compression.
Everything is seeded, so the output is identical on every run.

Usage: python reference/degrade.py [images_dir]
Writes robot_<name>.png next to each original (PNG so the C++ side reads exactly these pixels).
"""

import os
import sys

import cv2
import numpy as np

CAM_W, CAM_H = 640, 480
BRIGHTNESS = 0.6  # dim lighting
NOISE_SIGMA = 6.0  # sensor noise, more visible in low light
BLUR_SIGMA = 0.8  # cheap lens / slight defocus
JPEG_QUALITY = 40
SEED = 1234


def crop_to_4_3(img):
    h, w = img.shape[:2]
    if w * 3 > h * 4:  # too wide
        new_w = h * 4 // 3
        x = (w - new_w) // 2
        return img[:, x:x + new_w]
    new_h = w * 3 // 4
    y = (h - new_h) // 2
    return img[y:y + new_h, :]


def degrade(bgr, rng):
    img = cv2.resize(crop_to_4_3(bgr), (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), BLUR_SIGMA)
    img = img.astype(np.float32) * BRIGHTNESS
    img += rng.normal(0.0, NOISE_SIGMA, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    images_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "images")
    count = 0
    for name in sorted(os.listdir(images_dir)):
        if name.startswith("robot_"):
            continue
        bgr = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rng = np.random.default_rng(SEED + count)  # one seed per image, independent of the others
        out_name = "robot_" + os.path.splitext(name)[0] + ".png"
        cv2.imwrite(os.path.join(images_dir, out_name), degrade(bgr, rng))
        count += 1
    print(f"wrote {count} degraded images to {images_dir}")


if __name__ == "__main__":
    main()
