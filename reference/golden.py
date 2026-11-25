"""Python reference for the C++ pipeline.

Runs the same models with the same pre/post-processing as src/models.cpp and writes
tests/golden/faces.txt, which the C++ golden test compares against.

Usage: python reference/golden.py [images_dir] [models_dir] [out_file]

Output format (plain text, easy to read from C++ with >>):
    image <name> <num_faces> <preprocess_sum>
    <x> <y> <w> <h> <score> <emotion_index> <emotion_prob>     (one line per face)
"""

import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)  # errors only; these older models trigger harmless warnings

DET_W, DET_H = 320, 240
EMO_SIZE = 64
SCORE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3


def make_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def preprocess_ultraface(bgr):
    img = cv2.resize(bgr, (DET_W, DET_H))  # bilinear, same as cv::resize default
    rgb = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    x = (rgb - 127.0) / 128.0
    return x.transpose(2, 0, 1)[None].copy()  # HWC -> 1xCxHxW


def preprocess_ferplus(gray):
    img = cv2.resize(gray, (EMO_SIZE, EMO_SIZE))
    return img.astype(np.float32)[None, None]  # 1x1x64x64, raw 0..255


def iou(a, b):
    # boxes are (x, y, w, h), same as cv::Rect2f
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = (x2 - x1) * (y2 - y1) if x2 > x1 and y2 > y1 else 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(faces, threshold):
    faces = sorted(faces, key=lambda f: -f["score"])
    kept = []
    for f in faces:
        if all(iou(f["box"], k["box"]) <= threshold for k in kept):
            kept.append(f)
    return kept


def detect(session, bgr):
    x = preprocess_ultraface(bgr)
    scores, boxes = session.run(["scores", "boxes"], {session.get_inputs()[0].name: x})
    rows, cols = bgr.shape[:2]
    faces = []
    for score, b in zip(scores[0, :, 1], boxes[0]):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1 = b[0] * np.float32(cols), b[1] * np.float32(rows)
        x2, y2 = b[2] * np.float32(cols), b[3] * np.float32(rows)
        box = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        faces.append({"box": box, "score": float(score)})
    return nms(faces, NMS_THRESHOLD), float(x.sum(dtype=np.float64))


def classify(session, bgr, face):
    # same as cv::Rect(face.box) & image rect: round each field, then clip
    x, y, w, h = (int(np.rint(v)) for v in face["box"])
    rows, cols = bgr.shape[:2]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, cols), min(y + h, rows)
    if x2 <= x1 or y2 <= y1:
        face["emotion"], face["prob"] = -1, 0.0
        return
    gray = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    logits = session.run(None, {session.get_inputs()[0].name: preprocess_ferplus(gray)})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    face["emotion"] = int(probs.argmax())
    face["prob"] = float(probs.max())


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    images_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "images")
    models_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "models")
    out_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, "tests", "golden", "faces.txt")

    detector = make_session(os.path.join(models_dir, "version-RFB-320.onnx"))
    classifier = make_session(os.path.join(models_dir, "emotion-ferplus-8.onnx"))

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    total = 0
    with open(out_file, "w", newline="\n") as out:
        for name in sorted(os.listdir(images_dir)):
            bgr = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"skipping unreadable file {name}")
                continue
            faces, pre_sum = detect(detector, bgr)
            for f in faces:
                classify(classifier, bgr, f)
            out.write(f"image {name} {len(faces)} {pre_sum:.4f}\n")
            for f in faces:
                x, y, w, h = f["box"]
                out.write(f"{x:.3f} {y:.3f} {w:.3f} {h:.3f} {f['score']:.6f} {f['emotion']} {f['prob']:.6f}\n")
            total += len(faces)
    print(f"wrote {out_file}: {total} faces")


if __name__ == "__main__":
    main()
