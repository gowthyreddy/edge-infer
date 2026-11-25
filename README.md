# edge-infer

C++20 face + expression perception service for a low-cost classroom robot. Work in progress.

## Setup (Ubuntu 24.04 / WSL)

```bash
sudo apt install -y cmake ninja-build libopencv-dev libgtest-dev clang-format clang-tidy python3-venv
./models/download.sh
./data/download.sh

python3 -m venv .venv && .venv/bin/pip install -r reference/requirements.txt
.venv/bin/python reference/degrade.py    # robot-camera copies of the test images

cmake -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
./build/infer_image path/to/face.jpg models
```

## Correctness: golden outputs

`reference/golden.py` runs the same models with the same pre/post-processing in Python and writes
`tests/golden/faces.txt`. The C++ test `Golden.MatchesPythonReference` checks every image against it:
same face count, boxes within 1 px, scores within 1e-3, same emotion label, probability within 1e-2,
and the same preprocessed input tensor (sum within 1e-4 relative).

Test images: 26 public images from OpenCV's test data (`opencv_extra`, pinned commit), from single
portraits to a class photo, plus one image with no face. Each also gets a **robot-camera copy**
(`reference/degrade.py`): 640x480, 0.6x brightness (dim classroom), Gaussian noise sigma 6, blur
sigma 0.8, JPEG quality 40. These settings approximate a cheap robot camera; they are not measured
from real hardware. 52 images, 163 faces in total.

Regenerate the golden file only when the pipeline is meant to change:
`.venv/bin/python reference/golden.py`

## Models

[UltraFace RFB-320](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface)
and [FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) from the ONNX Model Zoo.
ONNX Runtime 1.22.0 (C++ from the NuGet package, Python from PyPI).
