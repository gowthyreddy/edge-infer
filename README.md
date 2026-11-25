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

## Multithreaded pipeline

```
decode (1 thread) -> preprocess (thread pool) -> inference (1 thread) -> output (caller's thread)
```

- `BoundedQueue<T>` (mutex + two condition variables) connects the stages. When full it either
  blocks the producer (`Block`, backpressure) or drops the oldest item (`DropOldest`, latest frame wins).
- `ThreadPool`: fixed workers pulling `std::function` tasks from a `BoundedQueue`.
- The pool can finish frames out of order; inference skips any frame older than one it already
  processed, so results always come out in frame order.
- Shutdown runs in order: decode stops → pool drains and joins → inference drains → output drains.
  `stop()` can be called from any thread, including the result callback.

```bash
.venv/bin/python reference/make_video.py        # data/robot_camera.avi: 26 s at 30 fps, 640x480
./build/run_pipeline data/robot_camera.avi      # replay at 30 fps, drop-oldest
./build/run_pipeline data/robot_camera.avi --fps 0 --block   # as fast as possible, no drops
```

### First measurements

AMD Ryzen 7 8840HS (16 logical CPUs) under WSL2, ONNX Runtime 1.22.0, 1 intra-op thread, Release
build, the 780-frame robot-camera video replayed at 30 fps with drop-oldest. Single runs; proper
benchmarks (repeated runs, warm-up excluded) come later.

| Queue capacity | Preprocess threads | Processed / read | p50 latency | p99 latency |
|---|---|---|---|---|
| 2, dropping only at the pool's input | 2 | 252 / 300* | 33 ms | 789 ms |
| 2 | 1 | 584 / 780 | 76 ms | 258 ms |
| 2 | 2 | 577 / 780 | 80 ms | 261 ms |
| 1 | 1 | 569 / 780 | 49 ms | 222 ms |
| 1 (default) | 2 | 570 / 780 | 49 ms | 217 ms |

\* first 300 frames only.

What this showed:
- **Drop stale frames in front of the slowest stage.** The first version dropped frames only at
  the pool's input. Inference is the slowest stage, so already-preprocessed frames waited in the
  queue in front of it and p99 latency reached ~0.8 s. Applying the drop policy to that queue too
  cut p99 to ~0.26 s; a capacity of 1 cut it to ~0.22 s.
- **A second preprocessing thread did not help.** Preprocessing is not the bottleneck; inference is,
  especially on the 37-face class photo, where the emotion model runs once per face.
- The pipeline sustains ~22 processed frames/s against a 30 fps camera on this machine; the rest are
  dropped as stale, by design.

### Sanitizers

All tests run clean under ASan + UBSan and TSan, locally and in CI:

```bash
cmake -B build-address -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSANITIZE=address,undefined
cmake -B build-thread  -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSANITIZE=thread
setarch $(uname -m) -R ctest --test-dir build-thread   # TSan needs ASLR off on newer kernels
```

- ASan found a leak when opening a missing video: OpenCV's default backend probes for USB cameras
  through libgphoto2, which leaks. Fixed by opening videos with the FFmpeg backend explicitly.
- TSan reports inside Intel TBB (OpenCV's thread pool) and GDAL are false positives, since those
  libraries aren't built with TSan. `tests/tsan.supp` suppresses only those two libraries; our own
  code is fully checked.

## Models

[UltraFace RFB-320](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface)
and [FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) from the ONNX Model Zoo.
ONNX Runtime 1.22.0 (C++ from the NuGet package, Python from PyPI).
