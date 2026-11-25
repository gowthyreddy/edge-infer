# edge-infer

C++20 face + expression perception service for a low-cost classroom robot. Work in progress (week 1).

## Setup (Ubuntu 24.04 / WSL)

```bash
sudo apt install -y cmake ninja-build libopencv-dev libgtest-dev clang-format clang-tidy
./models/download.sh
cmake -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
./build/infer_image path/to/face.jpg models
```

Models: [UltraFace RFB-320](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface)
and [FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) from the ONNX Model Zoo.
