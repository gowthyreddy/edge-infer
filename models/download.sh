#!/usr/bin/env bash
# Downloads the public ONNX models from the ONNX Model Zoo. No model binaries are committed.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/onnx/models/raw/main/validated/vision/body_analysis
curl -L --fail -o version-RFB-320.onnx  "$BASE/ultraface/models/version-RFB-320.onnx"
curl -L --fail -o emotion-ferplus-8.onnx "$BASE/emotion_ferplus/model/emotion-ferplus-8.onnx"
ls -lh *.onnx
