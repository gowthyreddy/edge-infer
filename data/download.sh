#!/usr/bin/env bash
# Downloads the test images from OpenCV's public test-data repo (opencv_extra),
# pinned to one commit so the golden outputs stay reproducible. No images are committed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p images
BASE=https://raw.githubusercontent.com/opencv/opencv_extra/94e6473e1344d99855678ea531aedaa49eea16ff/testdata

FILES=(
    cv/cascadeandhog/images/addams-family.png
    cv/cascadeandhog/images/audrybt1.png
    cv/cascadeandhog/images/bttf301.png
    cv/cascadeandhog/images/churchill-downs.png
    cv/cascadeandhog/images/class57.png
    cv/cascadeandhog/images/er.png
    cv/cascadeandhog/images/image_00000000_0.png
    cv/cascadeandhog/images/karen-and-rob.png
    cv/cascadeandhog/images/larroquette.png
    cv/cascadeandhog/images/mickymouse-self-p.png
    cv/cascadeandhog/images/mona-lisa.png
    cv/cascadeandhog/images/rehg-thanksgiving-1994.png
    cv/cascadeandhog/images/waynesworld2.png
    cv/face/100032540_1.jpg
    cv/face/100040721_1.jpg
    cv/face/david1.jpg
    cv/face/david2.jpg
    cv/face/detect.jpg
    cv/dnn_face/recognition/Aaron_Tippin_0001.jpg
    cv/dnn_face/recognition/Alvaro_Uribe_0028.jpg
    cv/dnn_face/recognition/Alvaro_Uribe_0029.jpg
    cv/dnn_face/recognition/Jose_Luis_Rodriguez_Zapatero_0001.jpg
    cv/shared/lena.png
    cv/shared/baboon.png
    dnn/grace_hopper_227.png
    dnn/gray_face.png
)

for f in "${FILES[@]}"; do
    name=$(basename "$f")
    [ -f "images/$name" ] || curl -sL --fail -o "images/$name" "$BASE/$f"
done
echo "$(ls images | wc -l) images in data/images"
