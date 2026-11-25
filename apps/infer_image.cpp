// Single-threaded end-to-end run on one image: detect faces, classify each face's expression.
// Usage: infer_image <image> [models_dir]

#include <chrono>
#include <cstdio>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "edge_infer/models.h"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <image> [models_dir]\n", argv[0]);
        return 1;
    }
    std::string models_dir = argc > 2 ? argv[2] : "models";

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", argv[1]);
        return 1;
    }

    try {
        auto t0 = Clock::now();
        edge_infer::FaceDetector detector(models_dir + "/version-RFB-320.onnx");
        edge_infer::EmotionClassifier classifier(models_dir + "/emotion-ferplus-8.onnx");
        std::printf("models loaded in %.1f ms\n", ms_since(t0));

        auto t1 = Clock::now();
        std::vector<edge_infer::Face> faces = detector.detect(image);
        double detect_ms = ms_since(t1);

        auto t2 = Clock::now();
        for (edge_infer::Face& f : faces) classifier.classify(image, f);
        double classify_ms = ms_since(t2);

        std::printf("%zu face(s) in %dx%d image\n", faces.size(), image.cols, image.rows);
        for (const edge_infer::Face& f : faces) {
            std::printf("  box=(%.0f,%.0f,%.0f,%.0f) score=%.2f emotion=%s (%.2f)\n", f.box.x,
                        f.box.y, f.box.width, f.box.height, f.score,
                        f.emotion >= 0 ? edge_infer::kEmotionLabels[f.emotion] : "-",
                        f.emotion_prob);
        }
        std::printf("detect: %.2f ms, classify: %.2f ms\n", detect_ms, classify_ms);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
