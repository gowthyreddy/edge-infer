// Compares the C++ pipeline with the Python reference outputs in tests/golden/faces.txt
// (written by reference/golden.py). Skips if the models or test images are not downloaded.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "edge_infer/models.h"

using namespace edge_infer;
namespace fs = std::filesystem;

// tolerances (C++ and Python use the same ONNX Runtime and OpenCV versions, so these are tight)
constexpr float kBoxTolPx = 1.0f;
constexpr float kScoreTol = 1e-3f;
constexpr float kProbTol = 1e-2f;
constexpr double kPreSumRelTol = 1e-4;

struct GoldenImage {
    std::string name;
    double preprocess_sum = 0.0;
    std::vector<Face> faces;
};

static std::vector<GoldenImage> load_golden(const std::string& path) {
    std::vector<GoldenImage> images;
    std::ifstream in(path);
    std::string tag;
    while (in >> tag) {
        GoldenImage g;
        int num_faces = 0;
        in >> g.name >> num_faces >> g.preprocess_sum;
        for (int i = 0; i < num_faces; i++) {
            Face f;
            in >> f.box.x >> f.box.y >> f.box.width >> f.box.height >> f.score >> f.emotion >>
                f.emotion_prob;
            g.faces.push_back(f);
        }
        images.push_back(g);
    }
    return images;
}

TEST(Golden, MatchesPythonReference) {
    const fs::path root = EDGE_INFER_ROOT;
    const fs::path images_dir = root / "data" / "images";
    const fs::path models_dir = root / "models";
    if (!fs::exists(models_dir / "version-RFB-320.onnx") || !fs::exists(images_dir)) {
        GTEST_SKIP() << "models or images missing: run models/download.sh and data/download.sh";
    }

    std::vector<GoldenImage> golden =
        load_golden((root / "tests" / "golden" / "faces.txt").string());
    ASSERT_GE(golden.size(), 20u) << "golden file missing or too small";

    FaceDetector detector((models_dir / "version-RFB-320.onnx").string());
    EmotionClassifier classifier((models_dir / "emotion-ferplus-8.onnx").string());

    int total_faces = 0;
    for (const GoldenImage& g : golden) {
        SCOPED_TRACE(g.name);
        cv::Mat image = cv::imread((images_dir / g.name).string(), cv::IMREAD_COLOR);
        ASSERT_FALSE(image.empty());

        // same input tensor as Python?
        std::vector<float> pre = preprocess_ultraface(image);
        double pre_sum = std::accumulate(pre.begin(), pre.end(), 0.0);
        EXPECT_NEAR(pre_sum, g.preprocess_sum, std::abs(g.preprocess_sum) * kPreSumRelTol + 1e-2);

        std::vector<Face> faces = detector.detect(image);
        for (Face& f : faces) classifier.classify(image, f);
        ASSERT_EQ(faces.size(), g.faces.size());

        // match each golden face to the C++ face that overlaps it most (order can differ on ties)
        for (const Face& expected : g.faces) {
            const Face* best = nullptr;
            float best_iou = 0.0f;
            for (const Face& f : faces) {
                if (iou(f.box, expected.box) > best_iou) {
                    best_iou = iou(f.box, expected.box);
                    best = &f;
                }
            }
            ASSERT_NE(best, nullptr);
            EXPECT_NEAR(best->box.x, expected.box.x, kBoxTolPx);
            EXPECT_NEAR(best->box.y, expected.box.y, kBoxTolPx);
            EXPECT_NEAR(best->box.width, expected.box.width, kBoxTolPx);
            EXPECT_NEAR(best->box.height, expected.box.height, kBoxTolPx);
            EXPECT_NEAR(best->score, expected.score, kScoreTol);
            EXPECT_EQ(best->emotion, expected.emotion);
            EXPECT_NEAR(best->emotion_prob, expected.emotion_prob, kProbTol);
        }
        total_faces += static_cast<int>(faces.size());
    }
    std::printf("compared %zu images, %d faces\n", golden.size(), total_faces);
}
