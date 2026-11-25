#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace edge_infer {

// UltraFace (version-RFB-320) input size
constexpr int kDetWidth = 320;
constexpr int kDetHeight = 240;

// FER+ input size
constexpr int kEmotionSize = 64;
constexpr int kNumEmotions = 8;
extern const char* const kEmotionLabels[kNumEmotions];

struct Face {
    cv::Rect2f box;  // pixel coordinates in the original image
    float score = 0.0f;
    int emotion = -1;  // index into kEmotionLabels, -1 = not classified yet
    float emotion_prob = 0.0f;
};

// Helpers. Exposed in the header so they can be unit tested.
float iou(const cv::Rect2f& a, const cv::Rect2f& b);
std::vector<Face> nms(std::vector<Face> faces, float iou_threshold);
// BGR uint8 image -> 1x3x240x320 float, RGB, (pixel - 127) / 128
std::vector<float> preprocess_ultraface(const cv::Mat& bgr);
// grayscale uint8 face crop -> 1x1x64x64 float, raw 0..255 values
std::vector<float> preprocess_ferplus(const cv::Mat& gray);

class FaceDetector {
   public:
    explicit FaceDetector(const std::string& model_path, float score_threshold = 0.7f,
                          float nms_threshold = 0.3f);
    std::vector<Face> detect(const cv::Mat& bgr);
    // Same as detect(), for when preprocess_ultraface() already ran (e.g. on another thread).
    // image_cols/rows are the original image size, used to scale boxes back to pixels.
    std::vector<Face> detect_preprocessed(std::vector<float>& input, int image_cols,
                                          int image_rows);

   private:
    Ort::Session session_;
    std::string input_name_;
    float score_threshold_;
    float nms_threshold_;
};

class EmotionClassifier {
   public:
    explicit EmotionClassifier(const std::string& model_path);
    // fills face.emotion and face.emotion_prob
    void classify(const cv::Mat& bgr, Face& face);

   private:
    Ort::Session session_;
    std::string input_name_;
    std::string output_name_;
};

}  // namespace edge_infer
