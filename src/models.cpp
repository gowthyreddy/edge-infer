#include "edge_infer/models.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace edge_infer {

const char* const kEmotionLabels[kNumEmotions] = {"neutral", "happiness", "surprise", "sadness",
                                                  "anger",   "disgust",   "fear",     "contempt"};

// ---------------- ONNX Runtime setup ----------------

// ONNX Runtime wants a single Env per process.
static Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge_infer");
    return env;
}

static Ort::SessionOptions session_options() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);  // single-threaded for now; tuned from benchmarks later
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

static std::string input_name(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions alloc;
    return session.GetInputNameAllocated(0, alloc).get();
}

static std::string output_name(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions alloc;
    return session.GetOutputNameAllocated(0, alloc).get();
}

static Ort::Value make_tensor(std::vector<float>& data, const std::array<int64_t, 4>& shape) {
    static Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(),
                                           shape.size());
}

// ---------------- helpers ----------------

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// ponytail: O(n^2), fine for the few dozen boxes left after score thresholding
std::vector<Face> nms(std::vector<Face> faces, float iou_threshold) {
    std::sort(faces.begin(), faces.end(),
              [](const Face& a, const Face& b) { return a.score > b.score; });
    std::vector<Face> kept;
    for (const Face& f : faces) {
        bool overlaps = false;
        for (const Face& k : kept) {
            if (iou(f.box, k.box) > iou_threshold) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) kept.push_back(f);
    }
    return kept;
}

std::vector<float> preprocess_ultraface(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("preprocess_ultraface: expected a non-empty BGR uint8 image");
    }
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(kDetWidth, kDetHeight));

    const int plane = kDetWidth * kDetHeight;
    std::vector<float> out(3 * plane);
    for (int y = 0; y < kDetHeight; y++) {
        for (int x = 0; x < kDetWidth; x++) {
            const cv::Vec3b& px = resized.at<cv::Vec3b>(y, x);
            int i = y * kDetWidth + x;
            // OpenCV stores BGR, the model wants RGB planes (CHW)
            out[0 * plane + i] = (px[2] - 127.0f) / 128.0f;
            out[1 * plane + i] = (px[1] - 127.0f) / 128.0f;
            out[2 * plane + i] = (px[0] - 127.0f) / 128.0f;
        }
    }
    return out;
}

std::vector<float> preprocess_ferplus(const cv::Mat& gray) {
    if (gray.empty() || gray.type() != CV_8UC1) {
        throw std::invalid_argument(
            "preprocess_ferplus: expected a non-empty grayscale uint8 image");
    }
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(kEmotionSize, kEmotionSize));
    std::vector<float> out(kEmotionSize * kEmotionSize);
    for (int y = 0; y < kEmotionSize; y++) {
        for (int x = 0; x < kEmotionSize; x++) {
            out[y * kEmotionSize + x] = resized.at<uchar>(y, x);
        }
    }
    return out;
}

// ---------------- FaceDetector ----------------

FaceDetector::FaceDetector(const std::string& model_path, float score_threshold,
                           float nms_threshold)
    : session_(ort_env(), model_path.c_str(), session_options()),
      input_name_(input_name(session_)),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold) {}

std::vector<Face> FaceDetector::detect(const cv::Mat& bgr) {
    std::vector<float> input = preprocess_ultraface(bgr);
    Ort::Value tensor = make_tensor(input, {1, 3, kDetHeight, kDetWidth});

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {"scores", "boxes"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 2);

    // scores: [1, N, 2] (background, face), boxes: [1, N, 4] normalized x1 y1 x2 y2
    const float* scores = outputs[0].GetTensorData<float>();
    const float* boxes = outputs[1].GetTensorData<float>();
    int64_t num_anchors = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];

    std::vector<Face> faces;
    for (int64_t i = 0; i < num_anchors; i++) {
        float score = scores[i * 2 + 1];
        if (score < score_threshold_) continue;
        Face f;
        f.score = score;
        f.box = cv::Rect2f(cv::Point2f(boxes[i * 4 + 0] * bgr.cols, boxes[i * 4 + 1] * bgr.rows),
                           cv::Point2f(boxes[i * 4 + 2] * bgr.cols, boxes[i * 4 + 3] * bgr.rows));
        faces.push_back(f);
    }
    return nms(std::move(faces), nms_threshold_);
}

// ---------------- EmotionClassifier ----------------

EmotionClassifier::EmotionClassifier(const std::string& model_path)
    : session_(ort_env(), model_path.c_str(), session_options()),
      input_name_(input_name(session_)),
      output_name_(output_name(session_)) {}

void EmotionClassifier::classify(const cv::Mat& bgr, Face& face) {
    cv::Rect roi = cv::Rect(face.box) & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.empty()) return;

    cv::Mat gray;
    cv::cvtColor(bgr(roi), gray, cv::COLOR_BGR2GRAY);
    std::vector<float> input = preprocess_ferplus(gray);
    Ort::Value tensor = make_tensor(input, {1, 1, kEmotionSize, kEmotionSize});

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
    const float* logits = outputs[0].GetTensorData<float>();

    // softmax over the 8 logits, then take the best one
    float max_logit = *std::max_element(logits, logits + kNumEmotions);
    float sum = 0.0f;
    std::array<float, kNumEmotions> probs{};
    for (int i = 0; i < kNumEmotions; i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    int best = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
    face.emotion = best;
    face.emotion_prob = probs[best] / sum;
}

}  // namespace edge_infer
