#include <gtest/gtest.h>

#include "edge_infer/models.h"

using namespace edge_infer;

TEST(Iou, IdenticalBoxesIsOne) {
    cv::Rect2f a(10, 10, 20, 20);
    EXPECT_FLOAT_EQ(iou(a, a), 1.0f);
}

TEST(Iou, DisjointBoxesIsZero) { EXPECT_FLOAT_EQ(iou({0, 0, 10, 10}, {50, 50, 10, 10}), 0.0f); }

TEST(Iou, HalfOverlap) {
    // overlap 5x10 = 50, union 100 + 100 - 50 = 150
    EXPECT_NEAR(iou({0, 0, 10, 10}, {5, 0, 10, 10}), 50.0f / 150.0f, 1e-6);
}

TEST(Nms, KeepsHighestScoreAndDropsOverlap) {
    std::vector<Face> faces(3);
    faces[0].box = {0, 0, 10, 10};
    faces[0].score = 0.8f;
    faces[1].box = {1, 1, 10, 10};  // overlaps faces[0] heavily
    faces[1].score = 0.9f;
    faces[2].box = {100, 100, 10, 10};  // far away
    faces[2].score = 0.75f;

    std::vector<Face> kept = nms(faces, 0.3f);
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
    EXPECT_FLOAT_EQ(kept[1].score, 0.75f);
}

TEST(Preprocess, UltrafaceChannelOrderAndNormalize) {
    // solid colour image: B=0, G=127, R=255
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 127, 255));
    std::vector<float> out = preprocess_ultraface(img);

    const int plane = kDetWidth * kDetHeight;
    ASSERT_EQ(out.size(), static_cast<size_t>(3 * plane));
    EXPECT_FLOAT_EQ(out[0], (255 - 127) / 128.0f);        // R plane first
    EXPECT_FLOAT_EQ(out[plane], 0.0f);                    // G plane
    EXPECT_FLOAT_EQ(out[2 * plane], (0 - 127) / 128.0f);  // B plane last
}

TEST(Preprocess, RejectsWrongImageType) {
    cv::Mat gray(10, 10, CV_8UC1);
    EXPECT_THROW(preprocess_ultraface(gray), std::invalid_argument);
    cv::Mat color(10, 10, CV_8UC3);
    EXPECT_THROW(preprocess_ferplus(color), std::invalid_argument);
    EXPECT_THROW(preprocess_ultraface(cv::Mat()), std::invalid_argument);
}

TEST(Preprocess, UltrafaceOutputSizeIsFixed) {
    // whatever the input size, the model always gets 3 x 240 x 320
    for (cv::Size size : {cv::Size(32, 32), cv::Size(640, 480), cv::Size(1920, 1080)}) {
        cv::Mat img(size, CV_8UC3, cv::Scalar(10, 20, 30));
        EXPECT_EQ(preprocess_ultraface(img).size(),
                  static_cast<size_t>(3 * kDetWidth * kDetHeight));
    }
}

TEST(Preprocess, UltrafaceNormalizeRange) {
    // 0 -> -127/128, 255 -> 128/128 = 1.0, for every value in every channel
    cv::Mat black(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat white(240, 320, CV_8UC3, cv::Scalar(255, 255, 255));
    for (float v : preprocess_ultraface(black)) ASSERT_FLOAT_EQ(v, -127.0f / 128.0f);
    for (float v : preprocess_ultraface(white)) ASSERT_FLOAT_EQ(v, 1.0f);
}

TEST(Preprocess, UltrafaceKeepsPixelPositions) {
    // left half red, right half blue: after resize the R plane must be bright on the left only
    cv::Mat img(240, 320, CV_8UC3, cv::Scalar(255, 0, 0));       // blue
    img(cv::Rect(0, 0, 160, 240)).setTo(cv::Scalar(0, 0, 255));  // red left half
    std::vector<float> out = preprocess_ultraface(img);
    int row = 100;
    EXPECT_FLOAT_EQ(out[row * kDetWidth + 10], 1.0f);               // R plane, left
    EXPECT_FLOAT_EQ(out[row * kDetWidth + 300], -127.0f / 128.0f);  // R plane, right
}

TEST(Preprocess, FerplusResizesAndKeepsRawValues) {
    cv::Mat face(100, 80, CV_8UC1, cv::Scalar(200));
    std::vector<float> out = preprocess_ferplus(face);
    ASSERT_EQ(out.size(), static_cast<size_t>(kEmotionSize * kEmotionSize));
    for (float v : out) ASSERT_FLOAT_EQ(v, 200.0f);  // FER+ wants raw 0..255, no scaling
}
