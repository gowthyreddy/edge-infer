// End-to-end tests of the threaded pipeline on a short generated video.
// Skipped if the models or test images are not downloaded.

#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "edge_infer/pipeline.h"

using namespace edge_infer;
namespace fs = std::filesystem;

class PipelineTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const fs::path root = EDGE_INFER_ROOT;
        models_dir_ = (root / "models").string();
        fs::path image_path = root / "data" / "images" / "robot_karen-and-rob.png";
        if (!fs::exists(root / "models" / "version-RFB-320.onnx") || !fs::exists(image_path)) {
            GTEST_SKIP() << "models or images missing: run the download scripts and degrade.py";
        }
        // a short video: the same face image, shifted 1 px per frame
        video_path_ = (fs::temp_directory_path() / "edge_infer_test.avi").string();
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        cv::VideoWriter writer(video_path_, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                               image.size());
        ASSERT_TRUE(writer.isOpened());
        for (int i = 0; i < kFrames; i++) {
            cv::Mat frame;
            cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, i, 0, 1, 0);
            cv::warpAffine(image, frame, shift, image.size());
            writer.write(frame);
        }
    }

    PipelineConfig config() const {
        PipelineConfig c;
        c.video_path = video_path_;
        c.models_dir = models_dir_;
        c.fps = 0;  // as fast as possible: worst case for the queues
        return c;
    }

    static constexpr int kFrames = 20;
    std::string models_dir_;
    std::string video_path_;
};

TEST_F(PipelineTest, EveryFrameIsProcessedOrDropped) {
    for (OverflowPolicy policy : {OverflowPolicy::Block, OverflowPolicy::DropOldest}) {
        PipelineConfig c = config();
        c.policy = policy;
        Pipeline pipeline(c);
        int last_id = -1;
        PipelineStats stats = pipeline.run([&](const FrameResult& r) {
            EXPECT_GT(r.frame_id, last_id);  // results come out in frame order
            last_id = r.frame_id;
            EXPECT_GE(r.faces.size(), 1u);  // there is a face in every frame
        });
        EXPECT_EQ(stats.frames_read, kFrames);
        EXPECT_GT(stats.frames_processed, 0);
        EXPECT_EQ(stats.frames_processed + stats.frames_dropped, stats.frames_read);
        EXPECT_EQ(stats.latencies_ms.size(), static_cast<size_t>(stats.frames_processed));
    }
}

TEST_F(PipelineTest, CameraRateWithFastPipelineDropsNothing) {
    PipelineConfig c = config();
    c.fps = 4;  // slow enough that every frame is done before the next one arrives
    c.max_frames = 6;
    Pipeline pipeline(c);
    PipelineStats stats = pipeline.run();
    EXPECT_EQ(stats.frames_read, 6);
    EXPECT_EQ(stats.frames_processed, 6);
    EXPECT_EQ(stats.frames_dropped, 0);
}

TEST_F(PipelineTest, EarlyStopFromCallback) {
    PipelineConfig c = config();
    c.policy = OverflowPolicy::Block;  // no drops, so at least 3 results are guaranteed
    Pipeline pipeline(c);
    int seen = 0;
    PipelineStats stats = pipeline.run([&](const FrameResult&) {
        if (++seen == 3) pipeline.stop();
    });
    // run() returned (no hang) and everything in flight was drained
    EXPECT_GE(stats.frames_processed, 3);
    EXPECT_EQ(stats.frames_processed + stats.frames_dropped, stats.frames_read);
}

TEST_F(PipelineTest, ExceptionInCallbackStopsCleanly) {
    Pipeline pipeline(config());
    EXPECT_THROW(pipeline.run([](const FrameResult&) { throw std::runtime_error("consumer"); }),
                 std::runtime_error);
}

TEST(PipelineErrors, MissingVideoThrows) {
    const fs::path root = EDGE_INFER_ROOT;
    if (!fs::exists(root / "models" / "version-RFB-320.onnx")) GTEST_SKIP();
    PipelineConfig c;
    c.video_path = "does_not_exist.avi";
    c.models_dir = (root / "models").string();
    Pipeline pipeline(c);
    EXPECT_THROW(pipeline.run(), std::runtime_error);
}
