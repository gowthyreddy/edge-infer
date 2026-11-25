#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#include "edge_infer/bounded_queue.h"
#include "edge_infer/models.h"

namespace edge_infer {

using Clock = std::chrono::steady_clock;

struct PipelineConfig {
    std::string video_path;
    std::string models_dir = "models";
    double fps = 30.0;           // replay rate, like a camera; 0 = as fast as possible
    int preprocess_threads = 2;  // size of the preprocessing thread pool
    size_t queue_capacity =
        1;  // per queue; 1 gave the lowest p99 latency (stale frames never wait)
    OverflowPolicy policy = OverflowPolicy::DropOldest;  // when preprocessing falls behind
    int max_frames = 0;                                  // 0 = whole video
};

struct FrameResult {
    int frame_id = 0;
    Clock::time_point captured;  // when the frame was read from the video
    std::vector<Face> faces;
    double latency_ms = 0.0;  // capture -> result delivered
};

struct PipelineStats {
    int frames_read = 0;
    int frames_processed = 0;
    int frames_dropped = 0;  // dropped by the queue policy or skipped as stale
    double seconds = 0.0;
    std::vector<double> latencies_ms;  // one per processed frame
};

// decode -> preprocess (thread pool) -> inference -> output, each on its own thread(s),
// connected by bounded queues. Frames that are older than one already processed are skipped,
// so results always come out in frame order and latency stays bounded.
class Pipeline {
   public:
    explicit Pipeline(const PipelineConfig& config);

    // Runs until the video ends or stop() is called. on_result runs on the output thread
    // (the thread that called run()), once per processed frame, in frame order.
    PipelineStats run(const std::function<void(const FrameResult&)>& on_result = nullptr);

    // Thread-safe; can be called from on_result or from another thread.
    void stop() { stop_requested_ = true; }

   private:
    PipelineConfig config_;
    FaceDetector detector_;
    EmotionClassifier classifier_;
    std::atomic<bool> stop_requested_{false};
};

}  // namespace edge_infer
