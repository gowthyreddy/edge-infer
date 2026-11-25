#include "edge_infer/pipeline.h"

#include <exception>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <thread>

#include "edge_infer/thread_pool.h"

namespace edge_infer {

namespace {

// A frame after preprocessing, waiting for inference.
struct PreparedFrame {
    int id = 0;
    Clock::time_point captured;
    cv::Mat image;              // original frame, needed to crop faces for the emotion model
    std::vector<float> tensor;  // detector input
};

double ms_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config),
      detector_(config.models_dir + "/version-RFB-320.onnx"),
      classifier_(config.models_dir + "/emotion-ferplus-8.onnx") {
    // We manage threads ourselves; stop OpenCV from starting its own pool inside resize etc.
    // (it would compete with our threads for the same cores).
    cv::setNumThreads(1);
}

PipelineStats Pipeline::run(const std::function<void(const FrameResult&)>& on_result) {
    // FFmpeg backend explicitly: with the default (CAP_ANY), a bad path makes OpenCV probe for
    // USB cameras through libgphoto2, which leaks memory (found with ASan).
    cv::VideoCapture video(config_.video_path, cv::CAP_FFMPEG);
    if (!video.isOpened()) throw std::runtime_error("cannot open video: " + config_.video_path);

    stop_requested_ = false;
    PipelineStats stats;
    const Clock::time_point start = Clock::now();

    // The policy applies to both queues in front of the slow stages. Inference is the slowest
    // stage, so stale frames must be dropped right in front of it: dropping only at the pool's
    // input let preprocessed frames pile up there and pushed p99 latency to ~0.8 s.
    // Results are never dropped; the output stage is fast.
    ThreadPool preprocess_pool(config_.preprocess_threads, config_.queue_capacity, config_.policy);
    BoundedQueue<PreparedFrame> prepared(config_.queue_capacity, config_.policy);
    BoundedQueue<FrameResult> results(config_.queue_capacity, OverflowPolicy::Block);
    int stale_skipped = 0;            // written only by the inference thread
    std::exception_ptr worker_error;  // written only by the inference thread

    // ---- decode thread: read frames at camera rate, hand each one to the pool ----
    std::thread decode_thread([&] {
        const auto period = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(config_.fps > 0 ? 1.0 / config_.fps : 0.0));
        Clock::time_point next_frame = Clock::now();
        int id = 0;
        while (!stop_requested_) {
            if (config_.max_frames > 0 && id >= config_.max_frames) break;
            cv::Mat image;  // new Mat each time, so the reader never overwrites a queued frame
            if (!video.read(image)) break;
            Clock::time_point captured = Clock::now();
            stats.frames_read++;

            // cv::Mat copies are cheap (reference counted), so the lambda can own the frame
            preprocess_pool.submit([&prepared, id, captured, image] {
                PreparedFrame p;
                p.id = id;
                p.captured = captured;
                p.image = image;
                p.tensor = preprocess_ultraface(image);
                prepared.push(std::move(p));  // false if inference has shut down; that's fine
            });
            id++;

            if (config_.fps > 0) {
                next_frame += period;
                std::this_thread::sleep_until(next_frame);
            }
        }
        // shutdown in order: finish queued preprocessing, then tell inference no more is coming
        preprocess_pool.shutdown();
        prepared.close();
    });

    // ---- inference thread: detector + emotion classifier ----
    std::thread inference_thread([&] {
        try {
            int last_id = -1;
            while (std::optional<PreparedFrame> p = prepared.pop()) {
                // the pool can finish frames out of order; a newer frame already went through
                if (p->id < last_id) {
                    stale_skipped++;
                    continue;
                }
                last_id = p->id;
                FrameResult r;
                r.frame_id = p->id;
                r.captured = p->captured;
                r.faces = detector_.detect_preprocessed(p->tensor, p->image.cols, p->image.rows);
                for (Face& f : r.faces) classifier_.classify(p->image, f);
                results.push(std::move(r));
            }
        } catch (...) {
            worker_error = std::current_exception();
            stop_requested_ = true;
            prepared.close();  // so pool workers blocked on push() can exit
        }
        results.close();
    });

    // ---- output: this thread ----
    std::exception_ptr output_error;
    try {
        while (std::optional<FrameResult> r = results.pop()) {
            r->latency_ms = ms_between(r->captured, Clock::now());
            stats.latencies_ms.push_back(r->latency_ms);
            stats.frames_processed++;
            if (on_result) on_result(*r);
        }
    } catch (...) {
        // on_result threw: stop everything, but still join the threads before rethrowing
        output_error = std::current_exception();
        stop_requested_ = true;
        results.close();
    }

    decode_thread.join();
    inference_thread.join();
    if (worker_error) std::rethrow_exception(worker_error);
    if (output_error) std::rethrow_exception(output_error);

    stats.frames_dropped =
        static_cast<int>(preprocess_pool.dropped() + prepared.dropped()) + stale_skipped;
    stats.seconds = ms_between(start, Clock::now()) / 1000.0;
    return stats;
}

}  // namespace edge_infer
