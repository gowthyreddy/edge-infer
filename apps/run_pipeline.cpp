// Runs the multithreaded pipeline on a video file replayed at camera rate and prints a summary.
// Usage: run_pipeline <video> [--models DIR] [--fps 30] [--threads 2] [--queue 1]
//                     [--max-frames N] [--block] [--print]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "edge_infer/pipeline.h"

static double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(p / 100.0 * (values.size() - 1) + 0.5);
    return values[index];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <video> [--models DIR] [--fps 30] [--threads 2] [--queue 1] "
                     "[--max-frames N] [--block] [--print]\n",
                     argv[0]);
        return 1;
    }

    edge_infer::PipelineConfig config;
    config.video_path = argv[1];
    bool print_results = false;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        bool has_value = i + 1 < argc;
        if (arg == "--models" && has_value) {
            config.models_dir = argv[++i];
        } else if (arg == "--fps" && has_value) {
            config.fps = std::atof(argv[++i]);
        } else if (arg == "--threads" && has_value) {
            config.preprocess_threads = std::atoi(argv[++i]);
        } else if (arg == "--queue" && has_value) {
            config.queue_capacity = std::atoi(argv[++i]);
        } else if (arg == "--max-frames" && has_value) {
            config.max_frames = std::atoi(argv[++i]);
        } else if (arg == "--block") {
            config.policy = edge_infer::OverflowPolicy::Block;
        } else if (arg == "--print") {
            print_results = true;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (config.preprocess_threads < 1 || config.queue_capacity < 1) {
        std::fprintf(stderr, "--threads and --queue must be at least 1\n");
        return 1;
    }

    try {
        edge_infer::Pipeline pipeline(config);
        edge_infer::PipelineStats stats = pipeline.run([&](const edge_infer::FrameResult& r) {
            if (!print_results) return;
            std::printf("frame %d: %zu face(s), %.1f ms", r.frame_id, r.faces.size(), r.latency_ms);
            for (const edge_infer::Face& f : r.faces) {
                std::printf("  [%s %.2f]",
                            f.emotion >= 0 ? edge_infer::kEmotionLabels[f.emotion] : "-",
                            f.emotion_prob);
            }
            std::printf("\n");
        });

        std::printf("policy: %s, preprocess threads: %d, queue: %zu, replay fps: %.1f\n",
                    config.policy == edge_infer::OverflowPolicy::Block ? "block" : "drop-oldest",
                    config.preprocess_threads, config.queue_capacity, config.fps);
        std::printf("frames read: %d, processed: %d, dropped: %d\n", stats.frames_read,
                    stats.frames_processed, stats.frames_dropped);
        std::printf("throughput: %.1f processed frames/s over %.2f s\n",
                    stats.frames_processed / stats.seconds, stats.seconds);
        std::printf("latency (capture -> result): p50 %.1f ms, p99 %.1f ms\n",
                    percentile(stats.latencies_ms, 50), percentile(stats.latencies_ms, 99));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
