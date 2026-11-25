#include "edge_infer/thread_pool.h"

#include <cstdio>
#include <exception>

namespace edge_infer {

ThreadPool::ThreadPool(size_t num_threads, size_t queue_capacity, OverflowPolicy policy)
    : tasks_(queue_capacity, policy) {
    for (size_t i = 0; i < num_threads; i++) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() { shutdown(); }

bool ThreadPool::submit(std::function<void()> task) { return tasks_.push(std::move(task)); }

void ThreadPool::shutdown() {
    tasks_.close();  // workers finish what is queued, then pop() returns nullopt
    for (std::thread& t : workers_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::worker_loop() {
    while (std::optional<std::function<void()>> task = tasks_.pop()) {
        try {
            (*task)();
        } catch (const std::exception& e) {
            // an exception leaving a thread calls std::terminate, so report it and keep going
            std::fprintf(stderr, "ThreadPool: task threw: %s\n", e.what());
        }
    }
}

}  // namespace edge_infer
