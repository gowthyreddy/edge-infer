#pragma once

#include <functional>
#include <thread>
#include <vector>

#include "edge_infer/bounded_queue.h"

namespace edge_infer {

// Fixed number of worker threads pulling tasks from a BoundedQueue.
// The queue's policy decides what submit() does when the workers fall behind:
// Block makes the caller wait, DropOldest discards the oldest waiting task.
class ThreadPool {
   public:
    ThreadPool(size_t num_threads, size_t queue_capacity,
               OverflowPolicy policy = OverflowPolicy::Block);
    ~ThreadPool();  // calls shutdown()

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Returns false if the pool is shut down.
    bool submit(std::function<void()> task);

    // Runs the tasks still in the queue, then joins the workers. Safe to call more than once,
    // but only from one thread (the owner).
    void shutdown();

    size_t dropped() const { return tasks_.dropped(); }

   private:
    void worker_loop();

    BoundedQueue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
};

}  // namespace edge_infer
