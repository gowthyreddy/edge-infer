#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace edge_infer {

// What push() does when the queue is full.
enum class OverflowPolicy {
    Block,       // wait until there is space (backpressure on the producer)
    DropOldest,  // throw away the oldest item to make room (latest frame wins)
};

// Fixed-capacity FIFO queue shared between threads, built on one mutex and two condition
// variables. close() stops new pushes and wakes every waiting thread; pop() then keeps
// returning the remaining items and returns std::nullopt once the queue is empty.
template <typename T>
class BoundedQueue {
   public:
    explicit BoundedQueue(size_t capacity, OverflowPolicy policy = OverflowPolicy::Block)
        : capacity_(capacity), policy_(policy) {
        if (capacity == 0) throw std::invalid_argument("BoundedQueue capacity must be > 0");
    }

    // Not copyable or movable: other threads hold references to the queue.
    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    // Returns false if the queue is closed (the item is not added).
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (policy_ == OverflowPolicy::Block) {
            // wait() with a predicate loops internally: wakeups can be spurious, and another
            // producer may take the free slot before this thread gets the lock back
            not_full_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
        }
        if (closed_) return false;
        if (items_.size() >= capacity_) {  // only reachable with DropOldest
            items_.pop_front();
            dropped_++;
        }
        items_.push_back(std::move(item));
        lock.unlock();  // unlock first so the woken thread doesn't block on the mutex again
        not_empty_.notify_one();
        return true;
    }

    // Blocks until an item is available. Returns std::nullopt when closed and empty.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return item;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

    // How many items DropOldest has thrown away so far.
    size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

   private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> items_;
    const size_t capacity_;
    const OverflowPolicy policy_;
    bool closed_ = false;
    size_t dropped_ = 0;
};

}  // namespace edge_infer
