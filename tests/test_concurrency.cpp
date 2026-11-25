// BoundedQueue and ThreadPool: behaviour and stress tests. Run these under TSan too.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "edge_infer/bounded_queue.h"
#include "edge_infer/thread_pool.h"

using namespace edge_infer;
using namespace std::chrono_literals;

// ---------------- BoundedQueue ----------------

TEST(BoundedQueue, FifoOrder) {
    BoundedQueue<int> q(3);
    q.push(1);
    q.push(2);
    q.push(3);
    EXPECT_EQ(*q.pop(), 1);
    EXPECT_EQ(*q.pop(), 2);
    EXPECT_EQ(*q.pop(), 3);
}

TEST(BoundedQueue, ZeroCapacityThrows) {
    EXPECT_THROW(BoundedQueue<int>(0), std::invalid_argument);
}

TEST(BoundedQueue, DropOldestKeepsNewest) {
    BoundedQueue<int> q(2, OverflowPolicy::DropOldest);
    for (int i = 1; i <= 5; i++) EXPECT_TRUE(q.push(i));  // never blocks
    EXPECT_EQ(q.dropped(), 3u);
    EXPECT_EQ(*q.pop(), 4);
    EXPECT_EQ(*q.pop(), 5);
}

TEST(BoundedQueue, CloseDrainsThenReturnsNullopt) {
    BoundedQueue<int> q(4);
    q.push(1);
    q.push(2);
    q.close();
    EXPECT_FALSE(q.push(3));  // no pushes after close
    EXPECT_EQ(*q.pop(), 1);
    EXPECT_EQ(*q.pop(), 2);
    EXPECT_FALSE(q.pop().has_value());
}

TEST(BoundedQueue, CloseWakesBlockedPop) {
    BoundedQueue<int> q(1);
    std::thread consumer([&] { EXPECT_FALSE(q.pop().has_value()); });
    std::this_thread::sleep_for(20ms);  // let the consumer block
    q.close();
    consumer.join();  // hangs here if close() doesn't wake it
}

TEST(BoundedQueue, CloseWakesBlockedPush) {
    BoundedQueue<int> q(1);
    q.push(1);  // now full
    std::thread producer([&] { EXPECT_FALSE(q.push(2)); });
    std::this_thread::sleep_for(20ms);
    q.close();
    producer.join();
}

TEST(BoundedQueue, BlockPolicyMakesProducerWait) {
    BoundedQueue<int> q(1);
    q.push(1);
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        q.push(2);
        pushed = true;
    });
    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(pushed);  // still waiting for space
    EXPECT_EQ(*q.pop(), 1);
    producer.join();
    EXPECT_TRUE(pushed);
    EXPECT_EQ(*q.pop(), 2);
}

TEST(BoundedQueue, StressManyProducersManyConsumers) {
    const int kProducers = 4, kConsumers = 4, kItemsEach = 20000;
    BoundedQueue<int> q(8);
    std::atomic<long long> sum{0};
    std::atomic<int> count{0};

    std::vector<std::thread> consumers;
    for (int c = 0; c < kConsumers; c++) {
        consumers.emplace_back([&] {
            while (std::optional<int> v = q.pop()) {
                sum += *v;
                count++;
            }
        });
    }
    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; p++) {
        producers.emplace_back([&] {
            for (int i = 1; i <= kItemsEach; i++) q.push(i);
        });
    }
    for (std::thread& t : producers) t.join();
    q.close();
    for (std::thread& t : consumers) t.join();

    // nothing lost, nothing duplicated
    EXPECT_EQ(count, kProducers * kItemsEach);
    EXPECT_EQ(sum, 1LL * kProducers * kItemsEach * (kItemsEach + 1) / 2);
}

TEST(BoundedQueue, StressSlowConsumerWithDropOldest) {
    // fast producer (camera), slow consumer (inference): the queue must drop, never block,
    // and the consumer must still see frames in increasing order
    const int kFrames = 2000;
    BoundedQueue<int> q(2, OverflowPolicy::DropOldest);
    std::vector<int> received;
    std::thread consumer([&] {
        while (std::optional<int> v = q.pop()) {
            received.push_back(*v);
            std::this_thread::sleep_for(100us);
        }
    });
    for (int i = 0; i < kFrames; i++) q.push(i);
    q.close();
    consumer.join();

    EXPECT_EQ(received.size() + q.dropped(), static_cast<size_t>(kFrames));
    EXPECT_GT(q.dropped(), 0u);
    for (size_t i = 1; i < received.size(); i++) ASSERT_LT(received[i - 1], received[i]);
    EXPECT_EQ(received.back(), kFrames - 1);  // the newest frame always survives
}

// ---------------- ThreadPool ----------------

TEST(ThreadPool, RunsEveryTask) {
    std::atomic<int> done{0};
    ThreadPool pool(4, 16);
    for (int i = 0; i < 10000; i++) {
        ASSERT_TRUE(pool.submit([&] { done++; }));
    }
    pool.shutdown();  // runs everything still queued
    EXPECT_EQ(done, 10000);
}

TEST(ThreadPool, SubmitAfterShutdownFails) {
    ThreadPool pool(2, 4);
    pool.shutdown();
    pool.shutdown();  // second call is harmless
    EXPECT_FALSE(pool.submit([] {}));
}

TEST(ThreadPool, ThrowingTaskDoesNotKillWorker) {
    std::atomic<int> done{0};
    ThreadPool pool(1, 4);  // one worker: if it died, the second task would never run
    pool.submit([] { throw std::runtime_error("boom"); });
    pool.submit([&] { done++; });
    pool.shutdown();
    EXPECT_EQ(done, 1);
}

TEST(ThreadPool, DropOldestDropsWhenWorkersAreBusy) {
    std::atomic<int> done{0};
    ThreadPool pool(1, 2, OverflowPolicy::DropOldest);
    for (int i = 0; i < 200; i++) {
        pool.submit([&] {
            std::this_thread::sleep_for(200us);
            done++;
        });
    }
    pool.shutdown();
    EXPECT_GT(pool.dropped(), 0u);
    EXPECT_EQ(done + static_cast<int>(pool.dropped()), 200);
}

TEST(ThreadPool, DestructorJoinsWorkers) {
    std::atomic<int> done{0};
    {
        ThreadPool pool(3, 8);
        for (int i = 0; i < 100; i++) pool.submit([&] { done++; });
    }  // no shutdown() call: the destructor must finish the work and join
    EXPECT_EQ(done, 100);
}
