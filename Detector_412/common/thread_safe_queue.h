#pragma once
#include <optional>
#include <queue>
#include <mutex>

template<typename T>
class ThreadSafeQueue {

public:
    ThreadSafeQueue(size_t capacity) {
        capacity_ = capacity;
    };
    virtual ~ThreadSafeQueue() = default;

    // 避免构造拷贝
    ThreadSafeQueue(const ThreadSafeQueue<T>&) = delete;
    // 避免赋值拷贝
    ThreadSafeQueue& operator=(const ThreadSafeQueue<T>&) = delete;

    // 可移动

    ThreadSafeQueue(ThreadSafeQueue<T>&& other) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_ = std::move(other.queue_);
    }

    unsigned long size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    std::optional<T> Dequeue() {    // std::optional need c++17
        std::lock_guard<std::mutex> lock(mutex_);

        // condition_variable make the Dequeue blocked when the queue is empty, which may block the called loop, and cannot quit gracefully
        // When there is no data, wait till someone fills it. Lock is automatically released in the wait and obtained again after the wait 
        // while (queue_.size()==0)
        //     cond_.wait(lock);

        if (queue_.empty()) {
            return {};
        }
        T front = queue_.front();
        queue_.pop();
        return front;
    }

    bool Enqueue(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (queue_.size() < capacity_) {
            queue_.push(item);
            // cond_.notify_one();
            return true;
        }
        else {
            return false;
        }
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    size_t capacity_;
    // std::condition_variable cond_;
};