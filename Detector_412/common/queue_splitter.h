/* 
本组件职责：
    将一个队列转换成两个相同的队列
*/
#pragma once
#include "thread_safe_queue.h"
// #include <optional>
#include <memory>
#include <thread>
#include <optional>

template <typename T>
class QueueSplitter{
public:
    QueueSplitter(std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> inputqueue,
    std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> outputqueue1,
    std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> outputqueue2);
    ~QueueSplitter();
    void Splitstep();
    void Split();
private:
    std::atomic<bool> should_exit_;  
    std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> inputqueue_;
    std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> outputqueue1_;
    std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> outputqueue2_;
    std::thread consume_thread_;

};

template <typename T>
QueueSplitter<T>::QueueSplitter(std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> inputqueue,
                std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> outputqueue1,
                std::shared_ptr<ThreadSafeQueue<std::shared_ptr<T>>> outputqueue2){
                    inputqueue_ = inputqueue;
                    outputqueue1_ = outputqueue1;
                    outputqueue2_ = outputqueue2;
                    should_exit_ = false;
                }

template <typename T>
QueueSplitter<T>::~QueueSplitter(){
    if(consume_thread_.joinable()){
        should_exit_ = true;
        consume_thread_.join();

    }
}
 

template <typename T>
void QueueSplitter<T>::Splitstep(){
    while(!should_exit_){
        std::optional<std::shared_ptr<T>> inputframe = inputqueue_->Dequeue();
        if (!inputframe) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));     // TODO: CHECK IF THIS INTERVAL IS APPROPRIATE
            continue;
        }
        
        auto start = std::chrono::high_resolution_clock::now();//计时

        if (!outputqueue1_ ->Enqueue(*inputframe))  {
            std::cout << "split queue1 full" << std::endl;
        }
        else {
            std::cout << "split queue1 enqueue success, new size:" << outputqueue1_->size() << std::endl;

        }
        if (!outputqueue2_ ->Enqueue(*inputframe))  {
            std::cout << "split queue2 full" << std::endl;
        }
        else {
            std::cout << "split queue2 enqueue success, new size:" << outputqueue2_->size() << std::endl;

        }

        // auto finish = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> time_span = finish - start;
        // LOG(INFO)<<"queue splitter time:"<< time_span.count();


    }
}

template <typename T>
void QueueSplitter<T>::Split(){
    assert(!consume_thread_.joinable());
    consume_thread_ = std::thread(&QueueSplitter<T>::Splitstep, this);
}
