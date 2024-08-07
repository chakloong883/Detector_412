#pragma once
#include "common_frame.h"
#include "thread_safe_queue.h"
#include "common_frame_inside.h"

typedef std::shared_ptr<ImageFrame> ImageFramePtr;
typedef std::shared_ptr<ResultFrame> ResultFramePtr;
typedef std::shared_ptr<BatchImageFrame> BatchImageFramePtr;
typedef std::shared_ptr<BatchResultFrame> BatchResultFramePtr;




typedef ThreadSafeQueue<ImageFramePtr> ImageFrameQueue;
typedef ThreadSafeQueue<ResultFramePtr> ResultFrameQueue;
typedef ThreadSafeQueue<BatchImageFramePtr> BatchImageFrameQueue;
typedef ThreadSafeQueue<BatchResultFramePtr> BatchResultFrameQueue;



typedef std::shared_ptr<ImageFrameQueue> ImageFrameQueuePtr;
typedef std::shared_ptr<ResultFrameQueue> ResultFrameQueuePtr;
typedef std::shared_ptr<BatchImageFrameQueue> BatchImageFrameQueuePtr;
typedef std::shared_ptr<BatchResultFrameQueue> BatchResultFrameQueuePtr;
