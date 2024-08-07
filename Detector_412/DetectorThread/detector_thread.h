#pragma once
#include <thread>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include "../common/common_frame.h"
#include "../common/common_queue.h"
#include "../common/config_manager.h"
#include "../common/common_frame_inside.h"
#include "../yolo/yolo.h"
#include <map>


class DetectorThread {
public:
	DetectorThread();
	~DetectorThread();
	bool Init(std::string& configPath);
	bool push(ImageFrame& frame);
	bool get(ResultFrame& frame, std::string& uuid);
private:
	bool createDetector(std::string& detectorUse);
	void copyImageToCudaThread();
	void detectMaociBatchImages(std::vector<cv::Mat>& image, BatchResultFramePtr outputframe);
	void registerTraditionFun(std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr)> cb);
	std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr)> traditionalDetectBatchImagesFun_;

	void regularzation(ResultFrame& frame, cv::Mat& image, Circle& circle);
	void cropImageThread();
	void detectThread();
	void postprocessThread();
	int batchSize_;

	std::shared_ptr<yolo::YOLO>  yolo_;
	//ConfigManager* configManager_;
	
	std::thread copyImageToCudaThread_;
	std::thread cropImageThread_;
	std::thread detectThread_;
	std::thread postprocessThread_;

	std::atomic<bool> copyImageToCudaThreadShouldExit_;
	std::atomic<bool> cropImageThreadShouldExit_;
	std::atomic<bool> detectThreadShouldExit_;
	std::atomic<bool> postprocessThreadShouldExit_;

	ImageFrameQueuePtr imageQueue_;
	BatchImageFrameQueuePtr batchImageQueue_;
	BatchResultFrameQueuePtr batchResultQueue_;
	std::map<std::string, ResultFrame> resultFrameMap_;
	std::mutex resultFrameMapMutex_;
	std::condition_variable resultFrameMapCV_;
	std::shared_ptr<ConfigManager> configManager_;

};