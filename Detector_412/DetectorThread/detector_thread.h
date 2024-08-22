#pragma once
#include <thread>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include "../common/common_frame.h"
#include "../common/common_queue.h"
#include "../common/config_manager.h"
#include "../common/common_frame_inside.h"
#include "../common/image_process.h"
#include "../yolo/yolo.h"
#include "../common/tools.h"
#include "yaml-cpp/yaml.h"
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
	bool copyImageToCudaFunc(bool useSingle = true);
	void copyImageToCudaThread();
	void detectGeneral(std::vector<cv::Mat>& images, BatchResultFramePtr outputframe);
	void detectMaociBatchImages(std::vector<cv::Mat>& image, BatchResultFramePtr outputframe);
	void registerTraditionFun(std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr, int, int, bool)> cb);
	std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr, int, int, bool)> traditionalDetectBatchImagesFun_;

	void regularzation(ResultFrame& frame, cv::Mat& image, Circle& circle);
	void cropImageThread();
	bool detectFunc();
	void detectThread();
	bool postprocessFun();
	void postprocessThread();
	void allInOneThread();
	int batchSize_;

	std::shared_ptr<yolo::YOLO>  yolo_;
	std::shared_ptr<tools::CopyImageToCuda> copyImageToCuda_;
	//ConfigManager* configManager_;
	
	std::thread copyImageToCudaThread_;
	std::thread cropImageThread_;
	std::thread detectThread_;
	std::thread postprocessThread_;
	std::thread allInOneThread_;


	std::atomic<bool> copyImageToCudaThreadShouldExit_;
	std::atomic<bool> cropImageThreadShouldExit_;
	std::atomic<bool> detectThreadShouldExit_;
	std::atomic<bool> postprocessThreadShouldExit_;
	std::atomic<bool> allInOneThreadShouldExit_;


	ImageFrameQueuePtr imageQueue_;
	BatchImageFrameQueuePtr batchImageQueue_;
	BatchResultFrameQueuePtr batchResultQueue_;
	std::map<std::string, ResultFrame> resultFrameMap_;
	std::mutex resultFrameMapMutex_;
	std::condition_variable resultFrameMapCV_;
	std::shared_ptr<ConfigManager> configManager_;
	YAML::Node node_;
	
	bool needObjectDetection_ = true;
	int thresholdValue1_ = 188;
	int thresholdValue2_ = 25;
	bool inv_ = false;

	// 用于拷贝线程
	std::vector<ImagePos> imagePos_;
	std::vector<std::string> batchuuid_;
	void* data_ = nullptr;
	void* dataCpu_ = nullptr;
	unsigned char* dataPoint_ = nullptr;
	unsigned char* dataPointCpu_ = nullptr;
	int frameCount_ = 0;

};