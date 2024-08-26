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
	bool push(ImageFrameInside& frame);
	bool get(ResultFrameInside& frame, std::string& uuid);
private:
	bool createObjectDetection(std::string& detectorUse);
	void registerTraditionFun(std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr, int, int, bool)> cb);
	std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr, int, int, bool)> traditionalDetectBatchImagesFun_;

	bool detectFunc();
	bool postprocessFun();
	void detectThread();
	int batchSize_ = 1;

	std::shared_ptr<yolo::YOLO>  yolo_;
	std::shared_ptr<tools::CopyImageToCuda> copyImageToCuda_;
	std::shared_ptr<ImageProcess::DetectGeneralBatchImages> traditionalDetection_;
	
	std::thread detectThread_;
	std::atomic<bool> detectThreadShouldExit_;
	ImageFrameQueuePtr imageQueue_;
	BatchImageFrameQueuePtr batchImageQueue_;
	BatchResultFrameQueuePtr batchResultQueue_;
	std::map<std::string, ResultFrameInside> resultFrameMap_;
	std::mutex resultFrameMapMutex_;
	std::condition_variable resultFrameMapCV_;
	std::shared_ptr<ConfigManager> configManager_;
	YAML::Node node_;
	
	// ´ýÉ¾³ý
	bool needObjectDetection_ = true;
	bool needTraditionDetection_ = true;
	int thresholdValue1_ = 188;
	int thresholdValue2_ = 25;
	bool inv_ = false;

};