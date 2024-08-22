#pragma once
#include "common_frame.h"


struct ImagePos {
	int rowBias;
	int colBias;
	bool isLast;
};


struct BatchImageFrame {
	std::shared_ptr<void> buffer;
	std::shared_ptr<void> bufferCpu;
	std::shared_ptr<std::vector<ImagePos>> imagesPos;
	std::vector<std::string> batchuuid;
	int imageWidth;
	int imageHeight;
	int channelNum;
	int batchSize;
};


struct BatchResultFrame {
	std::shared_ptr<std::vector<std::vector<Defect>>> batchDefects;
	std::shared_ptr<std::vector<ImagePos>> imagesPos;
	std::shared_ptr<std::vector<Circle>> circles;
	std::vector<std::string> batchuuid;
};
