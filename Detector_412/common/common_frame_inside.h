#pragma once
#include "common_frame.h"

struct Point {
	float x;
	float y;
};

struct Size {
	float width = 0;
	float height = 0;
};

struct Circle {
	Point circlePoint;
	Size size;
	float angle;
	float radius;
};


struct ImageFrameInside {
	ImageFrame imageFrame;
	std::string uuid;
};

struct ResultFrameInside {
	ResultFrame resultFrame;
	Circle circle;
	std::string uuid;
};


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
