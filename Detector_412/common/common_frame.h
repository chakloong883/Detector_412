#pragma once
#include <memory>
#include <vector>
#include <string>
struct ImageFrame {
	std::shared_ptr<void> buffer;
	int imageWidth;
	int imageHeight;
	int channelNum;
    std::string uuid;
};

struct Box
{
    float left, top, right, bottom, confidence;
    int label;
    Box (float left, float top, float right, float bottom, float confidence, int label):left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
    Box() = default;
    void addBias(int rowBias, int colBias) {
        left += colBias;
        right += colBias;
        top += rowBias;
        bottom += rowBias;
    }
};

struct Defect {
    std::string defectName;
    Box box;
    Defect() = default;
    Defect (std::string defectName, Box box):defectName(defectName), box(box){}
};

struct ResultFrame {
    std::shared_ptr<std::vector<Defect>> defects;
    std::string uuid;
    bool NG;
    std::string NGStateMent;
};