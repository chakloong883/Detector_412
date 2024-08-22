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
struct Point {
    float x;
    float y;
};

struct Size {
    float width;
    float height;
};

struct Circle {
    Point circlePoint;
    Size size;
    float angle;
    float radius;
};


struct Box
{
    float left, top, right, bottom, width = 0.0, height = 0.0, distance0uter = 0.0, distanceInner = 0.0, confidence;
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
    std::string objFocus;
    float objValue = 0.0;
    Defect() = default;
    Defect (std::string defectName, Box box):defectName(defectName), box(box){}
};

struct ResultFrame {
    std::shared_ptr<std::vector<Defect>> defects;
    std::string uuid;
    Circle circle;
    bool NG;
    std::string NGStateMent;
};