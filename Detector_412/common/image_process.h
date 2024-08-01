#pragma once
#include <opencv2/opencv.hpp>

struct ImagePos {
    int rowBias;
    int colBias;
    bool isLast;
};


namespace ImageProcess {
    void cropImage(cv::Mat& img, std::vector<ImagePos>& imagePos, int& cropHeight, int& cropWidth, float& overLap);
};