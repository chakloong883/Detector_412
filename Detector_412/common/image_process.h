#pragma once
#include <opencv2/opencv.hpp>
#include "common_queue.h"


namespace ImageProcess {
    void cropImage(cv::Mat& img, std::vector<ImagePos>& imagePos, int& cropHeight, int& cropWidth, float& overLap);
    void detectMaociBatchImages(std::vector<cv::Mat>& images, BatchResultFramePtr outputframe, int thresholdValue1, int thresholdValue2, bool inv);
    void detectGeneral(std::vector<cv::Mat>& images, BatchResultFramePtr outputframe, int thresholdValue1, int thresholdValue2, bool inv = true);
};