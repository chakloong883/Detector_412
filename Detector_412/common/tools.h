#pragma once
#include "common_frame.h"
#include "common_queue.h"
#include "yaml-cpp/yaml.h"
#include "opencv2/opencv.hpp"

namespace tools {
    void getCVBatchImages(std::vector<cv::Mat>& batchImages, BatchImageFramePtr frame);
    float calculateDistance(const Point& p1, const Point& p2);
    double normalizeAngle(double angle);
    double calculateAngleDifference(double angle1, double angle2);
    void shrinkFilter(Defect& defect, Circle& circle, float& shrink, float& shrinkRatio, bool& keep, float& distanceBias);
    bool compare(const std::string& condition, float a);
    void regularzation(ResultFrame& frame, Circle& circle, YAML::Node& config);
    class CopyImageToCuda {
    public:
        CopyImageToCuda(int batchSize, ImageFrameQueuePtr inputQueue, BatchImageFrameQueuePtr outputQueue) :batchSize_(batchSize), inputQueue_(inputQueue), outputQueue_(outputQueue){}
        bool execute();
    private:
        std::vector<ImagePos> imagePos_;
        std::vector<std::string> batchuuid_;
        void* data_ = nullptr;
        void* dataCpu_ = nullptr;
        unsigned char* dataPoint_ = nullptr;
        unsigned char* dataPointCpu_ = nullptr;
        int frameCount_ = 0;
        int batchSize_;
        ImageFrameQueuePtr inputQueue_;
        BatchImageFrameQueuePtr outputQueue_;
    };

    double distance(const Point& p1, const Point& p2);

    // 计算点到线段的最短距离
    double pointToSegmentDistance(const Point& p, const Point& v, const Point& w);

    // 判断一个点是否在多边形内部
    bool isPointInPolygon(const Point& point, const std::vector<Point>& contour);

    // 计算点到多边形轮廓的最短距离
    double shortestDistanceToContour(const Point& point, const std::vector<Point>& contour);


};