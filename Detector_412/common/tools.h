#pragma once
#include "common_frame.h"
#include "common_queue.h"
#include "yaml-cpp/yaml.h"
#include "glog_manager.h"
#include "opencv2/opencv.hpp"
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

namespace tools {
    float calculateDistance(const Point& p1, const Point& p2);
    double normalizeAngle(double angle);
    double calculateAngleDifference(double angle1, double angle2);
    void shrinkFilter(Defect& defect, Circle& circle, float& shrink, float& shrinkRatio, bool& keep, float& distanceBias);
    bool compare(const std::string& condition, float a);
    void regularzation(ResultFrameInside& frame, YAML::Node& config, std::shared_ptr<spdlog::logger> logger);
    class CopyImageToCuda {
    public:
        CopyImageToCuda(int batchSize, ImageFrameQueuePtr inputQueue, BatchImageFrameQueuePtr outputQueue, std::shared_ptr<spdlog::logger> logger) :batchSize_(batchSize), inputQueue_(inputQueue), outputQueue_(outputQueue), logger_(logger){}
        bool execute();
    private:
        std::vector<ImagePos> imagePos_;
        std::vector<std::string> batchuuid_;
        void* data_ = nullptr;
        void* dataCpu_ = nullptr;
        unsigned char* dataPoint_ = nullptr;
        int frameCount_ = 0;
        int batchSize_;
        ImageFrameQueuePtr inputQueue_;
        BatchImageFrameQueuePtr outputQueue_;
        std::shared_ptr<spdlog::logger> logger_;
    };

    double distance(const Point& p1, const Point& p2);

    // 计算点到线段的最短距离
    double pointToSegmentDistance(const Point& p, const Point& v, const Point& w);

    // 判断一个点是否在多边形内部
    bool isPointInPolygon(const Point& point, const std::vector<Point>& contour);

    // 计算点到多边形轮廓的最短距离
    double shortestDistanceToContour(const Point& point, const std::vector<Point>& contour);


    class HostTimer
    {
    public:
        HostTimer();
        float getUsedTime(); // while timing for cuda code, add "cudaDeviceSynchronize();" before this
        ~HostTimer();

    private:
        std::chrono::steady_clock::time_point t1;
        std::chrono::steady_clock::time_point t2;
    };

    class DeviceTimer
    {
    public:
        DeviceTimer();
        float getUsedTime();
        // overload
        DeviceTimer(cudaStream_t ctream);
        float getUsedTime(cudaStream_t ctream);

        ~DeviceTimer();

    private:
        cudaEvent_t start, end;
    };

};