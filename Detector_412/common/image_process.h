#pragma once
#include <opencv2/opencv.hpp>
#include "common_queue.h"
#include "config_manager.h"
#include <cuda_runtime_api.h>
#include "../yolo/kernel_function.h"
#include "../common/glog_manager.h"


namespace ImageProcess {
    void cropImage(cv::Mat& img, std::vector<ImagePos>& imagePos, int& cropHeight, int& cropWidth, float& overLap);

    class DetectGeneralBatchImages {
    public:
        DetectGeneralBatchImages(std::string& configPath);
        ~DetectGeneralBatchImages();
        virtual void execute(BatchImageFramePtr inputFrame, BatchResultFramePtr outputframe);
    protected:
        unsigned char* inputDevice_;
        unsigned char* grayDevice_;
        unsigned char* binaryDevice_;
        unsigned char* erodeDevice_;
        unsigned char* dilateDevice_;
        int batchSize_;
        int srcWidth_;
        int srcHeight_;
        int dstWidth_;
        int dstHeight_;
        int thresHold1_;
        int thresHold2_;

        bool inv_;
        std::string imageType_;
        std::shared_ptr<spdlog::logger> logger_;


    };

    class DetectMaociHuahenBatchImages : public DetectGeneralBatchImages {
    public:
        DetectMaociHuahenBatchImages(std::string& configPath):ImageProcess::DetectGeneralBatchImages(configPath){}
        virtual void execute(BatchImageFramePtr inputFrame, BatchResultFramePtr outputframe);
    };

    class DetectCornerBatchImages : public DetectGeneralBatchImages {
    public:
        DetectCornerBatchImages(std::string& configPath) :ImageProcess::DetectGeneralBatchImages(configPath) {}
        virtual void execute(BatchImageFramePtr inputFrame, BatchResultFramePtr outputframe);
    };
};