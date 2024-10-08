﻿#define _USE_MATH_DEFINES
#include <chrono>
#include<cuda_runtime_api.h>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <algorithm>
#include "detector_thread.h"
#include <cmath>


DetectorThread::DetectorThread() {
    imageQueue_ = std::make_shared<ImageFrameQueue>(100);
    batchImageQueue_ = std::make_shared<BatchImageFrameQueue>(100);
    batchResultQueue_ = std::make_shared<BatchResultFrameQueue>(100);
}

DetectorThread::~DetectorThread() {
    if (detectThread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(detectMutex_);
            detectThreadShouldExit_ = true;
        }
        detectCV_.notify_all();
        detectThread_.join();
    }
    yolo_.reset();
}


bool DetectorThread::push(ImageFrameInside& frame) {
    if (!imageQueue_->Enqueue(std::make_shared<ImageFrameInside>(frame))) {
        logger_->error("image queue full!");
        return false;
    }
    else {
        logger_->info("imageFrame add!, size: {}", imageQueue_->size());
        {
            std::lock_guard<std::mutex> lock(this->detectMutex_);
        }
        this->detectCV_.notify_all();
    }
    return true;
}

bool DetectorThread::get(ResultFrameInside& frame, std::string& uuid) {
    std::unique_lock<std::mutex> lock(resultFrameMapMutex_);
    resultFrameMapCV_.wait(lock, [&] {return resultFrameMap_.find(uuid) != resultFrameMap_.end(); });
    frame = resultFrameMap_[uuid];
    resultFrameMap_.erase(uuid);
    return true;
}

void DetectorThread::registerTraditionFun(std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr, int, int, bool)> cb) {
    traditionalDetectBatchImagesFun_ = cb;
}



bool DetectorThread::detectFunc() {
    auto batchImage = batchImageQueue_->Dequeue();
    if (!batchImage) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return false;
    }
    auto start1 = std::chrono::high_resolution_clock::now();
    auto batchuuid = (*batchImage)->batchuuid;
    auto buffer = (*batchImage)->buffer;
    auto imagesPos = (*batchImage)->imagesPos;
    
    BatchResultFramePtr outputFrame(new BatchResultFrame({
        std::make_shared<std::vector<std::vector<Defect>>>(batchSize_),
        imagesPos,
        std::make_shared<std::vector<Circle>>(),
        batchuuid
    }));

    if (anomalyDetection_) {
        anomalyDetection_->setInputData(buffer);
        utils::DeviceTimer d_t1; anomalyDetection_->preprocess();  float t1 = d_t1.getUsedTime();
        utils::DeviceTimer d_t2; anomalyDetection_->infer();       float t2 = d_t2.getUsedTime();
        utils::DeviceTimer d_t3; anomalyDetection_->postprocess(); float t3 = d_t3.getUsedTime();
        logger_->info("anomaly detection:");
        logger_->info("preprocess time = {}, infer time = {}, postprocess time = {}", t1 / batchSize_, t2 / batchSize_, t3 / batchSize_);
        auto batchDefects = anomalyDetection_->getObjectss();
        for (std::size_t i = 0; i < outputFrame->batchDefects->size(); i++) {
            outputFrame->batchDefects->at(i).insert(outputFrame->batchDefects->at(i).end(), batchDefects[i].begin(), batchDefects[i].end());
        }
        anomalyDetection_->reset();
    }

    if (yolo_) {
        yolo_->setInputData(buffer);
        utils::DeviceTimer d_t1; yolo_->preprocess();  float t1 = d_t1.getUsedTime();
        utils::DeviceTimer d_t2; yolo_->infer();       float t2 = d_t2.getUsedTime();
        utils::DeviceTimer d_t3; yolo_->postprocess(static_cast<size_t>(batchSize_)); float t3 = d_t3.getUsedTime();
        logger_->info("yolo detection:");
        logger_->info("preprocess time = {}, infer time = {}, postprocess time = {}", t1 / batchSize_, t2 / batchSize_, t3 / batchSize_);

        auto batchDefects = yolo_->getObjectss();
        for (std::size_t i = 0; i < outputFrame->batchDefects->size(); i++) {
            outputFrame->batchDefects->at(i).insert(outputFrame->batchDefects->at(i).end(), batchDefects[i].begin(), batchDefects[i].end());
        }
        yolo_->reset();
    }

    if (traditionalDetection_) {
        traditionalDetection_->execute(*batchImage, outputFrame);
    }
    
    if (!batchResultQueue_->Enqueue(outputFrame)) {
        logger_->error("batchResultQueue full");
        return false;
    }
    else {
        logger_->info("batchResultQueue add. new size: {}", batchResultQueue_->size());

    }
    auto start2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = start2 - start1;
    logger_->info("检测线程耗时:{} 毫秒", elapsed.count());
    return true;
}


bool DetectorThread::postprocessFun() {
    auto batchResultFrame = batchResultQueue_->Dequeue();
    if (!batchResultFrame) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        return false;
    }
    auto start1 = std::chrono::high_resolution_clock::now();
    auto batchuuid = (*batchResultFrame)->batchuuid;
    auto batchDefects = (*batchResultFrame)->batchDefects;
    auto imagesPos = (*batchResultFrame)->imagesPos;

    Circle circle({ {0,0}, {0,0}, 0.0, 0.0 });
    for (std::size_t i = 0; i < batchDefects->size(); i++) {
        auto uuid = (*batchResultFrame)->batchuuid[i];
        auto rowBias = (*batchResultFrame)->imagesPos->at(i).rowBias;
        auto colBias = (*batchResultFrame)->imagesPos->at(i).colBias;
        auto isLast = (*batchResultFrame)->imagesPos->at(i).isLast;
        auto defects = batchDefects->at(i);
        cv::RotatedRect rect;
        if ((*batchResultFrame)->circles->size()) {
            circle = (*batchResultFrame)->circles->at(i);
        }
        for (std::size_t i = 0; i < defects.size(); i++) {
            defects[i].box.addBias(rowBias, colBias);
        }

        if (isLast) {
            ResultFrameInside resultFrame({ { std::make_shared<std::vector<Defect>>(defects), defects.size(), false, ""}, circle, uuid});
            cv::Mat image;
            if (node_["defect_filter"]){
                tools::regularzation(resultFrame, node_, logger_);
            }
            if (resultFrame.resultFrame.numDefects) {
                resultFrame.resultFrame.NG = true;
            }
            std::lock_guard<std::mutex> lock(resultFrameMapMutex_);
            resultFrameMap_[uuid] = resultFrame;
            resultFrameMapCV_.notify_all();
        }
    }
    auto start2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = start2 - start1;
    logger_->info("后处理线程耗时: {} 毫秒", elapsed.count());
    return true;
}

void DetectorThread::detectThread() {
    while (!detectThreadShouldExit_) {
        {
            std::unique_lock<std::mutex> lock(this->detectMutex_);
            this->detectCV_.wait(lock, [this] {return (imageQueue_->size() || detectThreadShouldExit_); });
        }
        if (detectThreadShouldExit_) {
            break;
        }
        if (!copyImageToCuda_->execute()) {
            continue;
        }
        detectFunc();
        postprocessFun();
    }
}


bool DetectorThread::createObjectDetection(std::string& configPath) {
    batchSize_ = node_["object_detection"]["batchsize"].as<int>();
    auto objectDetectorUse = node_["object_detection"]["objectdetector"].as<std::string>();
    std::vector<std::string> modelChoice = { "yolov5", "yolov6", "yolov7", "yolov8" };
    auto iterPoint = std::find(modelChoice.begin(), modelChoice.end(), objectDetectorUse);
    if (iterPoint != modelChoice.end()) {
        if (std::distance(modelChoice.begin(), iterPoint) <= 2) {
            yolo_ = std::make_shared<yolo::YOLO>(configPath);
        }
        else if (*iterPoint == "yolov8") {
            logger_->info("yolov8 use!");
            yolo_ = std::make_shared<yolo::YOLOV8>(configPath);
        }
    }
    else {
        logger_->error("object detector use error, detector name: {}", objectDetectorUse);
        return false;
    }
    if (!yolo_->init()) {
        logger_->error("initEngine() ocur errors!");
        return false;
    }
    yolo_->check();
    return true;

}


bool DetectorThread::Init(std::string& configPath) {
    configManager_ = ConfigManager::GetInstance(configPath);
    node_ = configManager_->getConfig();
    auto logManager = GlogManager::GetInstance(configPath);
    logger_ = logManager->getLogger();
    if (!logger_) {
        std::cout << "日志记录器获取失败！" << std::endl;
        return false;
    }
    if (node_["object_detection"]) {
        batchSize_ = node_["object_detection"]["batchsize"].as<int>();
        if (!createObjectDetection(configPath)) {
            logger_->error("create detector failed!");
            return false;
        }
    }
    if (node_["anomaly_detection"]) {
        batchSize_ = node_["anomaly_detection"]["batchsize"].as<int>();
        anomalyDetection_ = std::make_shared<AnomalyDetection>(configPath);
        if (!anomalyDetection_->init()) {
            logger_->error("anomalydetection init failed!");
        }
        anomalyDetection_->check();
    }

    if (node_["tradition_detection"]) {
        batchSize_ = node_["tradition_detection"]["batchsize"].as<int>();
        if (node_["tradition_detection"]["method"]) {
            needTraditionDetection_ = true;
            if (node_["tradition_detection"]["method"].as<std::string>() == "detectmaociyijiaohuahen") {
                traditionalDetection_ = std::make_shared<ImageProcess::DetectMaociHuahenBatchImages>(configPath);
            }
            else if (node_["tradition_detection"]["method"].as<std::string>() == "general"){
                traditionalDetection_ = std::make_shared<ImageProcess::DetectGeneralBatchImages>(configPath);
            }
            else if (node_["tradition_detection"]["method"].as<std::string>() == "detectcorner") {
                traditionalDetection_ = std::make_shared<ImageProcess::DetectCornerBatchImages>(configPath);
            }
        }
    }
    copyImageToCuda_ = std::make_shared<tools::CopyImageToCuda>(batchSize_, imageQueue_, batchImageQueue_, logger_);
    assert(!detectThread_.joinable());
    detectThread_ = std::thread(&DetectorThread::detectThread, this);
    return true;
}
