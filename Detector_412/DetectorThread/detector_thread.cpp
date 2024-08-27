#define _USE_MATH_DEFINES
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
        detectThreadShouldExit_ = true;
        detectThread_.join();
    }
    yolo_.reset();
}


bool DetectorThread::push(ImageFrameInside& frame) {
    if (!imageQueue_->Enqueue(std::make_shared<ImageFrameInside>(frame))) {
        std::cout << "image queue full!" << std::endl;
        return false;
    }
    else {
        std::cout << "imageFrame add!, size: " << imageQueue_->size() << std::endl;
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
    auto bufferCpu = (*batchImage)->bufferCpu;
    auto imagesPos = (*batchImage)->imagesPos;
    
    BatchResultFramePtr outputFrame(new BatchResultFrame({
        std::make_shared<std::vector<std::vector<Defect>>>(batchSize_),
        imagesPos,
        std::make_shared<std::vector<Circle>>(),
        batchuuid
    }));
    
    if (yolo_) {
        yolo_->setInputData(buffer);
        utils::DeviceTimer d_t1; yolo_->preprocess();  float t1 = d_t1.getUsedTime();
        utils::DeviceTimer d_t2; yolo_->infer();       float t2 = d_t2.getUsedTime();
        utils::DeviceTimer d_t3; yolo_->postprocess(static_cast<size_t>(batchSize_)); float t3 = d_t3.getUsedTime();
        sample::gLogInfo << "preprocess time = " << t1 / batchSize_ << "; "
            "infer time = " << t2 / batchSize_ << "; "
            "postprocess time = " << t3 / batchSize_ << std::endl;
        *(outputFrame->batchDefects) = yolo_->getObjectss();
        yolo_->reset();
    }

    if (traditionalDetection_) {
        traditionalDetection_->execute(*batchImage, outputFrame);
    }
    
    if (!batchResultQueue_->Enqueue(outputFrame)) {
        std::cout << "queue full" << std::endl;
        return false;
    }
    else {
        std::cout << "output queue add. new size: " << batchResultQueue_->size() << std::endl;

    }
    auto start2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = start2 - start1;
    std::cout << "检测线程耗时: " << elapsed.count() << " 毫秒" << std::endl;
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
            ResultFrameInside resultFrame({ { std::make_shared<std::vector<Defect>>(defects), false, "" }, circle, uuid });
            cv::Mat image;
            tools::regularzation(resultFrame, node_);
            std::lock_guard<std::mutex> lock(resultFrameMapMutex_);
            resultFrameMap_[uuid] = resultFrame;
            resultFrameMapCV_.notify_all();
        }
    }
    auto start2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = start2 - start1;
    std::cout << "后处理线程耗时: " << elapsed.count() << " 毫秒" << std::endl;
    return true;
}

void DetectorThread::detectThread() {
    while (!detectThreadShouldExit_) {
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
            std::cout << "yolov8 use!" << std::endl;
            yolo_ = std::make_shared<yolo::YOLOV8>(configPath);
        }
    }
    else {
        std::cout << "object detector use error, detector name:" << objectDetectorUse << std::endl;
        return false;
    }
    if (!yolo_->init()) {
        sample::gLogError << "initEngine() ocur errors!" << std::endl;
        return false;
    }
    yolo_->check();
    return true;

}


bool DetectorThread::Init(std::string& configPath) {
    configManager_ = ConfigManager::GetInstance(configPath);
    node_ = configManager_->getConfig();
    //this->registerTraditionFun(std::bind(&ImageProcess::detectGeneral, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
    if (node_["object_detection"]) {
        batchSize_ = node_["object_detection"]["batchsize"].as<int>();
        if (!createObjectDetection(configPath)) {
            std::cout << "create detector failed!" << std::endl;
            return false;
        }
    }
    if (node_["tradition_detection"]) {
        batchSize_ = node_["tradition_detection"]["batchsize"].as<int>();
        if (node_["tradition_detection"]["method"]) {
            needTraditionDetection_ = true;
            if (node_["tradition_detection"]["method"].as<std::string>() == "detectmaociyijiaohuahen") {
                //this->registerTraditionFun(std::bind(&ImageProcess::detectMaociBatchImages, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
                traditionalDetection_ = std::make_shared<ImageProcess::DetectMaociHuahenBatchImages>(configPath);
            }
            else {
                traditionalDetection_ = std::make_shared<ImageProcess::DetectGeneralBatchImages>(configPath);
            }
        }
        //if (node_["tradition_detection"]["thresholdvalue1"]) {
        //    thresholdValue1_ = node_["tradition_detection"]["thresholdvalue1"].as<int>();
        //}
        //if (node_["tradition_detection"]["thresholdvalue2"]) {
        //    thresholdValue2_ = node_["tradition_detection"]["thresholdvalue2"].as<int>();
        //}
        //if (node_["tradition_detection"]["inv"]) {
        //    inv_ = node_["tradition_detection"]["inv"].as<bool>();
        //}
    }
    copyImageToCuda_ = std::make_shared<tools::CopyImageToCuda>(batchSize_, imageQueue_, batchImageQueue_);
    assert(!detectThread_.joinable());
    detectThread_ = std::thread(&DetectorThread::detectThread, this);
    return true;
}
