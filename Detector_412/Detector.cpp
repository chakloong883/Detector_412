#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include "detector.h"
#include <string>
#include <iostream>
#include <functional>
#include <future>
#include "DetectorThread/detector_thread.h"
#include "common/config_manager.h"


//Detector* Detector::instance{ nullptr };
//std::mutex Detector::mutex_;

static int processNum = 0;

#include <random>
#include <sstream>

namespace uuid {
    static std::random_device              rd;
    static std::mt19937                    gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::string generate_uuid_v4() {
        std::stringstream ss;
        int i;
        ss << std::hex;
        for (i = 0; i < 8; i++) {
            ss << dis(gen);
        }
        ss << "-";
        for (i = 0; i < 4; i++) {
            ss << dis(gen);
        }
        ss << "-4";
        for (i = 0; i < 3; i++) {
            ss << dis(gen);
        }
        ss << "-";
        ss << dis2(gen);
        for (i = 0; i < 3; i++) {
            ss << dis(gen);
        }
        ss << "-";
        for (i = 0; i < 12; i++) {
            ss << dis(gen);
        };
        return ss.str();
    }
}

struct Detector::Impl {
public:
    Impl(std::string&configPath) {
        configManager_ = ConfigManager::GetInstance(configPath);
        auto node = configManager_->getConfig();
        if (node["object_detecion"]) {
            if (node["object_detecion"]["imagetype"]) {
                imageType_ = node["object_detecion"]["imagetype"].as<std::string>();
            }
            if (node["object_detecion"]["imagesize"]) {
                imageSize_ = node["object_detecion"]["imagesize"].as<int>();
            }

        }
        if (node["drawimage"]) {
            drawImage_ = node["drawimage"].as<bool>();
        }
        detectorThread_ = std::make_shared<DetectorThread>();
        if (!detectorThread_->Init(configPath)) {
            throw "detector Thread init error!";
        }
    }

    ~Impl() {
        detectorThread_.reset();
    }

    bool process(std::string& uuid, ImageFrame& inputframe, ResultFrame& resultframe) {
        if (imageType_ == "rgb") {
            if (inputframe.channelNum != 3) {
                std::cout << "wrong channel num!, the imageType is " << imageType_ << ", the channel num is:" << inputframe.channelNum << std::endl;
                return false;
            }
        }
        else if (imageType_ == "gray") {
            if (inputframe.channelNum != 1) {
                std::cout << "wrong channel num!, the imageType is " << imageType_ << ", the channel num is:" << inputframe.channelNum << std::endl;
                return false;
            }
        }
        if (inputframe.imageHeight != imageSize_ || inputframe.imageWidth != imageSize_) {
            std::cout << "wrong image size, the image Height is " << inputframe.imageHeight;
            std::cout << ", the imageWidth is " << inputframe.imageWidth;
            std::cout << "it should be: " << imageSize_ << "." << std::endl;
            return false;
        }
        auto start1 = std::chrono::high_resolution_clock::now();
        if (!detectorThread_->push(inputframe)) {
            std::cout << "the input queue full!" << std::endl;
            return false;
        }
        auto start2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = start2 - start1;
        std::cout << "push耗时: " << elapsed.count() << " 毫秒" << std::endl;

        if (!detectorThread_->get(resultframe, uuid)) {
            std::cout << "can't get the result frame!" << std::endl;
            return false;
        }
        auto start3 = std::chrono::high_resolution_clock::now();
        elapsed = start3 - start2;
        std::cout << "get耗时: " << elapsed.count() << " 毫秒" << std::endl;

        if (drawImage_) {
            auto circle = resultframe.circle;
            cv::Point2f pointCenter(circle.circlePoint.x, circle.circlePoint.y);
            cv::Size2f size(circle.size.width, circle.size.height);
            cv::RotatedRect rect(pointCenter, size, circle.angle);
            auto rect1 = rect;
            rect1.size.width += 50 * 0.4;
            rect1.size.height += 50 * 0.4;
            unsigned char* point = static_cast<unsigned char*>(inputframe.buffer.get());
            auto cvImageType = inputframe.channelNum == 1 ? CV_8UC1 : CV_8UC3;
            cv::Mat image(inputframe.imageHeight, inputframe.imageWidth, cvImageType, point);
            cv::ellipse(image, rect, cv::Scalar(255, 0, 0), 2);
            cv::ellipse(image, rect1, cv::Scalar(0, 0, 255), 2);
            for (std::size_t i = 0; i < resultframe.defects->size(); i++) {
                auto defectName = resultframe.defects->at(i).defectName;
                std::string labelText = defectName + "_" + resultframe.defects->at(i).objFocus + ":" + cv::format("%.3f", resultframe.defects->at(i).objValue);
                cv::Point textOrigin(resultframe.defects->at(i).box.left, resultframe.defects->at(i).box.top - 5);
                cv::putText(image, labelText, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
                cv::Point p1(resultframe.defects->at(i).box.left, resultframe.defects->at(i).box.top);
                cv::Point p2(resultframe.defects->at(i).box.right, resultframe.defects->at(i).box.bottom);
                cv::rectangle(image, cv::Rect(p1, p2), cv::Scalar(128, 77, 207), 2);
            }
            start2 = std::chrono::high_resolution_clock::now();
            elapsed = start2 - start3;
            std::cout << "画图耗时: " << elapsed.count() << " 毫秒，" << "平均用时：" << elapsed.count() / resultframe.defects->size() << "毫秒" << std::endl;
        }
        //std::unique_lock<std::mutex> lock(processMutex_);
        //std::cout << "processNum:" << processNum << std::endl;
        //processNum += 1;
        return true;
    }
private:
    std::shared_ptr<ConfigManager> configManager_;
    std::shared_ptr<DetectorThread> detectorThread_;
    std::mutex processMutex_;
    std::string imageType_ = "rgb";
    int imageSize_ = 1280;
    bool drawImage_ = true;

};



Detector::Detector(std::string& configPath) {
    pimpl = std::make_unique<Impl>(configPath);
}

Detector::~Detector() {
    pimpl->~Impl();
    std::cout << "析构成功" << std::endl;
}

//Detector* Detector::GetInstance(std::string configPath) {
//    if (instance == nullptr) {
//        std::lock_guard<std::mutex> lock(mutex_);
//        if (instance == nullptr) {
//            Detector* temp = new Detector(configPath);
//            instance = temp;
//        }
//    }
//    return instance;
//}


bool Detector::process(ImageFrame& inputframe, ResultFrame& resultframe) {
    std::string uuid = uuid::generate_uuid_v4();
    inputframe.uuid = uuid;
    return pimpl->process(uuid, inputframe, resultframe);
}

