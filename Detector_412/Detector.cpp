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
        if (!detectorThread_->push(inputframe)) {
            std::cout << "the input queue full!" << std::endl;
            return false;
        }
        if (!detectorThread_->get(resultframe, uuid)) {
            std::cout << "can't get the result frame!" << std::endl;
            return false;
        }
        std::cout << "get result frame success!" << std::endl;
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

};



Detector::Detector(std::string& configPath) {
    pimpl = std::make_unique<Impl>(configPath);
}

Detector::~Detector() {
    pimpl->~Impl();
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

