#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include "detector.h"
#include <string>
#include <iostream>
#include <functional>
#include <future>
#include "DetectorThread/detector_thread.h"


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
        detectorThread_ = std::make_shared<DetectorThread>();
        if (!detectorThread_->Init(configPath)) {
            throw "detector Thread init error!";
        }
    }

    ~Impl() {
        detectorThread_.reset();
    }

    bool process(std::string& uuid, ImageFrame& inputframe, ResultFrame& resultframe) {
        if (!detectorThread_->push(inputframe)) {
            return false;
        }
        if (!detectorThread_->get(resultframe, uuid)) {
            return false;
        }
        //std::unique_lock<std::mutex> lock(processMutex_);
        //std::cout << "processNum:" << processNum << std::endl;
        //processNum += 1;
        return true;
    }
private:
    std::shared_ptr<DetectorThread> detectorThread_;
    std::mutex processMutex_;
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

