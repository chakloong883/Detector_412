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
        node_ = configManager_->getConfig();
        if (node_["anomaly_detection"]) {
            if (node_["anomaly_detection"]["imagetype"]) {
                imageType_ = node_["anomaly_detection"]["imagetype"].as<std::string>();
            }
            if (node_["anomaly_detection"]["imagesizeH"]) {
                imageSizeH_ = node_["anomaly_detection"]["imagesizeH"].as<int>();
                imageSizeW_ = node_["anomaly_detection"]["imagesizeW"].as<int>();
            }

        }
        if (node_["object_detection"]) {
            if (node_["object_detection"]["imagetype"]) {
                imageType_ = node_["object_detection"]["imagetype"].as<std::string>();
            }
            if (node_["object_detection"]["imagesizeH"]) {
                imageSizeH_ = node_["object_detection"]["imagesizeH"].as<int>();
                imageSizeW_ = node_["object_detection"]["imagesizeW"].as<int>();
            }

        }
        if (node_["tradition_detection"]) {
            if (node_["tradition_detection"]["imagetype"]) {
                imageType_ = node_["tradition_detection"]["imagetype"].as<std::string>();
            }
            if (node_["tradition_detection"]["imagesizeH"]) {
                imageSizeH_ = node_["tradition_detection"]["imagesizeH"].as<int>();
                imageSizeW_ = node_["tradition_detection"]["imagesizeW"].as<int>();
            }

        }
        if (node_["drawimage"]) {
            drawImage_ = node_["drawimage"].as<bool>();
        }

        if (node_["drawlabel"]) {
            drawLabel_ = node_["drawlabel"].as<bool>();
        }

        detectorThread_ = std::make_shared<DetectorThread>();
        if (!detectorThread_->Init(configPath)) {
            throw "detector Thread init error!";
        }
        if (node_["shrinkratio"]) {
            shrinkRatio_ = node_["shrinkratio"].as<float>();
        }
        
    }

    ~Impl() {
        //detectorThread_.reset();
    }

    bool process(ImageFrameInside& inputFrameInside, ResultFrameInside& resultFrameInside) {
        auto inputframe = inputFrameInside.imageFrame;
        auto uuid = inputFrameInside.uuid;
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
        if (inputframe.imageHeight != imageSizeH_ || inputframe.imageWidth != imageSizeW_) {
            std::cout << "wrong image size, the image Height is " << inputframe.imageHeight;
            std::cout << ", the imageWidth is " << inputframe.imageWidth;
            std::cout << "it should be: " << imageSizeH_ << " and " << imageSizeW_ << "." << std::endl;
            return false;
        }
        auto start1 = std::chrono::high_resolution_clock::now();
        if (!detectorThread_->push(inputFrameInside)) {
            std::cout << "the input queue full!" << std::endl;
            return false;
        }
        auto start2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = start2 - start1;
        std::cout << "push耗时: " << elapsed.count() << " 毫秒" << std::endl;

        if (!detectorThread_->get(resultFrameInside, uuid)) {
            std::cout << "can't get the result frame!" << std::endl;
            return false;
        }
        auto start3 = std::chrono::high_resolution_clock::now();
        elapsed = start3 - start2;
        std::cout << "get耗时: " << elapsed.count() << " 毫秒" << std::endl;

        if (drawImage_) {
            auto cvImageType = inputframe.channelNum == 1 ? CV_8UC1 : CV_8UC3;
            unsigned char* point = static_cast<unsigned char*>(inputframe.buffer.get());
            cv::Mat image(inputframe.imageHeight, inputframe.imageWidth, cvImageType, point);
            auto circle = resultFrameInside.circle;
            if (circle.size.height != 0 && circle.size.width != 0) {
                cv::Point2f pointCenter(circle.circlePoint.x, circle.circlePoint.y);
                cv::Size2f size(circle.size.width, circle.size.height);
                cv::RotatedRect rect(pointCenter, size, circle.angle);
                auto rect1 = rect;
                rect1.size.width += 50 * shrinkRatio_ * 2;
                rect1.size.height += 50 * shrinkRatio_ * 2;
                auto rect2 = rect;
                rect2.size.width += -20 * shrinkRatio_ * 2;
                rect2.size.height += -20 * shrinkRatio_ * 2;
                auto rect3 = rect;
                rect3.size.width += -175 * shrinkRatio_ * 2;
                rect3.size.height += -175 * shrinkRatio_ * 2;
                cv::ellipse(image, rect, cv::Scalar(0, 50, 0), 2);
                cv::ellipse(image, rect1, cv::Scalar(0, 100, 255), 2);
                cv::ellipse(image, rect2, cv::Scalar(0, 110, 255), 2);
                cv::ellipse(image, rect3, cv::Scalar(0, 120, 255), 2);
            }


            for (std::size_t i = 0; i < resultFrameInside.resultFrame.defects->size(); i++) {
                auto defectName = resultFrameInside.resultFrame.defects->at(i).defectName;
                std::string labelText = defectName;
                if (resultFrameInside.resultFrame.defects->at(i).objFocus.size()) {
                    labelText += "_" + resultFrameInside.resultFrame.defects->at(i).objFocus + ":" + cv::format("%.3f", resultFrameInside.resultFrame.defects->at(i).objValue);
                }
                if (drawLabel_) {
                    cv::Point textOrigin(resultFrameInside.resultFrame.defects->at(i).box.left, resultFrameInside.resultFrame.defects->at(i).box.top - 5);
                    cv::putText(image, labelText, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 2);
                }
                cv::Point p1(resultFrameInside.resultFrame.defects->at(i).box.left, resultFrameInside.resultFrame.defects->at(i).box.top);
                cv::Point p2(resultFrameInside.resultFrame.defects->at(i).box.right, resultFrameInside.resultFrame.defects->at(i).box.bottom);
                auto color = defectName == "corner" ? cv::Scalar(255, 255, 255) : cv::Scalar(150, 150, 150);
                cv::rectangle(image, cv::Rect(p1, p2), color, 2);
            }
            if (!drawLabel_) {
                std::string text = "Total Objects: " + std::to_string(resultFrameInside.resultFrame.numDefects);
                cv::putText(image, text, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
            }
            start2 = std::chrono::high_resolution_clock::now();
            elapsed = start2 - start3;
            std::cout << "画图耗时: " << elapsed.count() << " 毫秒，" << "平均用时：" << elapsed.count() / resultFrameInside.resultFrame.defects->size() << "毫秒" << std::endl;
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
    int imageSizeH_ = 1280;
    int imageSizeW_ = 1280;
    float shrinkRatio_ = 0;
    bool drawImage_ = true;
    bool drawLabel_ = true;
    YAML::Node node_;

};



Detector::Detector(std::string& configPath) {
    pimpl = std::make_unique<Impl>(configPath);
}

Detector::~Detector() {
    pimpl->~Impl();
    std::cout << "析构成功" << std::endl;
}


bool Detector::process(ImageFrame& inputframe, ResultFrame& resultframe) {
    std::string uuid = uuid::generate_uuid_v4();
    ImageFrameInside inputFrameInside({ inputframe , uuid });
    ResultFrameInside resultFrameInside;
    if (!pimpl->process(inputFrameInside, resultFrameInside)) {
        return false;
    }
    else {
        resultframe = resultFrameInside.resultFrame;
        return true;
    }
}

