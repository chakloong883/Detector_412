#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include "detector.h"
#include <string>
#include <iostream>
#include <functional>
#include <future>
#include "DetectorThread/detector_thread.h"
#include "common/config_manager.h"
#include "common/glog_manager.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h" // �����ļ���־sink
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
        auto logManager = GlogManager::GetInstance(configPath);
        logger_ = logManager->getLogger();
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

        if (node_["draw_setting"]) {
            if (node_["draw_setting"]["drawimage"]) {
                drawImage_ = node_["draw_setting"]["drawimage"].as<bool>();
            }
            if (node_["draw_setting"]["drawlabel"]) {
                drawLabel_ = node_["draw_setting"]["drawlabel"].as<bool>();
            }
            if (node_["draw_setting"]["drawdefectsnum"]) {
                drawDefectsNum_ = node_["draw_setting"]["drawdefectsnum"].as<bool>();
            }
            if (node_["draw_setting"]["drawcolor"]) {
                if (node_["draw_setting"]["drawcolor"].IsSequence()) {
                    if (node_["draw_setting"]["drawcolor"].size() == 3) {
                        auto R = node_["draw_setting"]["drawcolor"][0].as<int>();
                        auto G = node_["draw_setting"]["drawcolor"][1].as<int>();
                        auto B = node_["draw_setting"]["drawcolor"][2].as<int>();
                        drawColor_ = cv::Scalar(R, G, B);
                    }
                }
            }
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
    }

    bool process(ImageFrameInside& inputFrameInside, ResultFrameInside& resultFrameInside) {
        auto inputframe = inputFrameInside.imageFrame;
        auto uuid = inputFrameInside.uuid;
        if (imageType_ == "rgb") {
            if (inputframe.channelNum != 3) {
                logger_->error("wrong channel num!, the imageType is {}, the channel num is {}", imageType_, inputframe.channelNum);
                return false;
            }
        }
        else if (imageType_ == "gray") {
            if (inputframe.channelNum != 1) {
                logger_->error("wrong channel num!, the imageType is {}, the channel num is {}", imageType_, inputframe.channelNum);

                return false;
            }
        }
        if (inputframe.imageHeight != imageSizeH_ || inputframe.imageWidth != imageSizeW_) {
            logger_->error("wrong image size, the image Height is  {}, the image Width is {}, it should be {} and {}", inputframe.imageHeight, inputframe.imageWidth, imageSizeH_, imageSizeW_);
            return false;
        }
        auto start1 = std::chrono::high_resolution_clock::now();
        if (!detectorThread_->push(inputFrameInside)) {
            logger_->error("the input queue full!");
            return false;
        }
        auto start2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = start2 - start1;
        logger_->info("push��ʱ: {} ����", elapsed.count());

        if (!detectorThread_->get(resultFrameInside, uuid)) {
            logger_->error("can't get the result frame!");
            return false;
        }
        auto start3 = std::chrono::high_resolution_clock::now();
        elapsed = start3 - start2;
        logger_->info("get��ʱ: {} ����", elapsed.count());

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
                    cv::putText(image, labelText, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, drawColor_, 2);
                }
                cv::Point p1(resultFrameInside.resultFrame.defects->at(i).box.left, resultFrameInside.resultFrame.defects->at(i).box.top);
                cv::Point p2(resultFrameInside.resultFrame.defects->at(i).box.right, resultFrameInside.resultFrame.defects->at(i).box.bottom);
                cv::rectangle(image, cv::Rect(p1, p2), drawColor_, 2);
            }
            if (drawDefectsNum_) {
                std::string text = "Total Objects: " + std::to_string(resultFrameInside.resultFrame.numDefects);
                cv::putText(image, text, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.5, drawColor_, 2);
            }
            start2 = std::chrono::high_resolution_clock::now();
            elapsed = start2 - start3;
            logger_->info("��ͼ��ʱ: {} ����", elapsed.count());
        }
        return true;
    }
private:
    std::shared_ptr<ConfigManager> configManager_;
    std::shared_ptr<spdlog::logger> logger_;
    std::shared_ptr<DetectorThread> detectorThread_;
    std::mutex processMutex_;
    std::string imageType_ = "rgb";
    int imageSizeH_ = 1280;
    int imageSizeW_ = 1280;
    float shrinkRatio_ = 0;
    bool drawImage_ = true;
    bool drawLabel_ = true;
    bool drawDefectsNum_ = false;
    cv::Scalar drawColor_ = cv::Scalar(0, 150, 0);
    YAML::Node node_;

};



Detector::Detector(std::string& configPath) {
    pimpl = std::make_unique<Impl>(configPath);
}

Detector::~Detector() {
    pimpl->~Impl();
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

