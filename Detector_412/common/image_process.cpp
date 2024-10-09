#define _USE_MATH_DEFINES
#include "image_process.h"
#include "tools.h"
#include <cmath>
#include <algorithm>


void ImageProcess::cropImage(cv::Mat& image, std::vector<ImagePos>& imagePos, int& cropHeight, int& cropWidth, float& overLap) {
	int rowBias = std::round(cropHeight - (cropHeight * overLap));
	int colBias = std::round(cropWidth - (cropWidth * overLap));

    int rows = image.rows;
    int cols = image.cols;
    int total = (rows + cropHeight - 1) / cropWidth * (cols + cropWidth - 1) / cropWidth;
    int frameNumber = 0;
    bool isLastRow = false;
    bool isLastCol = false;

    int count = 0;
    int count1 = 0;
    int y = 0;
    while (true) {
        int nextRowBiasAdd = y + rowBias + cropHeight > rows ? rows - cropHeight - y : rowBias;
        isLastRow = nextRowBiasAdd == 0;
        int x = 0;
        while (true) {
            count += 1;
            int nextColBiasAdd = x + colBias + cropWidth > cols ? cols - cropWidth - x : colBias;
            isLastCol = nextColBiasAdd == 0;
            cv::Mat cropped = cv::Mat::zeros(cropHeight, cropWidth, image.type());

            // Define the region of interest in the original image, 避免裁剪出界
            int width = std::min(cropWidth, cols - x);
            int height = std::min(cropHeight, rows - y);
            cv::Rect roi(x, y, width, height);

            // Copy the region of interest to the cropped image
            image(roi).copyTo(cropped(cv::Rect(0, 0, width, height)));


            imagePos.push_back({
                y,
                x,
                isLastRow && isLastCol
                });

            x += nextColBiasAdd;
            if (isLastCol) {
                break;
            }
        }
        y += nextRowBiasAdd;
        if (isLastRow) {
            break;
        }
    }

}



ImageProcess::DetectGeneralBatchImages::DetectGeneralBatchImages(std::string& configPath) {
    auto configManager = ConfigManager::GetInstance(configPath);
    auto logManager = GlogManager::GetInstance(configPath);
    logger_ = logManager->getLogger();
    auto node = configManager->getConfig();
    batchSize_ = node["tradition_detection"]["batchsize"].as<int>();
    srcHeight_ = node["tradition_detection"]["imagesizeH"].as<int>();
    srcWidth_ = node["tradition_detection"]["imagesizeW"].as<int>();
    dstHeight_ = node["tradition_detection"]["imagesizeH"].as<int>();
    dstWidth_ = node["tradition_detection"]["imagesizeW"].as<int>();
    imageType_ = node["tradition_detection"]["imagetype"].as<std::string>();
    inv_ = node["tradition_detection"]["inv"].as<bool>();
    thresHold1_ = node["tradition_detection"]["thresholdvalue1"].as<int>();
    thresHold2_ = node["tradition_detection"]["thresholdvalue2"].as<int>();


    CHECK(cudaMalloc(&grayDevice_, batchSize_ * dstHeight_ * dstWidth_ * sizeof(unsigned char)));
    CHECK(cudaMalloc(&binaryDevice_, batchSize_ * dstHeight_ * dstWidth_ * sizeof(unsigned char)));
    CHECK(cudaMalloc(&erodeDevice_, batchSize_ * dstHeight_ * dstWidth_ * sizeof(unsigned char)));
    CHECK(cudaMalloc(&dilateDevice_, batchSize_ * dstHeight_ * dstWidth_ * sizeof(unsigned char)));

}

ImageProcess::DetectGeneralBatchImages::~DetectGeneralBatchImages() {
    CHECK(cudaFree(grayDevice_));
    CHECK(cudaFree(binaryDevice_));
    CHECK(cudaFree(erodeDevice_));
    CHECK(cudaFree(dilateDevice_));
}


void ImageProcess::DetectGeneralBatchImages::execute(BatchImageFramePtr inputFrame, BatchResultFramePtr outputframe) {
    tools::DeviceTimer d1;
    auto gpuBuffer = inputFrame->buffer;
    unsigned char* grayGpuBuf = nullptr;
    if (imageType_ == "rgb") {
        grayGpuBuf = grayDevice_;
        rgb2grayDevice(batchSize_, static_cast<unsigned char*>(gpuBuffer.get()), srcWidth_, srcHeight_, grayDevice_, dstWidth_, dstHeight_);
        binaryDevice(batchSize_, grayDevice_, srcWidth_, srcHeight_, binaryDevice_, dstWidth_, dstHeight_, inv_, thresHold1_);
        erosionDevice(batchSize_, binaryDevice_, srcWidth_, srcHeight_, erodeDevice_, dstWidth_, dstHeight_, 5);
        dilationDevice(batchSize_, erodeDevice_, srcWidth_, srcHeight_, dilateDevice_, dstWidth_, dstHeight_, 5);
    }
    else {
        grayGpuBuf = static_cast<unsigned char*>(gpuBuffer.get());
        binaryDevice(batchSize_, static_cast<unsigned char*>(gpuBuffer.get()), srcWidth_, srcHeight_, binaryDevice_, dstWidth_, dstHeight_, inv_, thresHold1_);
        erosionDevice(batchSize_, binaryDevice_, srcWidth_, srcHeight_, erodeDevice_, dstWidth_, dstHeight_, 5);
        dilationDevice(batchSize_, erodeDevice_, srcWidth_, srcHeight_, dilateDevice_, dstWidth_, dstHeight_, 5);
    }
    logger_->info("cuda 二值化开运算耗时:{} ms", d1.getUsedTime());
    tools::DeviceTimer d2;
    int bufSize = inputFrame->batchSize * inputFrame->imageHeight* inputFrame->imageWidth;
    unsigned char* dilateBufCpu = new unsigned char[bufSize];
    cudaMemcpy(dilateBufCpu, dilateDevice_, bufSize, cudaMemcpyDeviceToHost);
    logger_->info("拷贝耗时:{} ms", d2.getUsedTime());
    unsigned char* dilateBufCpuTemp = dilateBufCpu;
    tools::HostTimer d3;
    for (int i = 0; i < batchSize_; i++) {
        cv::Mat openingImage(inputFrame->imageHeight, inputFrame->imageWidth, CV_8UC1, dilateBufCpuTemp);
        dilateBufCpuTemp += sizeof(unsigned char) * inputFrame->imageHeight * inputFrame->imageWidth;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(openingImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::RotatedRect rect;
        int circleIndex = -1;
        for (std::size_t j = 0; j < contours.size(); j++) {
            int area = cv::contourArea(contours[j]);
            if (area > 500000 && area < 800000) {
                rect = cv::fitEllipse(contours[j]);
                circleIndex = j;
                break;
            }
        }

        if (circleIndex == -1) {
            logger_->error("找不到料");
            continue;
        }

        Circle circle;
        auto center = rect.center;
        circle.circlePoint.x = rect.center.x;
        circle.circlePoint.y = rect.center.y;
        circle.radius = std::max(rect.size.width, rect.size.height) / 2;
        circle.size.height = rect.size.height;
        circle.size.width = rect.size.width;
        circle.angle = rect.angle;
        outputframe->circles->push_back(circle);

    }
    delete[]dilateBufCpu;
    logger_->info("检测圆耗时: {} ms", d3.getUsedTime());

}

void ImageProcess::DetectMaociHuahenBatchImages::execute(BatchImageFramePtr inputFrame, BatchResultFramePtr outputframe) {
    tools::DeviceTimer d1;
    auto gpuBuffer = inputFrame->buffer;
    unsigned char* grayGpuBuf = nullptr;
    if (imageType_ == "rgb") {
        grayGpuBuf = grayDevice_;
        rgb2grayDevice(batchSize_, static_cast<unsigned char*>(gpuBuffer.get()), srcWidth_, srcHeight_, grayDevice_, dstWidth_, dstHeight_);
        binaryDevice(batchSize_, grayDevice_, srcWidth_, srcHeight_, binaryDevice_, dstWidth_, dstHeight_, inv_, thresHold1_);
        erosionDevice(batchSize_, binaryDevice_, srcWidth_, srcHeight_, erodeDevice_, dstWidth_, dstHeight_, 10);
        dilationDevice(batchSize_, erodeDevice_, srcWidth_, srcHeight_, dilateDevice_, dstWidth_, dstHeight_, 10);
    }
    else {
        grayGpuBuf = static_cast<unsigned char*>(gpuBuffer.get());
        binaryDevice(batchSize_, static_cast<unsigned char*>(gpuBuffer.get()), srcWidth_, srcHeight_, binaryDevice_, dstWidth_, dstHeight_, inv_, thresHold1_);
        erosionDevice(batchSize_, binaryDevice_, srcWidth_, srcHeight_, erodeDevice_, dstWidth_, dstHeight_, 10);
        dilationDevice(batchSize_, erodeDevice_, srcWidth_, srcHeight_, dilateDevice_, dstWidth_, dstHeight_, 10);
    }
    logger_->info("cuda 二值化开运算耗时: {} ms", d1.getUsedTime());
    tools::DeviceTimer d2;
    int bufSize = inputFrame->batchSize * inputFrame->imageHeight * inputFrame->imageWidth;
    unsigned char* grayBufCpu = new unsigned char[bufSize];
    unsigned char* binaryBufCpu = new unsigned char[bufSize];
    unsigned char* dilateBufCpu = new unsigned char[bufSize];
    cudaMemcpy(grayBufCpu, grayGpuBuf, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(binaryBufCpu, binaryDevice_, bufSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(dilateBufCpu, dilateDevice_, bufSize, cudaMemcpyDeviceToHost);
    logger_->info("拷贝耗时: {} ms", d2.getUsedTime());
    unsigned char* grayBufCpuTemp = grayBufCpu;
    unsigned char* binaryBufCpuTemp = binaryBufCpu;
    unsigned char* dilateBufCpuTemp = dilateBufCpu;
    for (int i = 0; i < batchSize_; i++) {
        cv::Mat grayImage(inputFrame->imageHeight, inputFrame->imageWidth, CV_8UC1, grayBufCpuTemp);
        cv::Mat binaryImage(inputFrame->imageHeight, inputFrame->imageWidth, CV_8UC1, binaryBufCpuTemp);
        cv::Mat openingImage(inputFrame->imageHeight, inputFrame->imageWidth, CV_8UC1, dilateBufCpuTemp);
        grayBufCpuTemp += sizeof(unsigned char) * inputFrame->imageHeight * inputFrame->imageWidth;
        binaryBufCpuTemp += sizeof(unsigned char) * inputFrame->imageHeight * inputFrame->imageWidth;
        dilateBufCpuTemp += sizeof(unsigned char) * inputFrame->imageHeight * inputFrame->imageWidth;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(openingImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::RotatedRect rect;
        int circleIndex = -1;
        cv::Mat maskEllipse;
        cv::Mat maskEllipse1;
        for (std::size_t j = 0; j < contours.size(); j++) {
            int area = cv::contourArea(contours[j]);
            if (area > 500000 && area < 800000) {
                rect = cv::fitEllipse(contours[j]);
                //rect.size.width += 5;
                //rect.size.height += 5;
                maskEllipse = cv::Mat::zeros(grayImage.size(), grayImage.type());
                cv::ellipse(maskEllipse, rect, cv::Scalar(255), -1);
                auto rect1 = rect;
                rect1.size.width -= 20;
                rect1.size.height -= 20;
                maskEllipse1 = cv::Mat::zeros(grayImage.size(), grayImage.type());
                cv::ellipse(maskEllipse1, rect1, cv::Scalar(255), -1);
                circleIndex = j;
                break;
            }
        }
        if (circleIndex == -1) {
            logger_->error("找不到料");
            continue;
        }

        tools::HostTimer d_t4;

        cv::ellipse(grayImage, rect, cv::Scalar(0, 255, 0), 2);

        Circle circle;
        auto center = rect.center;
        circle.circlePoint.x = rect.center.x;
        circle.circlePoint.y = rect.center.y;
        // TODO: 看是否有必要调整为std::min
        circle.radius = std::max(rect.size.width, rect.size.height) / 2;
        circle.size.width = rect.size.width;
        circle.size.height = rect.size.height;
        circle.angle = rect.angle;

        // 乘运算，理论上只剩下一个连通域
        cv::multiply(openingImage, maskEllipse1, openingImage);

        // 减运算，只剩下毛刺
        cv::Mat burr;
        cv::subtract(binaryImage, maskEllipse, burr);

        // 对毛刺findcontour，找到离圆心最近的点的距离约等于半径的点
        cv::findContours(burr, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (std::size_t j = 0; j < contours.size(); j++) {
            bool keep = false;
            if (contours[j].size() >= 5) {
                for (std::size_t k = 0; k < contours[j].size(); k++) {
                    Point p1({ float(contours[j][k].x), float(contours[j][k].y) });
                    Point p2({ rect.center.x, rect.center.y });
                    float distance = tools::calculateDistance(p1, p2);
                    if (distance - std::max(rect.size.width, rect.size.height) / 2 < 20) {
                        keep = true;
                        break;
                    }
                }

                if (keep == true) {
                    int area = cv::contourArea(contours[j]);
                    float distanceOuter = 0.0;
                    float distanceInner = 3000.0;
                    for (std::size_t k = 0; k < contours[j].size(); k++) {
                        Point p1({ rect.center.x , rect.center.y });
                        Point p2({ float(contours[j][k].x), float(contours[j][k].y) });
                        float distanceTemp = tools::calculateDistance(p1, p2);
                        if (distanceTemp > distanceOuter) {
                            distanceOuter = distanceTemp;
                        }
                        if (distanceTemp < distanceInner) {
                            distanceInner = distanceTemp;
                        }
                    }

                    if (area > 50 && contours[j].size() >= 5) {
                        cv::RotatedRect minRect = cv::minAreaRect(contours[j]);
                        cv::Rect boundingRect = cv::boundingRect(contours[j]);
                        auto rect1 = cv::fitEllipse(contours[j]);
                        Box box(boundingRect.tl().x, boundingRect.tl().y, boundingRect.br().x, boundingRect.br().y, 1.0, 0);
                        cv::rectangle(grayImage, boundingRect, cv::Scalar(0));
                        box.width = minRect.size.width;
                        box.height = minRect.size.height;
                        box.distance0uter = distanceOuter;
                        box.distanceInner = distanceInner;
                        auto midPoint = rect1.center;
                        auto angle = 180.0 - rect1.angle - 90.0;
                        float angle1 = 0.0;
                        if (midPoint.x - circle.circlePoint.x != 0) {
                            auto k = (-(midPoint.y - circle.circlePoint.y)) / (midPoint.x - circle.circlePoint.x);
                            angle1 = atan(k) * (180.0 / M_PI);
                        }
                        else {
                            angle1 = 90;
                        }
                        double angleDifference = tools::calculateAngleDifference(angle, angle1);
                        if (angleDifference > 70) {
                            outputframe->batchDefects->at(i).push_back(Defect("yijiao", box));
                        }
                        else {
                            outputframe->batchDefects->at(i).push_back(Defect("maoci", box));
                        }
                    }
                }
            }

        }

        logger_->info("检测毛刺溢胶用时: {} ms", d_t4.getUsedTime());
        tools::HostTimer d_t5;
        //检测划痕
        cv::Mat filterImage;
        cv::bitwise_and(grayImage, grayImage, filterImage, openingImage);
        cv::threshold(filterImage, filterImage, thresHold2_, 255, cv::THRESH_BINARY);
        cv::findContours(filterImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (std::size_t j = 0; j < contours.size(); j++) {
            if (contours[j].size() >= 5) {
                int area = cv::contourArea(contours[j]);
                if (area > 10) {
                    bool keep = false;
                    float distanceOuter = 0.0;
                    float distanceInner = 3000.0;
                    for (std::size_t k = 0; k < contours[j].size(); k++) {
                        Point p1({ rect.center.x , rect.center.y });
                        Point p2({ float(contours[j][k].x), float(contours[j][k].y) });
                        float distance = tools::calculateDistance(p1, p2);
                        if (distance > distanceOuter) {
                            distanceOuter = distance;
                        }
                        if (distance < distanceInner) {
                            distanceInner = distance;
                        }
                        if (distance < std::max(rect.size.width, rect.size.height) / 2 - 10) {
                            keep = true;
                        }
                    }
                    if (keep) {
                        cv::Rect boundingRect = cv::boundingRect(contours[j]);
                        cv::rectangle(grayImage, boundingRect, cv::Scalar(0));
                        Box box(boundingRect.tl().x, boundingRect.tl().y, boundingRect.br().x, boundingRect.br().y, 1.0, 0);
                        box.distance0uter = distanceOuter;
                        box.distanceInner = distanceInner;
                        outputframe->batchDefects->at(i).push_back(Defect("huahen", box));
                    }
                }

            }
        }
        logger_->info("检测划痕用时: {} ms", d_t5.getUsedTime());
        outputframe->circles->push_back(circle);
    }
    delete[]grayBufCpu;
    delete[]binaryBufCpu;
    delete[]dilateBufCpu;
}

void ImageProcess::DetectCornerBatchImages::execute(BatchImageFramePtr inputFrame, BatchResultFramePtr outputframe) {
    tools::DeviceTimer d1;
    auto gpuBuffer = inputFrame->buffer;
    unsigned char* grayGpuBuf = nullptr;
    if (imageType_ == "rgb") {
        rgb2grayDevice(batchSize_, static_cast<unsigned char*>(gpuBuffer.get()), srcWidth_, srcHeight_, grayDevice_, dstWidth_, dstHeight_);
        binaryDevice(batchSize_, grayDevice_, srcWidth_, srcHeight_, binaryDevice_, dstWidth_, dstHeight_, inv_, thresHold1_);
    }
    else {
        binaryDevice(batchSize_, static_cast<unsigned char*>(gpuBuffer.get()), srcWidth_, srcHeight_, binaryDevice_, dstWidth_, dstHeight_, inv_, thresHold1_);
    }
    logger_->info("cuda 二值化运算耗时: {} ms", d1.getUsedTime());

    tools::DeviceTimer d2;
    int bufSize = inputFrame->batchSize * inputFrame->imageHeight * inputFrame->imageWidth;
    unsigned char* bufCpu = new unsigned char[bufSize];
    cudaMemcpy(bufCpu, binaryDevice_, bufSize, cudaMemcpyDeviceToHost);
    logger_->info("拷贝耗时: {} ms", d2.getUsedTime());
    unsigned char* bufCpuTemp = bufCpu;
    tools::HostTimer d3;
    for (int i = 0; i < batchSize_; i++) {
        cv::Mat image(inputFrame->imageHeight, inputFrame->imageWidth, CV_8UC1, bufCpuTemp);
        bufCpuTemp += sizeof(unsigned char) * inputFrame->imageHeight * inputFrame->imageWidth;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(image, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        for (std::size_t j = 0; j < contours.size(); j++) {
            cv::Rect boundingRect = cv::boundingRect(contours[j]);
            auto left = boundingRect.br().x;
            auto top = boundingRect.br().y;
            auto right = left + 2;
            auto bottom = top + 2;
            Box box(left, top, right, bottom, 1.0, 0);
            outputframe->batchDefects->at(i).push_back(Defect("corner", box));
        }
    }
    delete[]bufCpu;
    logger_->info("检测角点耗时: {} ms", d3.getUsedTime());

}