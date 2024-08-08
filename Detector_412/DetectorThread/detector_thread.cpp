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
    if (copyImageToCudaThread_.joinable()) {
        copyImageToCudaThreadShouldExit_ = true;
        copyImageToCudaThread_.join();
    }

    if (cropImageThread_.joinable()) {
        cropImageThreadShouldExit_ = true;
        cropImageThread_.join();
    }
    if (detectThread_.joinable()) {
        detectThreadShouldExit_ = true;
        detectThread_.join();
    }

    if (postprocessThread_.joinable()) {
        postprocessThreadShouldExit_ = true;
        postprocessThread_.join();
    }

    yolo_.reset();
}


bool DetectorThread::push(ImageFrame& frame) {
    if (!imageQueue_->Enqueue(std::make_shared<ImageFrame>(frame))) {
        std::cout << "image queue full!" << std::endl;
    }
    else {
        std::cout << "imageFrame add!, size: " << imageQueue_->size() << std::endl;
    }
    return true;
}

bool DetectorThread::get(ResultFrame& frame, std::string& uuid) {
    std::unique_lock<std::mutex> lock(resultFrameMapMutex_);
    resultFrameMapCV_.wait(lock, [&] {return resultFrameMap_.find(uuid) != resultFrameMap_.end(); });
    frame = resultFrameMap_[uuid];
    resultFrameMap_.erase(uuid);
    return true;
}

void DetectorThread::registerTraditionFun(std::function<void(std::vector<cv::Mat>&, BatchResultFramePtr)> cb) {
    traditionalDetectBatchImagesFun_ = cb;
}


void DetectorThread::copyImageToCudaThread() {
    void* data = nullptr;
    void* dataCpu = nullptr;
    unsigned char* dataPoint = nullptr;
    unsigned char* dataPointCpu = nullptr;
    int frameCount = 0;
    std::vector<ImagePos> imagePos;
    std::vector<std::string> batchuuid;
    while (!copyImageToCudaThreadShouldExit_) {
        auto imageFrame = imageQueue_->Dequeue();
        if (!imageFrame) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }
        auto uuid = (*imageFrame)->uuid;
        auto imageBuf = (*imageFrame)->buffer;
        auto imageWidth = (*imageFrame)->imageWidth;
        auto imageHeight = (*imageFrame)->imageHeight;
        auto channelNum = (*imageFrame)->channelNum;
        auto bufferSize = imageWidth * imageHeight * channelNum;

        if (frameCount % batchSize_ == 0) {
            batchuuid.clear();
            imagePos.clear();
            CHECK(cudaMalloc(&data, batchSize_ * bufferSize));
            dataCpu = new unsigned char[batchSize_ * bufferSize];
            dataPointCpu = static_cast<unsigned char*>(dataCpu);
            dataPoint = static_cast<unsigned char*>(data);
        }
        memcpy(dataPointCpu, imageBuf.get(), bufferSize);
        cudaMemcpy(dataPoint, imageBuf.get(), bufferSize, cudaMemcpyHostToDevice);
        dataPoint += bufferSize;
        dataPointCpu += bufferSize;
        batchuuid.push_back(uuid);
        imagePos.push_back({0,0,true});
        if (frameCount % batchSize_ == batchSize_ - 1) {
            BatchImageFramePtr batchImageFrame(new BatchImageFrame({
                    std::shared_ptr<void>(data, [](void* p) {CHECK(cudaFree(p)); }),
                    std::shared_ptr<void>(dataCpu, [](void* p) {delete[]p; }),
                    std::make_shared<std::vector<ImagePos>>(imagePos),
                    batchuuid,
                    imageWidth,
                    imageHeight,
                    channelNum,
                    batchSize_
                }));

            if (!batchImageQueue_->Enqueue(batchImageFrame)) {
                std::cout << "batchImageQueue full!" << std::endl;
            }
            else {
                std::cout << "batchImageQueue add. new size: " << batchImageQueue_->size() << std::endl;
            }
            data = nullptr;
            dataPoint = nullptr;
        }
        frameCount++;
    }
}

#ifdef _DEBUG
void getCVBatchImages(unsigned char*& batchImagesHost, std::vector<cv::Mat>& batchImages, BatchImageFramePtr frame) {
    auto buf = frame->buffer;
    auto batchSize = frame->batchSize;
    auto imageWidth = frame->imageWidth;
    auto imageHeight = frame->imageHeight;
    auto imageChannel = frame->channelNum;
    
    batchImagesHost = new unsigned char[batchSize * sizeof(unsigned char) * imageHeight * imageWidth * imageChannel];;
    CHECK(cudaMemcpy(batchImagesHost, buf.get(), batchSize * sizeof(unsigned char) * imageHeight * imageWidth* imageChannel, cudaMemcpyDeviceToHost));
    unsigned char* point = static_cast<unsigned char*>(batchImagesHost);
    auto cvImageType = imageChannel == 1?CV_8UC1:CV_8UC3;
    for (int i = 0; i < batchSize; i++) {
        cv::Mat image(imageHeight, imageWidth, cvImageType, point);
        batchImages.push_back(image);
        point += sizeof(unsigned char) * imageHeight * imageWidth * imageChannel;
    }
    return;
}
#endif // _DEBUG

void getCVBatchImages(std::vector<cv::Mat>& batchImages, BatchImageFramePtr frame) {
    auto bufCpu = frame->bufferCpu;
    auto batchSize = frame->batchSize;
    auto imageWidth = frame->imageWidth;
    auto imageHeight = frame->imageHeight;
    auto imageChannel = frame->channelNum;

    //batchImagesHost = new unsigned char[batchSize * sizeof(unsigned char) * imageHeight * imageWidth * imageChannel];;
    //CHECK(cudaMemcpy(batchImagesHost, buf.get(), batchSize * sizeof(unsigned char) * imageHeight * imageWidth * imageChannel, cudaMemcpyDeviceToHost));
    unsigned char* point = static_cast<unsigned char*>(bufCpu.get());
    auto cvImageType = imageChannel == 1 ? CV_8UC1 : CV_8UC3;
    for (int i = 0; i < batchSize; i++) {
        cv::Mat image(imageHeight, imageWidth, cvImageType, point);
        batchImages.push_back(image);
        point += sizeof(unsigned char) * imageHeight * imageWidth * imageChannel;
    }
    return;
}


float calculateDistance(const Point& p1, const Point& p2) {
    float xDiff = p2.x - p1.x;
    float yDiff = p2.y - p1.y;
    return sqrt(xDiff * xDiff + yDiff * yDiff);
}

void shrinkFilter(Defect& defect, Circle& circle, float& shrink, float& shrinkRatio,bool& keep) {
    Point boxCenter;
    boxCenter.x = (defect.box.left + defect.box.right) / 2;
    boxCenter.y = (defect.box.top + defect.box.bottom) / 2;
    /*auto height = defect.box.bottom - defect.box.top;
    auto width = defect.box.right - defect.box.left;
    auto length = height > width ? height : width;*/
    float distance = calculateDistance(circle.circlePoint, boxCenter);
    if (distance > circle.radius - shrinkRatio * shrink) {
        keep = false;
    }
    else {
        keep = true;
    }
}

bool compare(const std::string& condition, float a, float shrinkRatio=1.0) {
    std::regex pattern(R"((>=|<=|>|<|=|!=)(-?\d+))");
    std::smatch matches;

    if (std::regex_match(condition, matches, pattern)) {
        if (matches.size() != 3) {
            throw std::invalid_argument("Invalid condition format");
        }

        std::string op = matches[1];
        float value = shrinkRatio *std::stof(matches[2]);

        if (op == ">=") {
            return a >= value;
        }
        else if (op == "<=") {
            return a <= value;
        }
        else if (op == ">") {
            return a > value;
        }
        else if (op == "<") {
            return a < value;
        }
        else if (op == "=") {
            return a == value;
        }
        else if (op == "!=") {
            return a != value;
        }
        else {
            throw std::invalid_argument("Unknown operator");
        }
    }
    else {
        throw std::invalid_argument("Invalid condition format");
    }
}

void DetectorThread::regularzation(ResultFrame &frame, cv::Mat& image, Circle& circle) {
    auto defect = frame.defects;
    std::stringstream NGStateMent;
    for (auto it = defect->begin(); it != defect->end();) {
        auto defectName = it->defectName;
        if (!configManager_) {
            std::cout << "configManager未初始化！" << std::endl;
        }
        // TODO ：config改为私有成员变量
        auto config = configManager_->getConfig();
        auto defectFilter = config["defect_filter"];
        if (!defectFilter[defectName]) {
            std::cout << "规则未找到该缺陷表述：" << defectName << std::endl;
            ++it;
            continue;
        }
        else {
            float shrink = 0.0;
            float shrinkRatio = 0.4;
            if (defectFilter[defectName]["shrink"]) {
                shrink = defectFilter[defectName]["shrink"].as<float>();
            }
            if (config["shrinkratio"]) {
                shrinkRatio = config["shrinkratio"].as<float>();
            }

            // TODO 考虑用frame里拿坐标圆心，半径，小圆半径
            Point centerOfCircle;
            centerOfCircle.x = 640.0;
            centerOfCircle.y = 632.0;
            int radius = 413;
            int radiusSmall = 300;

            if (config["radiussmall"]) {
                radiusSmall = config["radiussmall"].as<int>();
            }

            bool keep = true;
            shrinkFilter(*it, circle, shrink, shrinkRatio, keep);
            if (!keep) {
                it = defect->erase(it);
                continue;
            }
            else {
                if (defectFilter[defectName]["judge"] && defectFilter[defectName]["judge"].IsSequence()) {
                    for (const YAML::Node& item : defectFilter[defectName]["judge"]) {
                        float objValue = 0.0;
                        if (!item["obj"]) {
                            std::cout << "配置文件里有judge，找不到obj" << std::endl;
                            continue;
                        }
                        else {
                            auto objFocus = item["obj"].as<std::string>();
                            if (objFocus == "thickness") {
                                objValue = it->box.right - it->box.left;
                            }
                            else if (objFocus == "length" || objFocus == "width") {
                                auto height = it->box.bottom - it->box.top;
                                auto width = it->box.right - it->box.left;
                                if (objFocus == "length") {
                                    objValue = height > width ? height : width;
                                }
                                else {
                                    objValue = height > width ? width : height;
                                }
                            }

                            else if (objFocus == "area_in_circle" || objFocus == "area_out_circle" || objFocus == "area") {
                                Point boxCenter;
                                boxCenter.x = (it->box.left + it->box.right) / 2;
                                boxCenter.y = (it->box.top + it->box.bottom) / 2;
                                auto area = (it->box.right - it->box.left) * (it->box.bottom - it->box.top);
                                float distance = calculateDistance(centerOfCircle, boxCenter);
                                //TODO 补充小圆半径
                                if (objFocus == "area_in_circle") {
                                    if (distance < radiusSmall) {
                                        objValue = area;
                                    }
                                    else {
                                        continue;
                                    }
                                }
                                else if (objFocus == "area_out_circle") {
                                    if (distance >= radiusSmall && distance < radius) {
                                        objValue = area;
                                    }
                                    else {
                                        continue;
                                    }
                                }
                                else {
                                    objValue = area;
                                }
                            }

                        }
                        if (!item["NG"]) {
                            std::cout << "找不到NG标准" << std::endl;
                            continue;
                        }
                        else {
                            auto NGStandard = item["NG"].as<std::string>();
                            if (compare(NGStandard, objValue, shrinkRatio)) {
                                frame.NG = true;
                                NGStateMent << "The " << item["obj"].as<std::string>() << " of " << defectName << " " << NGStandard << ".";
                                NGStateMent << "The " << item["obj"].as<std::string>() << " value is:" << objValue << std::endl;

                            }
                        }
                    }
                }
                ++it;
            }
        }

    }
    frame.NGStateMent = NGStateMent.str();

}

double normalizeAngle(double angle) {
    // 将角度规范化到 -180° 到 180° 之间
    while (angle > 180.0) angle -= 360.0;
    while (angle < -180.0) angle += 360.0;
    return angle;
}

double calculateAngleDifference(double angle1, double angle2) {
    // 计算两角度之间的差值
    double difference = normalizeAngle(angle1 - angle2);
    // 将差值转换为正数夹角
    difference = fabs(difference);
    // 如果夹角大于 90°，则用 180° - 差值
    if (difference > 90.0) {
        difference = 180.0 - difference;
    }
    return difference;
}

void DetectorThread::detectMaociBatchImages(std::vector<cv::Mat>& images, BatchResultFramePtr outputframe) {
    for (std::size_t i = 0; i < images.size(); i++) {
        cv::Mat imageThresh;
        auto image = images[i];
        std::vector<Defect> defect;
        if (image.type() != CV_8UC1) {
            if (image.type() == CV_8UC3) {
                cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
            }
            else {
                std::cout << "wrong type, tradition detect failed!" << std::endl;
                continue;
            }
        }
        // 模糊
        cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
        // 二值化
        cv::threshold(image, imageThresh, 188, 255, cv::THRESH_BINARY_INV);
        cv::Mat labels, stats, centroids;
        //https://blog.csdn.net/qq_43199575/article/details/133810085
        int numComponents = cv::connectedComponentsWithStats(imageThresh, labels, stats, centroids);
        cv::Mat Mask;
        // 筛面积
        for (int i = 1; i < numComponents; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            Mask = (labels == i);

            if (area > 500000 && area < 800000) {
                Mask = (labels == i);
                break;
            }

        }
        // 开运算
        cv::Mat element;
        //element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
        element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        cv::Mat openingCircle;
        cv::morphologyEx(Mask, openingCircle, cv::MORPH_OPEN, element);
        
        // 检测椭圆
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(openingCircle, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.size()) {
            std::cout << "detect openning circle failed" << std::endl;
            continue;
        }
        cv::RotatedRect rect =  cv::fitEllipse(contours[0]);
        auto rect1 = rect;

        rect1.size.width += 10;
        rect1.size.height += 10;
        cv::ellipse(image, rect1, cv::Scalar(0, 255, 0), 2);

        // 制造椭圆掩膜
        cv::Mat maskEllipse = cv::Mat::zeros(image.size(), image.type());
        cv::ellipse(maskEllipse, rect1, cv::Scalar(255), -1);

        // 乘运算
        cv::multiply(openingCircle, maskEllipse, openingCircle);


        Circle circle;
        auto center = rect.center;
        circle.circlePoint.x = rect.center.x;
        circle.circlePoint.y = rect.center.y;
        circle.radius = std::max(rect.size.width, rect.size.height)/2;
        

        // 减运算
        cv::Mat burr;
        cv::subtract(Mask, openingCircle, burr);


        // 面积判断
        numComponents = cv::connectedComponentsWithStats(burr, labels, stats, centroids);
        for (int j = 1; j < numComponents; j++) {
            int area = stats.at<int>(j, cv::CC_STAT_AREA);
            if (area > 200) {
                cv::Mat maskNow = (labels == j);
                cv::findContours(maskNow, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                double perimeter = cv::arcLength(contours[0], true);
                //double ratio = perimeter / area;
                auto left = stats.at<int>(j, cv::CC_STAT_LEFT);
                auto top = stats.at<int>(j, cv::CC_STAT_TOP);
                auto right = stats.at<int>(j, cv::CC_STAT_LEFT) + stats.at<int>(j, cv::CC_STAT_WIDTH);
                auto bottom = stats.at<int>(j, cv::CC_STAT_TOP) + stats.at<int>(j, cv::CC_STAT_HEIGHT);
                Box box(left, top, right, bottom, 1.0, 0);
                if (contours[0].size() >= 5) {
                    rect = cv::fitEllipse(contours[0]);
                    //cv::ellipse(image, rect, cv::Scalar(0, 255, 0), 2);
                    auto midPoint = rect.center;
                    auto angle = 180.0 - rect.angle - 90.0;
                    float angle1 = 0.0;
                    if (midPoint.x - circle.circlePoint.x != 0) {
                        auto k = (-(midPoint.y - circle.circlePoint.y)) / (midPoint.x - circle.circlePoint.x);
                        angle1 = atan(k) * (180.0 / M_PI);
                    }
                    else {
                        angle1 = 90;
                    }
                    double angleDifference = calculateAngleDifference(angle, angle1);
                    if (angleDifference > 70 || area > 2000 ) {
                        outputframe->batchDefects->at(i).push_back(Defect("yijiao", box));
                    }
                    else {
                        outputframe->batchDefects->at(i).push_back(Defect("maoci", box));
                    }
                }
            }

        }
        
        outputframe->circles->push_back(circle);
    }
}


void DetectorThread::detectThread() {
    while (!detectThreadShouldExit_) {
        auto batchImage = batchImageQueue_->Dequeue();
        if (!batchImage) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto batchuuid = (*batchImage)->batchuuid;
        auto buffer = (*batchImage)->buffer;
        auto bufferCpu = (*batchImage)->bufferCpu;
        auto imagesPos = (*batchImage)->imagesPos;
        yolo_->setInputData(buffer);
        utils::DeviceTimer d_t1; yolo_->preprocess();  float t1 = d_t1.getUsedTime();
        utils::DeviceTimer d_t2; yolo_->infer();       float t2 = d_t2.getUsedTime();
        utils::DeviceTimer d_t3; yolo_->postprocess(static_cast<size_t>(batchSize_)); float t3 = d_t3.getUsedTime();
        sample::gLogInfo << "preprocess time = " << t1 / batchSize_ << "; "
            "infer time = " << t2 / batchSize_ << "; "
            "postprocess time = " << t3 / batchSize_ << std::endl;

#ifdef _DEBUG
        std::vector<cv::Mat> batchImages;
        unsigned char* batchImagesHost = nullptr;
        getCVBatchImages(batchImagesHost, batchImages, (*batchImage));
        utils::draw(yolo_->getObjectss(), batchImages, batchuuid, true);
        delete[]batchImagesHost;

#endif
        std::vector<cv::Mat> batchCpuImages;
        getCVBatchImages(batchCpuImages, (*batchImage));
        auto batchDefects = yolo_->getObjectss();
        BatchResultFramePtr outputFrame(new BatchResultFrame({
            std::make_shared<std::vector<std::vector<Defect>>>(yolo_->getObjectss()),
            imagesPos,
            std::make_shared<std::vector<Circle>>(),
            batchuuid
            }));

        traditionalDetectBatchImagesFun_(batchCpuImages, outputFrame);

        if (!batchResultQueue_->Enqueue(outputFrame)) {
            //sample::gLogInfo << "queue full" << std::endl;
            std::cout << "queue full" << std::endl;
        }
        else {
            //sample::gLogInfo << "output queue add. new size: " << batchResultQueue_->size() << std::endl;
            std::cout << "output queue add. new size: " << batchResultQueue_->size() << std::endl;

        }
        yolo_->reset();
    }

}

void DetectorThread::postprocessThread() {
    while (!postprocessThreadShouldExit_) {
        auto batchResultFrame = batchResultQueue_->Dequeue();
        if (!batchResultFrame) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }
        auto batchuuid = (*batchResultFrame)->batchuuid;
        auto batchDefects = (*batchResultFrame)->batchDefects;
        auto imagesPos = (*batchResultFrame)->imagesPos;

        for (std::size_t i = 0; i < batchDefects->size(); i++) {
            auto uuid = (*batchResultFrame)->batchuuid[i];
            auto rowBias = (*batchResultFrame)->imagesPos->at(i).rowBias;
            auto colBias = (*batchResultFrame)->imagesPos->at(i).colBias;
            auto isLast = (*batchResultFrame)->imagesPos->at(i).isLast;
            auto defects = batchDefects->at(i);
            auto circle = (*batchResultFrame)->circles->at(i);
            for (std::size_t i = 0; i < defects.size(); i++) {
                defects[i].box.addBias(rowBias, colBias);
            }
            
            if (isLast) {
                ResultFrame resultFrame({ std::make_shared<std::vector<Defect>>(defects), uuid, false, ""});
                cv::Mat image;
                regularzation(resultFrame, image, circle);

                std::lock_guard<std::mutex> lock(resultFrameMapMutex_);
                //resultFrameMap_[uuid] = ResultFrame({ std::make_shared<std::vector<Defect>>(defects), uuid} );
                resultFrameMap_[uuid] = resultFrame;
                resultFrameMapCV_.notify_all();
            }
        }

    }
}

bool DetectorThread::createDetector(std::string& configPath) {
    configManager_ = ConfigManager::GetInstance(configPath);
    auto node = configManager_->getConfig();
    batchSize_ = node["object_detecion"]["batchsize"].as<int>();
    auto objectDetectorUse = node["object_detecion"]["objectdetector"].as<std::string>();
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
    if (!createDetector(configPath)) {
        std::cout << "create detector failed!" << std::endl;
        return false;
    }

    assert(!copyImageToCudaThread_.joinable());
    copyImageToCudaThread_ = std::thread(&DetectorThread::copyImageToCudaThread, this);

    assert(!detectThread_.joinable());
    detectThread_ = std::thread(&DetectorThread::detectThread, this);

    assert(!postprocessThread_.joinable());
    postprocessThread_ = std::thread(&DetectorThread::postprocessThread, this);

    this->registerTraditionFun(std::bind(&DetectorThread::detectMaociBatchImages, this, std::placeholders::_1, std::placeholders::_2));
    return true;
}
