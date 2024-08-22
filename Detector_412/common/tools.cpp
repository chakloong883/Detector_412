#include "tools.h"
#include <iostream>
#include <regex>
#include<cuda_runtime_api.h>
#include "../yolo/kernel_function.h"


namespace tools {
    void getCVBatchImages(std::vector<cv::Mat>& batchImages, BatchImageFramePtr frame) {
        auto bufCpu = frame->bufferCpu;
        auto batchSize = frame->batchSize;
        auto imageWidth = frame->imageWidth;
        auto imageHeight = frame->imageHeight;
        auto imageChannel = frame->channelNum;
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


    double normalizeAngle(double angle) {
        // 将角度规范化到 -180° 到 180° 之间
        while (angle > 180.0) angle -= 360.0;
        while (angle < -180.0) angle += 360.0;
        return angle;
    }

    double calculateAngleDifference(double angle1, double angle2) {
        // 计算两角度之间的差值
        double difference = tools::normalizeAngle(angle1 - angle2);
        // 将差值转换为正数夹角
        difference = fabs(difference);
        // 如果夹角大于 90°，则用 180° - 差值
        if (difference > 90.0) {
            difference = 180.0 - difference;
        }
        return difference;
    }

    void shrinkFilter(Defect& defect, Circle& circle, float& shrink, float& shrinkRatio, bool& keep, float& distanceBias) {
        Point p1, p2, p3, p4, p5;
        p1.x = defect.box.left;
        p1.y = defect.box.top;
        p2.x = defect.box.right;
        p2.y = defect.box.top;
        p3.x = defect.box.right;
        p3.y = defect.box.bottom;
        p4.x = defect.box.left;
        p4.y = defect.box.bottom;
        p5.x = (defect.box.left + defect.box.right) / 2;
        p5.y = (defect.box.top + defect.box.bottom) / 2;

        float distance1 = calculateDistance(circle.circlePoint, p1);
        float distance2 = calculateDistance(circle.circlePoint, p2);
        float distance3 = calculateDistance(circle.circlePoint, p3);
        float distance4 = calculateDistance(circle.circlePoint, p4);
        float distance5 = calculateDistance(circle.circlePoint, p5);
        float distance = 0.0;
        /*
        * 分两种情况。
        * 若最外延(distance0uter)和最内延(distanceInner)数值不为零，则用distance0uter和distanceInner计算是否需要保留
        * 若最外延(distance0uter)和最内延(distanceInner)数值为零，则用矩形框的四个点计算是否需要保留
        */

        if (defect.box.distance0uter != 0 && defect.box.distanceInner != 0) {
            if (shrink > 0) {
                distance = defect.box.distanceInner;
                if (distance > circle.radius - shrinkRatio * shrink) {
                    keep = false;
                }
                else {
                    distanceBias = (circle.radius - distance) / shrinkRatio;
                    keep = true;
                }
            }
            else if (shrink < 0) {
                distance = defect.box.distance0uter;
                if (distance < circle.radius - shrinkRatio * shrink) {
                    keep = false;
                }
                else {
                    distanceBias = (distance - circle.radius) / shrinkRatio;
                    keep = true;
                }
            }
            // 内缩等于0的情况默认都保留
            else {
                keep = true;
            }
        }
        else {
            if (shrink > 0) {
                distance = std::min({ distance1, distance2, distance3, distance4 });
                if (distance > circle.radius - shrinkRatio * shrink) {
                    keep = false;
                }
                else {
                    distanceBias = (circle.radius - distance) / shrinkRatio;
                    keep = true;
                }
            }
            else if (shrink < 0) {
                distance = std::max({ distance1, distance2, distance3, distance4 });
                if (distance < circle.radius - shrinkRatio * shrink) {
                    keep = false;
                }
                else {
                    distanceBias = (distance - circle.radius) / shrinkRatio;
                    keep = true;
                }
            }
            // 内缩等于0的情况默认都保留
            else {
                keep = true;
            }
        }
    }

    bool compare(const std::string& condition, float a) {
        // 改进后的正则表达式，支持整数和浮点数
        std::regex pattern(R"((>=|<=|>|<|=|!=)(-?\d+(\.\d+)?))");
        std::smatch matches;

        if (std::regex_search(condition, matches, pattern)) {
            if (matches.size() < 3) {
                throw std::invalid_argument("Invalid condition format");
            }

            std::string op = matches[1];
            float value = std::stof(matches[2]);

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

    //bool compare(const std::string& condition, float a) {
    //    //std::regex pattern(R"((>=|<=|>|<|=|!=)(-?\d+))");
    //    std::regex pattern(R"((>=|<=|>|<|=|!=)(-?\d+(\.\d+)?))");
    //    std::smatch matches;

    //    if (std::regex_match(condition, matches, pattern)) {
    //        if (matches.size() != 3) {
    //            throw std::invalid_argument("Invalid condition format");
    //        }

    //        std::string op = matches[1];
    //        float value = std::stof(matches[2]);

    //        if (op == ">=") {
    //            return a >= value;
    //        }
    //        else if (op == "<=") {
    //            return a <= value;
    //        }
    //        else if (op == ">") {
    //            return a > value;
    //        }
    //        else if (op == "<") {
    //            return a < value;
    //        }
    //        else if (op == "=") {
    //            return a == value;
    //        }
    //        else if (op == "!=") {
    //            return a != value;
    //        }
    //        else {
    //            throw std::invalid_argument("Unknown operator");
    //        }
    //    }
    //    else {
    //        throw std::invalid_argument("Invalid condition format");
    //    }
    //}




    void regularzation(ResultFrame& frame, Circle& circle, YAML::Node& config) {
        auto defect = frame.defects;
        std::stringstream NGStateMent;
        int num = 0;
        for (auto it = defect->begin(); it != defect->end();) {
            auto defectName = it->defectName;
            auto defectFilter = config["defect_filter"];
            if (!defectFilter[defectName]) {
                std::cout << "规则未找到该缺陷表述：" << defectName << std::endl;
                //TODO 消除注释
                //++it;
                //TODO 待删除
                it = defect->erase(it);
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
                float distance = 0.0;
                shrinkFilter(*it, circle, shrink, shrinkRatio, keep, distance);
                if (!keep) {
                    it = defect->erase(it);
                    continue;
                }
                else {
                    keep = false;
                    if (defectFilter[defectName]["judge"] && defectFilter[defectName]["judge"].IsSequence()) {
                        for (const YAML::Node& item : defectFilter[defectName]["judge"]) {
                            float objValue = 0.0;
                            std::string objFocus = "";
                            if (!item["obj"]) {
                                std::cout << "配置文件里有judge，找不到obj" << std::endl;
                                continue;
                            }
                            else {
                                objFocus = item["obj"].as<std::string>();
                                if (objFocus == "thickness") {
                                    objValue = (it->box.right - it->box.left)/ shrinkRatio;
                                }
                                else if (objFocus == "length" || objFocus == "width" || objFocus == "diagonal_length") {
                                    float height = 0.0, width = 0.0;
                                    if (it->box.width != 0 && it->box.height != 0) {
                                        height = it->box.height;
                                        width = it->box.width;
                                    }
                                    else {
                                        height = it->box.bottom - it->box.top;
                                        width = it->box.right - it->box.left;
                                    }

                                    if (objFocus == "length") {
                                        objValue = (height > width ? height : width)/ shrinkRatio;
                                    }
                                    else if (objFocus == "width") {
                                        objValue = (height > width ? width : height)/ shrinkRatio;
                                    }
                                    else {
                                        objValue = (std::sqrt(std::pow(height, 2) + std::pow(width, 2)))/ shrinkRatio;
                                    }
                                }

                                else if (objFocus == "area_in_circle" || objFocus == "area_out_circle" || objFocus == "area") {
                                    Point boxCenter;
                                    boxCenter.x = (it->box.left + it->box.right) / 2;
                                    boxCenter.y = (it->box.top + it->box.bottom) / 2;
                                    auto area = (it->box.right - it->box.left) * (it->box.bottom - it->box.top);
                                    float distance = calculateDistance(circle.circlePoint, boxCenter);
                                    //TODO 补充小圆半径
                                    if (objFocus == "area_in_circle") {
                                        if (distance < radiusSmall) {
                                            objValue = area / std::pow(shrinkRatio, 2);
                                        }
                                        else {
                                            continue;
                                        }
                                    }
                                    else if (objFocus == "area_out_circle") {
                                        if (distance >= radiusSmall && distance < circle.radius) {
                                            objValue = area / std::pow(shrinkRatio, 2);
                                        }
                                        else {
                                            continue;
                                        }
                                    }
                                    else {
                                        objValue = area / std::pow(shrinkRatio, 2);
                                    }
                                }
                                else if (objFocus == "distance_bias") {
                                    objValue = distance / shrinkRatio;
                                }
                                else if (objFocus == "confidence") {
                                    objValue = it->box.confidence;
                                }

                            }
                            if (!item["NG"]) {
                                std::cout << "找不到NG标准" << std::endl;
                                continue;
                            }
                            else {
                                auto NGStandard = item["NG"].as<std::string>();
                                if (compare(NGStandard, objValue)) {
                                    keep = true;
                                    frame.NG = true;
                                    NGStateMent << "The " << item["obj"].as<std::string>() << " of " << "the " << num << "th " << defectName << " " << NGStandard << ".";
                                    NGStateMent << "The " << item["obj"].as<std::string>() << " value is:" << objValue << std::endl;
                                    num += 1;
                                    it->objFocus = objFocus;
                                    it->objValue = objValue;
                                    // TODO: 考虑放上break
                                }
                                else {
                                    continue;
                                }
                            }
                        }
                    }
                    if (keep == false) {
                        it = defect->erase(it);
                        continue;
                    }
                    ++it;
                }
            }

        }
        frame.NGStateMent = NGStateMent.str();
    }

    bool CopyImageToCuda::execute(){
        auto imageFrame = inputQueue_->Dequeue();
        if (!imageFrame) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            return false;
        }
        auto start1 = std::chrono::high_resolution_clock::now();

        auto uuid = (*imageFrame)->uuid;
        auto imageBuf = (*imageFrame)->buffer;
        auto imageWidth = (*imageFrame)->imageWidth;
        auto imageHeight = (*imageFrame)->imageHeight;
        auto channelNum = (*imageFrame)->channelNum;
        auto bufferSize = imageWidth * imageHeight * channelNum;

        if (frameCount_ % batchSize_ == 0) {
            batchuuid_.clear();
            imagePos_.clear();
            CHECK(cudaMalloc(&data_, batchSize_ * bufferSize));
            dataCpu_ = new unsigned char[batchSize_ * bufferSize];
            dataPointCpu_ = static_cast<unsigned char*>(dataCpu_);
            dataPoint_ = static_cast<unsigned char*>(data_);
        }
        memcpy(dataPointCpu_, imageBuf.get(), bufferSize);
        cudaMemcpy(dataPoint_, imageBuf.get(), bufferSize, cudaMemcpyHostToDevice);
        dataPoint_ += bufferSize;
        dataPointCpu_ += bufferSize;
        batchuuid_.push_back(uuid);
        imagePos_.push_back({ 0,0,true });
        if (frameCount_ % batchSize_ == batchSize_ - 1) {
            BatchImageFramePtr batchImageFrame(new BatchImageFrame({
                    std::shared_ptr<void>(data_, [](void* p) {CHECK(cudaFree(p)); }),
                    std::shared_ptr<void>(dataCpu_, [](void* p) {delete[]p; }),
                    std::make_shared<std::vector<ImagePos>>(imagePos_),
                    batchuuid_,
                    imageWidth,
                    imageHeight,
                    channelNum,
                    batchSize_
                }));

            if (!outputQueue_->Enqueue(batchImageFrame)) {
                std::cout << "batchImageQueue full!" << std::endl;
            }
            else {
                std::cout << "batchImageQueue add. new size: " << outputQueue_->size() << std::endl;
            }
            data_ = nullptr;
            dataPoint_ = nullptr;
            frameCount_ = 0;

            auto start2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = start2 - start1;
            std::cout << "拷贝线程耗时: " << elapsed.count() << " 毫秒" << std::endl;
            return true;
        }
        else {
            frameCount_++;
            return false;
        }
    }


    // 计算两点之间的欧几里得距离
    double distance(const Point& p1, const Point& p2) {
        return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    }

    // 计算点到线段的最短距离
    double pointToSegmentDistance(const Point& p, const Point& v, const Point& w) {
        double l2 = distance(v, w) * distance(v, w);  // 线段长度的平方
        if (l2 == 0.0) return distance(p, v);  // v == w，线段退化为一个点

        // 计算投影的位置
        double t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
        t = std::max(0.0, std::min(1.0, t));

        Point projection = { v.x + t * (w.x - v.x), v.y + t * (w.y - v.y) };
        return distance(p, projection);
    }

    // 判断一个点是否在多边形内部
    bool isPointInPolygon(const Point& point, const std::vector<Point>& contour) {
        int n = contour.size();
        bool inside = false;

        // 遍历多边形的每一条边
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double xi = contour[i].x, yi = contour[i].y;
            double xj = contour[j].x, yj = contour[j].y;

            // 检查射线与边是否相交
            bool intersect = ((yi > point.y) != (yj > point.y)) &&
                (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
            if (intersect) {
                inside = !inside;
            }
        }

        return inside;
    }

    // 计算点到多边形轮廓的最短距离
    double shortestDistanceToContour(const Point& point, const std::vector<Point>& contour) {
        double minDistance = std::numeric_limits<double>::infinity();

        int n = contour.size();
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double dist = pointToSegmentDistance(point, contour[j], contour[i]);
            if (dist < minDistance) {
                minDistance = dist;
            }
        }

        return minDistance;
    }
};

