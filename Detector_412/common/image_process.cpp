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

void ImageProcess::detectMaociBatchImages(std::vector<cv::Mat>& images, BatchResultFramePtr outputframe, int thresholdValue1, int thresholdValue2, bool inv) {
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
        // 二值化
        auto start1 = std::chrono::high_resolution_clock::now();
        
        auto type = cv::ThresholdTypes::THRESH_BINARY;
        if (inv == true) {
            type = cv::ThresholdTypes::THRESH_BINARY_INV;
        }
        cv::threshold(image, imageThresh, thresholdValue1, 255, type);
        auto start2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = start2 - start1;
        std::cout << "二值化耗时: " << elapsed.count() << " 毫秒" << std::endl;

        // https://blog.csdn.net/qq_43199575/article/details/133810085
        // cuda加速：https://cloud.tencent.com/developer/article/1525799
        // cuda加速 https://www.bilibili.com/video/av71643385/?vd_source=8bc4fc0368edb389591ed834a5d0e2eb
        // cuda加速用例：https://github.com/opencv/opencv/tree/4.x/samples/gpu/
        // cudaapi: https://docs.opencv.org/4.10.0/d1/d1a/namespacecv_1_1cuda.html

        // 开运算
        cv::Mat element;
        element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat openingCircle;
        cv::morphologyEx(imageThresh, openingCircle, cv::MORPH_OPEN, element);

        // 拟合椭圆
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(openingCircle, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::RotatedRect rect;
        int circleIndex = -1;
        cv::Mat maskEllipse;
        cv::Mat maskEllipse1;
        for (std::size_t j = 0; j < contours.size(); j++) {
            int area = cv::contourArea(contours[j]);
            if (area > 500000 && area < 800000) {
                rect = cv::fitEllipse(contours[j]);
                rect.size.width += 5;
                rect.size.height += 5;
                maskEllipse = cv::Mat::zeros(image.size(), image.type());
                cv::ellipse(maskEllipse, rect, cv::Scalar(255), -1);
                auto rect1 = rect;
                rect1.size.width -= 20;
                rect1.size.height -= 20;
                maskEllipse1 = cv::Mat::zeros(image.size(), image.type());
                cv::ellipse(maskEllipse1, rect1, cv::Scalar(255), -1);
                circleIndex = j;
                break;
            }
        }

        if (circleIndex == -1) {
            std::cout << "找不到料" << std::endl;
            continue;
        }
        cv::ellipse(image, rect, cv::Scalar(0, 255, 0), 2);

        Circle circle;
        auto center = rect.center;
        circle.circlePoint.x = rect.center.x;
        circle.circlePoint.y = rect.center.y;
        circle.radius = std::max(rect.size.width, rect.size.height) / 2;
        circle.size.width = rect.size.width;
        circle.size.height = rect.size.height;
        circle.angle = rect.angle;

        // 乘运算，理论上只剩下一个连通域
        cv::multiply(openingCircle, maskEllipse1, openingCircle);

        // 减运算，只剩下毛刺
        cv::Mat burr;
        cv::subtract(imageThresh, maskEllipse, burr);

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
                        cv::rectangle(image, boundingRect, cv::Scalar(0));
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
        //检测划痕
        cv::Mat filterImage;
        cv::bitwise_and(image, image, filterImage, openingCircle);
        cv::threshold(filterImage, filterImage, thresholdValue2, 255, cv::THRESH_BINARY);
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
                        cv::rectangle(image, boundingRect, cv::Scalar(0));
                        Box box(boundingRect.tl().x, boundingRect.tl().y, boundingRect.br().x, boundingRect.br().y, 1.0, 0);
                        box.distance0uter = distanceOuter;
                        box.distanceInner = distanceInner;
                        outputframe->batchDefects->at(i).push_back(Defect("huahen", box));
                    }
                }

            }
        }

        outputframe->circles->push_back(circle);
    }
}

void ImageProcess::detectGeneral(std::vector<cv::Mat>& images, BatchResultFramePtr outputframe, int thresholdValue1, int thresholdValue2, bool inv) {
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
        // 二值化
        auto type = cv::ThresholdTypes::THRESH_BINARY;
        if (inv == true) {
            type = cv::ThresholdTypes::THRESH_BINARY_INV;
        }
        cv::threshold(image, imageThresh, thresholdValue1, 255, type);


        // 开运算
        cv::Mat element;
        element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat openingCircle;
        cv::morphologyEx(imageThresh, openingCircle, cv::MORPH_OPEN, element);

        // 拟合椭圆
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(openingCircle, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
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
            std::cout << "找不到料" << std::endl;
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
}
