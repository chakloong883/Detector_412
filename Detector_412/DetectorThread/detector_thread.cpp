#include <chrono>
#include<cuda_runtime_api.h>
#include "detector_thread.h"


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




void DetectorThread::copyImageToCudaThread() {
    void* data = nullptr;
    unsigned char* dataPoint = nullptr;
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
            dataPoint = static_cast<unsigned char*>(data);
        }
        cudaMemcpy(dataPoint, imageBuf.get(), bufferSize, cudaMemcpyHostToDevice);
        dataPoint += bufferSize;
        batchuuid.push_back(uuid);
        imagePos.push_back({0,0,true});
        if (frameCount % batchSize_ == batchSize_ - 1) {
            BatchImageFramePtr batchImageFrame(new BatchImageFrame({
                    std::shared_ptr<void>(data, [](void* p) {CHECK(cudaFree(p)); }),
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




void DetectorThread::detectThread() {
    while (!detectThreadShouldExit_) {
        auto batchImage = batchImageQueue_->Dequeue();
        if (!batchImage) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto batchuuid = (*batchImage)->batchuuid;
        auto buffer = (*batchImage)->buffer;
        // TODO ���Ƿ�װ���̳߳�
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

        BatchResultFramePtr outputFrame(new BatchResultFrame({
            std::make_shared<std::vector<std::vector<Defect>>>(yolo_->getObjectss()),
            imagesPos,
            batchuuid
            }));

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
            for (std::size_t i = 0; i < defects.size(); i++) {
                defects[i].box.addBias(rowBias, colBias);
            }
            if (isLast) {
                std::lock_guard<std::mutex> lock(resultFrameMapMutex_);
                resultFrameMap_[uuid] = ResultFrame({ std::make_shared<std::vector<Defect>>(defects), uuid} );
                resultFrameMapCV_.notify_all();
            }
        }

    }
}

bool DetectorThread::createDetector(std::string& configPath) {
    auto configManager = ConfigManager::GetInstance(configPath);
    auto node = configManager->getConfig();
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
    return true;
}
