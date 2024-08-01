# 412项目接口说明
## 与软件对接的接口
```
//common/common_frame.h
struct ImageFrame {
	std::shared_ptr<void> buffer;//图像buffer
	int imageWidth; // 图像宽度
	int imageHeight;// 图像高度
	int channelNum; // 通道数量
    std::string uuid; //独立的uuid
};

struct ResultFrame {
    std::shared_ptr<std::vector<Defect>> defects; // 检测结果
    std::string uuid;
};


//Detector.h
class __declspec(dllexport) Detector
{
public:
    Detector(std::string& configPath);
    ~Detector();
    bool process(ImageFrame& inputframe, ResultFrame& resultframe);
private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};
```

## 接口调用示例
```
#include <iostream>
#include "Detector.h"
#include <thread>
#include <cassert>


void threadTask(Detector* detector) {
    unsigned char* data = new unsigned char[1280*1280*3];
    ImageFrame frame{
        std::shared_ptr<unsigned char>(data, [](unsigned char* p) {delete[]p; }),
        1280,
        1280,
        3,
    };
    ResultFrame outputframe;
    detector->process(frame, outputframe);
    std::cout << "size:" << outputframe.defects->size() << std::endl;
}


int main()
{
    std::string s;
    std::cin >> s;
    std::string configPath = "D://zzl//u盘//bosch_camera_config";
    std::string imagePath = "D://zzl/data/test_rgb.jpg";

    Detector* detector1 = new Detector(configPath);

    for (int i = 0; i < 100; i++) {
        std::thread newthread1(threadTask, detector1);
        newthread1.detach();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    std::cin >> s;
    delete detector1;
}
```

## 流程
- 当软件构造Detector类时，会调用DetectorThread的Init函数，新建三个线程，分别是图像拷贝至cuda线程，推理线程，以及后处理线程。
    - 详见detector_thread.cpp
        ```c++
        bool DetectorThread::Init(std::string& configPath) {
            if (!createDetector(configPath)) {
                std::cout << "create detector failed!" << std::endl;
                return false;
            }
            //拷贝图片至cuda线程
            assert(!copyImageToCudaThread_.joinable());
            copyImageToCudaThread_ = std::thread(&DetectorThread::copyImageToCudaThread, this);
            //检测线程
            assert(!detectThread_.joinable());
            detectThread_ = std::thread(&DetectorThread::detectThread, this);
            //后处理线程
            assert(!postprocessThread_.joinable());
            postprocessThread_ = std::thread(&DetectorThread::postprocessThread, this);
            return true;
        }
        ```
- 当软件调用Detector的process方法时。
    - 调用DetectorThread的push方法，将图片压入cuda拷贝队列，调用get方法等待推理结果。
    - 拷贝线程将图像拷贝至显卡上。
    - 推理线程对图像进行推理（含目标检测的预处理，推理，目标检测的后处理）。
    - 后处理线程整合目标检测结果，压入resultFrameMap_中，通知等待线程获取结果。
        ```
        bool process(std::string& uuid, ImageFrame& inputframe, ResultFrame& resultframe) {
            if (!detectorThread_->push(inputframe)) {
                std::cout << "push失败" << std::endl;
                return false;
            }
            if (!detectorThread_->get(resultframe, uuid)) {
                std::cout << "get失败" << std::endl;
                return false;
            }
            return true;
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
        ```
## 配置文件说明
1. 配置文件路径的文件安排
```
│  app_config.yaml #app配置文件
│
├─labels
│      bosch_labels.txt #标签文件
│      cnc_labels.txt
│      coco_labels.txt
│
└─models
        bosch_yolov8_dynamic.onnx
        bosch_yolov8_dynamic.trt # 推理模型
        onnx2trt_dynamic.bash # 推理转换脚本
```
2. 模型转换脚本说明
    ```
    //onnx2trt_dynamic.bash
    //将onnx转为1280大小输入的动态batch模型
    trtexec   --onnx=bosch_yolov8_dynamic.onnx   --saveEngine=bosch_yolov8_1280_dynamic.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:3x3x1280x1280 --fp16
    ```
3. app_config.yaml说明
    ```
    //app_config.yaml
    object_detecion:
      objectdetector: yolov8 #若用yolov8代码导出的模型，则填写yolov8，若用yolov5导出的模型，则填写yolov5
      classnum: 5 #类别数量
      labelfile: bosch_labels.txt #标签文件名称，安排在labels下
      nmsthres: 0.45 #nms阈值
      confidencethres: 0.25 #置信度阈值
      dynamicbatch: true #是否是动态batch
      imagetype: rgb #输入图片类型，若是rgb类型填写rgb，若是gray类型填写gray
      imagesize: 1280 #图片尺寸大小
      modelsize: 1280 #模型接受图片大小
      batchsize: 1 #batchsize设置，与线程调用的数量有关，若是单个线程调用，只能填1
      modelpath: /models/bosch_1280_dynamic.trt #要调用的trt模型

    crop:
      cropwidth: 1280
      cropheight: 1280
      overlap: 0.2
    ```