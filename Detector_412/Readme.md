# 412项目算法接口说明
## 环境要求
需要将以下环境写入环境目录中，方便运行时加载
- YAMLCPP
- OpenCV4
- Tensorrt8.6
- CUDA11.6
## 属性表配置
- 拿到代码后需要修改Debug64和Release64的以下属性表位置，修改包含目录、库目录及附加依赖项
    - YAMLCPP_X64.prop
    - Tensorrt8_X64.prop
    - OpenCV4_Debug_X64.prop

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
    # 目标检测算法配置，如果没有配置则不使用目标检测算法
    object_detection:
      objectdetector: yolov8                        #若用yolov8代码导出的模型，则填写yolov8，若用yolov5导出的模型，则填写yolov5
      classnum: 5                                   # 类别数量
      labelfile: bosch_labels.txt                   # 标签文件名称，安排在labels下
      nmsthres: 0.45                                # nms阈值
      confidencethres: 0.25                         # 置信度阈值
      dynamicbatch: true                            # 是否是动态batch
      imagetype: rgb                                # 输入图片类型，若是rgb类型填写rgb，若是gray类型填写gray
      imagesize: 1280                               # 图片尺寸大小
      modelsize: 1280                               # 模型接受图片大小
      batchsize: 1                                  # batchsize设置，与线程调用的数量有关，若是单个线程调用，只能填1
      modelpath: /models/bosch_1280_dynamic.trt     # 要调用的trt模型
    # 传统检测算法配置，如果没有配置则不使用传统检测算法
    tradition_detection:
      imagetype: rgb                                # 输入图片类型，若是rgb类型填写rgb，若是gray类型填写gray
      imagesize: 1280                               # 图片尺寸大小
      batchsize: 1                                  # batchsize设置，与线程调用的数量有关，若是单个线程调用，只能填1
      method: general                               # 使用的传统方法，当前可选general(只检测圆)，detectmaociyijiaohuahen（检测圆及毛刺毛丝划痕）
      inv: true                                     # 检测圆时二值化是否需要反向操作（突出较亮的部分）
      thresholdvalue1: 200                          # 检测圆的阈值
      thresholdvalue2: 120                          # 检测划痕的阈值

    shrinkratio: 0.37                               # 内缩比例，1um对应多少像素
    
    # 缺陷过滤配置
    defect_filter:
      momianposun:                                  # 关注的缺陷类别，没有配置默认不出框
        shrink: 0                                   # 内缩大小，大于0是往内缩，小于0是往外扩，等于0既不内缩也不外扩
        judge:                                      # 判断ng的judge列表，没有配置默认不出框
        -
           obj: length                              # 关注的缺陷特征，例如长度，宽度，面积等
           NG: ">0"                                 # 达到NG所需要的条件
      momianquemo:
        shrink: 20
        judge:
        -
           obj: length
           NG: ">0"
      momianliewen:
        shrink: 140
        judge:
        -
           obj: length
           NG: ">0"
      momianyiwu_heidian:
        shrink: 20
        judge:
        -
           obj: area
           NG: ">2500"
      momianyayin_kuaizhuang:
        shrink: 20
        judge:
        -
           obj: area
           NG: ">2500"
      momianyayin_xianzhuang:
        shrink: 20
        judge:
        -
           obj: area
           NG: ">2500"
      momianyayin_tiaozhuang:
        shrink: 20
        judge:
        -
           obj: area
           NG: ">2500"
      momianjiafei:
        shrink: 20
        judge:
        -
           obj: area
           NG: ">2500"

      momianzangwu:
        shrink: 20
        judge:
        -
           obj: area
           NG: ">2500"


      momianaokeng:
        shrink: 20
        judge:
        -
           obj: length
           NG: ">0"

      momiansiwangluolu:
        shrink: 20
        judge:
          -
             obj: length
             NG: ">=99"
          -
             obj: width
             NG: ">=99"
      momianzhehen:
        shrink: 20
        judge:
        -
           obj: length
           NG: ">0"
    ```
## 与软件对接的接口
```
//common/common_frame.h
struct ImageFrame {
	std::shared_ptr<void> buffer;
	int imageWidth;
	int imageHeight;
	int channelNum;
};


struct Box
{
    float left, top, right, bottom, width = 0.0, height = 0.0, distance0uter = 0.0, distanceInner = 0.0, confidence;
    int label;
    Box (float left, float top, float right, float bottom, float confidence, int label):left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
    Box() = default;
    void addBias(int rowBias, int colBias) {
        left += colBias;
        right += colBias;
        top += rowBias;
        bottom += rowBias;
    }
};

struct Defect {
    std::string defectName;
    Box box;
    std::string objFocus;
    float objValue = 0.0;
    Defect() = default;
    Defect (std::string defectName, Box box):defectName(defectName), box(box){}
};

struct ResultFrame {
    std::shared_ptr<std::vector<Defect>> defects;
    bool NG;
    std::string NGStateMent;
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
- 当软件构造Detector类时，会调用DetectorThread的Init函数，根据配置文件要求，创建目标检测器，传统检测器，图像拷贝至cuda类，并新建检测线程，该线程会不断从图像队列中取图，完成拷贝、检测及后处理过程。
    - 详见detector_thread.cpp
        ```c++
        bool DetectorThread::Init(std::string& configPath) {
            configManager_ = ConfigManager::GetInstance(configPath);
            node_ = configManager_->getConfig();
            //this->registerTraditionFun(std::bind(&ImageProcess::detectGeneral, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
            if (node_["object_detecion"]) {
                batchSize_ = node_["object_detecion"]["batchsize"].as<int>();
                if (!createObjectDetection(configPath)) {
                    std::cout << "create detector failed!" << std::endl;
                    return false;
                }
            }
            if (node_["tradition_detection"]) {
                batchSize_ = node_["tradition_detection"]["batchsize"].as<int>();
                if (node_["tradition_detection"]["method"]) {
                    needTraditionDetection_ = true;
                    if (node_["tradition_detection"]["method"].as<std::string>() == "detectmaociyijiaohuahen") {
                        //this->registerTraditionFun(std::bind(&ImageProcess::detectMaociBatchImages, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
                        traditionalDetection_ = std::make_shared<ImageProcess::DetectMaociHuahenBatchImages>(configPath);
                    }
                    else {
                        traditionalDetection_ = std::make_shared<ImageProcess::DetectGeneralBatchImages>(configPath);
                    }
                }
            }
            // 新建图像拷贝用例
            copyImageToCuda_ = std::make_shared<tools::CopyImageToCuda>(batchSize_, imageQueue_, batchImageQueue_);
            // 开启检测线程
            assert(!detectThread_.joinable());
            detectThread_ = std::thread(&DetectorThread::detectThread, this);
            return true;
        }

        ```
- 当软件调用Detector的process方法时。
    - 调用DetectorThread的push方法，将图片压入图像队列中，检测线程在后台中读入图像，进行拷贝、推理、后处理。。
    - 调用get方法阻塞，等待检测线程完成推理。
        ```
        // Detector.cpp
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
        // DetectorThread.cpp
        bool DetectorThread::push(ImageFrameInside& frame) {
            if (!imageQueue_->Enqueue(std::make_shared<ImageFrameInside>(frame))) {
                std::cout << "image queue full!" << std::endl;
                return false;
            }
            else {
                std::cout << "imageFrame add!, size: " << imageQueue_->size() << std::endl;
            }
            return true;
        }

        bool DetectorThread::get(ResultFrameInside& frame, std::string& uuid) {
            std::unique_lock<std::mutex> lock(resultFrameMapMutex_);
            resultFrameMapCV_.wait(lock, [&] {return resultFrameMap_.find(uuid) != resultFrameMap_.end(); });
            frame = resultFrameMap_[uuid];
            resultFrameMap_.erase(uuid);
            return true;
        }
        ```