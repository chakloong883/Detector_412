# 412��Ŀ�㷨�ӿ�˵��
## ����Ҫ��
��Ҫ׼�����»������������»���д�뻷��Ŀ¼�У���������ʱ����dll
- [YAMLCPP](https://github.com/jbeder/yaml-cpp)
- [OpenCV4](https://opencv.org/releases/)
- [Tensorrt8.6](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
- [CUDA11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive)
## ���Ա�����
- �õ��������Ҫ�޸�Debug64��Release64���������Ա�λ�ã��޸İ���Ŀ¼����Ŀ¼������������
    - YAMLCPP_X64.prop
    - Tensorrt8_X64.prop
    - OpenCV4_Debug_X64.prop

## Դ���ļ�˵��
```
��  Detector.cpp                         #�ӿ�
��  Detector.h                           #�ӿ�������
��  dllmain.cpp                          #���ɽӿڳ���
��  framework.h                          
��  pch.cpp
��  pch.h
��  OpenCV4_Debug_X64.props              #OpenCV4_Debug���Ա�
��  OpenCV4_Release_X64.props            #OpenCV4_Release���Ա�
��  CUDA 11.6.props                      #CUDA���Ա�
��  Tensorrt8_X64.props                  #Tensorrt8���Ա�
��  YAMLCPP_X64.props                    #YAMLCPP_X64_Debug���Ա�
��  YAMLCPP_X64_Release.props            #YAMLCPP_X64_Release���Ա�
��  Readme.md                            #˵���ļ�
����common
��      common_frame.h                   #��������õĽṹ��
��      common_frame_inside.h            #�ڲ����ýṹ��
��      common_queue.h                   #�ڲ����ö��ж���
��      config_manager.cpp               #���ù����࣬����ģʽ
��      config_manager.h                 
��      image_process.cpp                #ͼ�����࣬������ͳ�ļ�ⷽ��
��      image_process.h
��      queue_splitter.h                 #���з���������ʱδʹ��
��      ThreadPool.h                     #�̳߳أ���ʱδʹ��
��      thread_safe_queue.h              #��ȫ����
��      tools.cpp                        #���õ�һЩ���ߣ���ʱ���ʱ����������
��      tools.h
��
����DetectorThread
��      detector_thread.cpp              #����߳���
��      detector_thread.h
��
����yolo                                 # yolo�����ļ�
        common_include.h    
        decode_yolov8.cu
        decode_yolov8.h
        kernel_function.cu               # һЩ���ú˺�������rgbת�Ҷȣ����͸�ʴ�˺�������ֵ���˺���
        kernel_function.h
        utils.cpp
        utils.h
        yolo.cpp
        yolo.h
```



## �ӿ�ʹ��˵��
### ���ɽӿ�
visual studioѡ��Releaseģʽ���Ҽ������̣�������ɣ������ڹ���Ŀ¼��x64/Release����dll��
### �ӿ�׼��
�������ɵ�Detector_412.dll��Detector_412.lib������Ҫ׼�����������Ŀ��ļ��������Ľӿڴ��Ӧ��׼�������ļ���
```
��  cudart64_110.dll
��  Detector.h
��  Detector_412.dll
��  Detector_412.lib
��  nvinfer.dll
��  opencv_world440.dll
��  yaml-cpp.dll
��
����common
        common_frame.h
```
### ������ԽӵĹ����ӿ�
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


### �ӿڵ���ʾ��
- �ڵ���ǰ��Ӧ׼������Ҫ�������ļ���
- ׼����ImageFrame��ResultFrame�����߳��е���process�������ýӿ�֧�ֶ��̵߳��ã�����ResultFrame�Ľ����
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
    std::string configPath = "D://zzl//u��//bosch_camera_config";
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

### �ӿڵ����ڲ�����
- ���������Detector��ʱ�������DetectorThread��Init���������������ļ�Ҫ�󣬴���Ŀ����������ͳ�������ͼ�񿽱���cuda�࣬���½�����̣߳����̻߳᲻�ϴ�ͼ�������ȡͼ����ɿ�������⼰������̡�
    - ���detector_thread.cpp
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
            // �½�ͼ�񿽱�����
            copyImageToCuda_ = std::make_shared<tools::CopyImageToCuda>(batchSize_, imageQueue_, batchImageQueue_);
            // ��������߳�
            assert(!detectThread_.joinable());
            detectThread_ = std::thread(&DetectorThread::detectThread, this);
            return true;
        }

        ```
- ���������Detector��process����ʱ��
    - ����DetectorThread��push��������ͼƬѹ��ͼ������У�����߳��ں�̨�ж���ͼ�񣬽��п���������������
    - ����get�����������ȴ�����߳��������
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
## �����ļ�˵��
1. �����ļ�·�����ļ�����
    ```
    ��  app_config.yaml #app�����ļ�
    ��
    ����labels
    ��      bosch_labels.txt #��ǩ�ļ�
    ��      cnc_labels.txt
    ��      coco_labels.txt
    ��
    ����models
            bosch_yolov8_dynamic.onnx
            bosch_yolov8_dynamic.trt # ����ģ��
            onnx2trt_dynamic.bash # ����ת���ű�
    ```
2. ģ��ת���ű�˵��
    ```
    //onnx2trt_dynamic.bash
    //��onnxתΪ1280��С����Ķ�̬batchģ��
    trtexec   --onnx=bosch_yolov8_dynamic.onnx   --saveEngine=bosch_yolov8_1280_dynamic.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:3x3x1280x1280 --fp16
    ```
3. app_config.yaml˵��
    ```
    //app_config.yaml
    # Ŀ�����㷨���ã����û��������ʹ��Ŀ�����㷨
    object_detection:
      objectdetector: yolov8                        #����yolov8���뵼����ģ�ͣ�����дyolov8������yolov5������ģ�ͣ�����дyolov5
      classnum: 5                                   # �������
      labelfile: bosch_labels.txt                   # ��ǩ�ļ����ƣ�������labels��
      nmsthres: 0.45                                # nms��ֵ
      confidencethres: 0.25                         # ���Ŷ���ֵ
      dynamicbatch: true                            # �Ƿ��Ƕ�̬batch
      imagetype: rgb                                # ����ͼƬ���ͣ�����rgb������дrgb������gray������дgray
      imagesize: 1280                               # ͼƬ�ߴ��С
      modelsize: 1280                               # ģ�ͽ���ͼƬ��С
      batchsize: 1                                  # batchsize���ã����̵߳��õ������йأ����ǵ����̵߳��ã�ֻ����1
      modelpath: /models/bosch_1280_dynamic.trt     # Ҫ���õ�trtģ��
    # ��ͳ����㷨���ã����û��������ʹ�ô�ͳ����㷨
    tradition_detection:
      imagetype: rgb                                # ����ͼƬ���ͣ�����rgb������дrgb������gray������дgray
      imagesize: 1280                               # ͼƬ�ߴ��С
      batchsize: 1                                  # batchsize���ã����̵߳��õ������йأ����ǵ����̵߳��ã�ֻ����1
      method: general                               # ʹ�õĴ�ͳ��������ǰ��ѡgeneral(ֻ���Բ)��detectmaociyijiaohuahen�����Բ��ë��ë˿���ۣ�
      inv: true                                     # ���Բʱ��ֵ���Ƿ���Ҫ���������ͻ�������Ĳ��֣�
      thresholdvalue1: 200                          # ���Բ����ֵ
      thresholdvalue2: 120                          # ��⻮�۵���ֵ

    shrinkratio: 0.37                               # ����������1um��Ӧ��������
    
    # ȱ�ݹ�������
    defect_filter:
      momianposun:                                  # ��ע��ȱ�����û������Ĭ�ϲ�����
        shrink: 0                                   # ������С������0����������С��0��������������0�Ȳ�����Ҳ������
        judge:                                      # �ж�ng��judge�б�û������Ĭ�ϲ�����
        -
           obj: length                              # ��ע��ȱ�����������糤�ȣ���ȣ������
           NG: ">0"                                 # �ﵽNG����Ҫ������
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