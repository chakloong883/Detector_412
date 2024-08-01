# 412��Ŀ�ӿ�˵��
## ������ԽӵĽӿ�
```
//common/common_frame.h
struct ImageFrame {
	std::shared_ptr<void> buffer;//ͼ��buffer
	int imageWidth; // ͼ����
	int imageHeight;// ͼ��߶�
	int channelNum; // ͨ������
    std::string uuid; //������uuid
};

struct ResultFrame {
    std::shared_ptr<std::vector<Defect>> defects; // �����
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

## �ӿڵ���ʾ��
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

## ����
- ���������Detector��ʱ�������DetectorThread��Init�������½������̣߳��ֱ���ͼ�񿽱���cuda�̣߳������̣߳��Լ������̡߳�
    - ���detector_thread.cpp
        ```c++
        bool DetectorThread::Init(std::string& configPath) {
            if (!createDetector(configPath)) {
                std::cout << "create detector failed!" << std::endl;
                return false;
            }
            //����ͼƬ��cuda�߳�
            assert(!copyImageToCudaThread_.joinable());
            copyImageToCudaThread_ = std::thread(&DetectorThread::copyImageToCudaThread, this);
            //����߳�
            assert(!detectThread_.joinable());
            detectThread_ = std::thread(&DetectorThread::detectThread, this);
            //�����߳�
            assert(!postprocessThread_.joinable());
            postprocessThread_ = std::thread(&DetectorThread::postprocessThread, this);
            return true;
        }
        ```
- ���������Detector��process����ʱ��
    - ����DetectorThread��push��������ͼƬѹ��cuda�������У�����get�����ȴ���������
    - �����߳̽�ͼ�񿽱����Կ��ϡ�
    - �����̶߳�ͼ�����������Ŀ�����Ԥ��������Ŀ����ĺ�����
    - �����߳�����Ŀ��������ѹ��resultFrameMap_�У�֪ͨ�ȴ��̻߳�ȡ�����
        ```
        bool process(std::string& uuid, ImageFrame& inputframe, ResultFrame& resultframe) {
            if (!detectorThread_->push(inputframe)) {
                std::cout << "pushʧ��" << std::endl;
                return false;
            }
            if (!detectorThread_->get(resultframe, uuid)) {
                std::cout << "getʧ��" << std::endl;
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
    object_detecion:
      objectdetector: yolov8 #����yolov8���뵼����ģ�ͣ�����дyolov8������yolov5������ģ�ͣ�����дyolov5
      classnum: 5 #�������
      labelfile: bosch_labels.txt #��ǩ�ļ����ƣ�������labels��
      nmsthres: 0.45 #nms��ֵ
      confidencethres: 0.25 #���Ŷ���ֵ
      dynamicbatch: true #�Ƿ��Ƕ�̬batch
      imagetype: rgb #����ͼƬ���ͣ�����rgb������дrgb������gray������дgray
      imagesize: 1280 #ͼƬ�ߴ��С
      modelsize: 1280 #ģ�ͽ���ͼƬ��С
      batchsize: 1 #batchsize���ã����̵߳��õ������йأ����ǵ����̵߳��ã�ֻ����1
      modelpath: /models/bosch_1280_dynamic.trt #Ҫ���õ�trtģ��

    crop:
      cropwidth: 1280
      cropheight: 1280
      overlap: 0.2
    ```