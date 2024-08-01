#pragma once
#include "../common/common_frame.h"
#include "../common/config_manager.h"
#include "common_include.h"
#include "utils.h"
#include "kernel_function.h"

namespace yolo
{
    class YOLO
    {
    public:
        YOLO(std::string & configPath);
        ~YOLO();

    public:
        virtual bool init();
        virtual void check();
        virtual void copy(const std::vector<cv::Mat>& imgsBatch);
        virtual void preprocess();
        virtual void registerPreprocessFun(std::function<void()> cb);
        virtual void gray_preprocess();
        virtual void rgb_preprocess();
        virtual bool infer();
        virtual void postprocess(const size_t& batchSize);
        virtual void reset();
        virtual void setInputData(std::shared_ptr<void>& data);

    public:
        // std::vector<std::vector<utils::Box>> getObjectss() const;
        std::vector<std::vector<Defect>> getObjectss() const;


    protected:
        std::function<void()> preprocess_fun_;
        std::unique_ptr<nvinfer1::IRuntime> m_runtime;
        std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    protected:
        utils::InitParameter m_param;
        nvinfer1::Dims m_output_dims;   
        int m_output_area;
        int m_total_objects;
        // std::vector<std::vector<utils::Box>> m_objectss;
        std::vector<std::vector<Defect>> m_objectss;

        utils::AffineMat m_dst2src;     

        // input
        unsigned char* m_input_src_device;
        float* m_input_resize_device;
        float* m_input_rgb_device;
        float* m_input_norm_device;
        float* m_input_hwc_device;
        // output
        float* m_output_src_device;
        float* m_output_objects_device;
        float* m_output_objects_host;
        int m_output_objects_width;     
        int* m_output_idx_device;      
        float* m_output_conf_device;
    };

    class YOLOV8 : public YOLO {
    public:
        YOLOV8(std::string& configPath);
        ~YOLOV8();
        virtual bool init();
        virtual void postprocess(const size_t& batchSize);
    private:
        float* m_output_src_transpose_device;
    };
}