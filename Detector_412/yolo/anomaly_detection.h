#pragma once
#include "../common/common_frame.h"
#include "../common/config_manager.h"
#include "common_include.h"
#include "utils.h"
#include "kernel_function.h"


class AnomalyDetection {
public:
    AnomalyDetection(const std::string& configPath);
    ~AnomalyDetection();
    bool init();
    void preprocess();
    bool infer();
    void postprocess();
    void setInputData(std::shared_ptr<void>& data);
    std::vector<std::vector<Defect>> getObjectss() const;
    void reset();
    void check();


private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::function<void()> preprocess_fun_;

    void registerPreprocessFun(std::function<void()> cb);
    void gray_preprocess();
    void rgb_preprocess();
    void gray_resize_preprocess();
    void rgb_resize_preprocess();
    utils::InitParameter param_;
    nvinfer1::Dims output_dims_;
    std::vector<std::vector<Defect>> batchObjects_;
    int totalObjects;

    utils::AffineMat dst2src_;
    utils::AffineMat mask2src_;

    float maskH_;
    float maskW_;

    // input
    unsigned char* input_src_device_;
    float* input_resize_device_;
    float* input_rgb_device_;
    float* input_norm_device_;
    float* input_hwc_device_;
    // output
    float* output_src_device_;
    float* output_src_device1_;

    unsigned char* output_src_mask_;
    unsigned char* output_src_host_;
};

