#include "anomaly_detection.h"

AnomalyDetection::AnomalyDetection(const std::string& configPath) {
    auto configManager = ConfigManager::GetInstance(configPath);
    auto node = configManager->getConfig();
    auto imageType = node["anomaly_detection"]["imagetype"].as<std::string>();


    param_.iou_thresh = node["anomaly_detection"]["nmsthres"].as<float>();
    param_.conf_thresh = node["anomaly_detection"]["confidencethres"].as<float>();
    param_.dynamic_batch = node["anomaly_detection"]["dynamicbatch"].as<bool>();
    param_.batch_size = node["anomaly_detection"]["batchsize"].as<int>();
    param_.src_h = node["anomaly_detection"]["imagesizeH"].as<int>();
    param_.src_w = node["anomaly_detection"]["imagesizeW"].as<int>();
    param_.dst_h = node["anomaly_detection"]["modelsize"].as<int>();
    param_.dst_w = node["anomaly_detection"]["modelsize"].as<int>();


    if (imageType == "gray") {
        if ((param_.src_h == param_.dst_h) && (param_.src_w == param_.dst_w)) {
            this->registerPreprocessFun(std::bind(&AnomalyDetection::gray_preprocess, this));
        }
        else {
            this->registerPreprocessFun(std::bind(&AnomalyDetection::gray_resize_preprocess, this));
        }
    }
    else if (imageType == "rgb") {
        if ((param_.src_h == param_.dst_h) && (param_.src_w == param_.dst_w)) {
            this->registerPreprocessFun(std::bind(&AnomalyDetection::rgb_preprocess, this));
        }
        else {
            this->registerPreprocessFun(std::bind(&AnomalyDetection::rgb_resize_preprocess, this));
        }
    }
    else {
        throw("error image type: %s", imageType.c_str());
    }

    param_.model_path = configPath + node["anomaly_detection"]["modelpath"].as<std::string>();
    param_.input_output_names = { "images",  "output0" };
    auto labelFileName = node["anomaly_detection"]["labelfile"].as<std::string>();
    auto labelFilePath = configPath + "/labels/" + labelFileName;
    std::ifstream file(labelFilePath);
    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            param_.class_names.push_back(line);
        }
        file.close();
    }
    else {
        // printf("unable to load label path:%s\n", labelFilePath.c_str());
        throw("unable to load label path:%s\n", labelFilePath.c_str());
    }

    // input
    input_src_device_ = nullptr;
    input_resize_device_ = nullptr;
    input_rgb_device_ = nullptr;
    input_norm_device_ = nullptr;
    input_hwc_device_ = nullptr;

    CHECK(cudaMalloc(&input_rgb_device_, param_.batch_size * 3 * param_.src_h * param_.src_w * sizeof(float)));
    CHECK(cudaMalloc(&input_resize_device_, param_.batch_size * 3 * param_.dst_h * param_.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&input_norm_device_, param_.batch_size * 3 * param_.dst_h * param_.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&input_hwc_device_, param_.batch_size * 3 * param_.dst_h * param_.dst_w * sizeof(float)));

    // output
    output_src_device_ = nullptr;
    batchObjects_.resize(param_.batch_size);

}

AnomalyDetection::~AnomalyDetection() {
    CHECK(cudaFree(input_resize_device_));
    CHECK(cudaFree(input_rgb_device_));
    CHECK(cudaFree(input_norm_device_));
    CHECK(cudaFree(input_hwc_device_));
    // output
    CHECK(cudaFree(output_src_device_));
    delete[] output_src_host_;
}

bool AnomalyDetection::init()
{
    std::cout << "model_path:" << param_.model_path << std::endl;
    std::vector<unsigned char> trt_file = utils::loadModel(param_.model_path);
    if (trt_file.empty())
    {
        return false;
    }
    // std::unique_ptr<nvinfer1::IRuntime> runtime =
    //     std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    this->runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    // if (runtime == nullptr)
    if (this->runtime_ == nullptr)
    {
        return false;
    }
    // this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));
    this->engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(this->runtime_->deserializeCudaEngine(trt_file.data(), trt_file.size()));


    if (this->engine_ == nullptr)
    {
        return false;
    }
    this->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(this->engine_->createExecutionContext());
    if (this->context_ == nullptr)
    {
        return false;
    }
    auto inputIndex = this->engine_->getBindingIndex("input");
    if (param_.dynamic_batch) // for some models only support static mutil-batch. eg: yolox
    {
        this->context_->setBindingDimensions(inputIndex, nvinfer1::Dims4(param_.batch_size, 3, param_.dst_h, param_.dst_w));
    }
    auto outputIndex = this->engine_->getBindingIndex("output");
    output_dims_ = this->context_->getBindingDimensions(outputIndex);
    totalObjects = output_dims_.d[1];
    assert(param_.batch_size <= output_dims_.d[0]);
    int outputArea = 1;
    for (int i = 1; i < output_dims_.nbDims; i++)
    {
        if (output_dims_.d[i] != 0)
        {
            outputArea *= output_dims_.d[i];
        }
    }
    maskH_ = output_dims_.d[2];
    maskW_ = output_dims_.d[3];

    CHECK(cudaMalloc(&output_src_device_, param_.batch_size * outputArea * sizeof(float)));
    CHECK(cudaMalloc(&output_src_mask_, param_.batch_size * outputArea * sizeof(unsigned char)));
    output_src_host_ = new unsigned char[param_.batch_size * outputArea * sizeof(unsigned char)];

    float a = float(param_.dst_h) / param_.src_h;
    float b = float(param_.dst_w) / param_.src_w;
    float scale = a < b ? a : b;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * param_.src_w + param_.dst_w + scale - 1) * 0.5,
        0.f, scale, (-scale * param_.src_h + param_.dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    
    a = float(maskH_) / param_.src_h;
    b = float(maskW_) / param_.src_w;
    scale = a < b ? a : b;
    cv::Mat src2mask = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * param_.src_w + maskW_ + scale - 1) * 0.5,
        0.f, scale, (-scale * param_.src_h + maskH_ + scale - 1) * 0.5);
    cv::Mat mask2src = cv::Mat::zeros(2, 3, CV_32FC1);

    cv::invertAffineTransform(src2dst, dst2src);
    cv::invertAffineTransform(src2mask, mask2src);

    dst2src_.v0 = dst2src.ptr<float>(0)[0];
    dst2src_.v1 = dst2src.ptr<float>(0)[1];
    dst2src_.v2 = dst2src.ptr<float>(0)[2];
    dst2src_.v3 = dst2src.ptr<float>(1)[0];
    dst2src_.v4 = dst2src.ptr<float>(1)[1];
    dst2src_.v5 = dst2src.ptr<float>(1)[2];

    mask2src_.v0 = mask2src.ptr<float>(0)[0];
    mask2src_.v1 = mask2src.ptr<float>(0)[1];
    mask2src_.v2 = mask2src.ptr<float>(0)[2];
    mask2src_.v3 = mask2src.ptr<float>(1)[0];
    mask2src_.v4 = mask2src.ptr<float>(1)[1];
    mask2src_.v5 = mask2src.ptr<float>(1)[2];
    return true;
}

void AnomalyDetection::registerPreprocessFun(std::function<void()> cb) {
    preprocess_fun_ = cb;
}

void AnomalyDetection::preprocess() {
    preprocess_fun_();
}

void AnomalyDetection::gray_preprocess()
{
    gray2rgbDevice(param_.batch_size, input_src_device_, param_.dst_w, param_.dst_h,
        input_rgb_device_, param_.dst_w, param_.dst_h);
    normDevice(param_.batch_size, input_rgb_device_, param_.dst_w, param_.dst_h,
        input_norm_device_, param_.dst_w, param_.dst_h, param_);
    hwc2chwDevice(param_.batch_size, input_norm_device_, param_.dst_w, param_.dst_h,
        input_hwc_device_, param_.dst_w, param_.dst_h);
}

void AnomalyDetection::gray_resize_preprocess()
{
    gray2rgbDevice(param_.batch_size, input_src_device_, param_.src_w, param_.src_h,
        input_rgb_device_, param_.src_w, param_.src_h);
    resizeDevice(param_.batch_size, input_rgb_device_, param_.src_w, param_.src_h,
        input_resize_device_, param_.dst_w, param_.dst_h, 114, dst2src_);
    normDevice(param_.batch_size, input_resize_device_, param_.dst_w, param_.dst_h,
        input_norm_device_, param_.dst_w, param_.dst_h, param_);
    hwc2chwDevice(param_.batch_size, input_norm_device_, param_.dst_w, param_.dst_h,
        input_hwc_device_, param_.dst_w, param_.dst_h);
}

void AnomalyDetection::rgb_preprocess()
{
    normDevice(param_.batch_size, input_src_device_, param_.dst_w, param_.dst_h,
        input_norm_device_, param_.dst_w, param_.dst_h, param_);
    hwc2chwDevice(param_.batch_size, input_norm_device_, param_.dst_w, param_.dst_h,
        input_hwc_device_, param_.dst_w, param_.dst_h);
}

void AnomalyDetection::rgb_resize_preprocess()
{
    resizeDevice(param_.batch_size, input_src_device_, param_.src_w, param_.src_h,
        input_resize_device_, param_.dst_w, param_.dst_h, 114, dst2src_);
    normDevice(param_.batch_size, input_resize_device_, param_.dst_w, param_.dst_h,
        input_norm_device_, param_.dst_w, param_.dst_h, param_);
    hwc2chwDevice(param_.batch_size, input_norm_device_, param_.dst_w, param_.dst_h,
        input_hwc_device_, param_.dst_w, param_.dst_h);
}


void AnomalyDetection::post_process() {
    decodeAnomalyDevice(param_.batch_size, output_src_device_, output_src_mask_, maskH_, maskW_, 0.5);
    CHECK(cudaMemcpy(output_src_device_, output_src_host_, maskH_ * maskW_ * param_.batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned char* ptr = output_src_host_;
    for (std::size_t i = 0; i < param_.batch_size; i++) {
        cv::Mat image(maskH_, maskW_, CV_8UC3, ptr);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (std::size_t j = 0; j < contours.size(); j++) {
            cv::Rect boundingRect = cv::boundingRect(contours[j]);
            float x_lt = dst2src_.v0 * boundingRect.tl().x + dst2src_.v1 * boundingRect.tl().y + dst2src_.v2;
            float y_lt = dst2src_.v3 * boundingRect.tl().x + dst2src_.v4 * boundingRect.tl().y + dst2src_.v5;
            float x_rb = dst2src_.v0 * boundingRect.br().x + dst2src_.v1 * boundingRect.br().y + dst2src_.v2;
            float y_rb = dst2src_.v3 * boundingRect.br().x + dst2src_.v4 * boundingRect.br().y + dst2src_.v5;
            Box box(x_lt, y_lt, x_rb, y_rb, 1, 0);
            batchObjects_[i].push_back(Defect("yichang", box));
        }
    }
}