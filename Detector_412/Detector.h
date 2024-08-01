#pragma once
#include <mutex>
#include<map>
#include "common/common_frame.h"
#include <memory>



//class __declspec(dllexport) Detector
//{
//private:
//    Detector(std::string& configPath);
//    class Impl;
//    std::unique_ptr<Impl> pimpl;
//
//    static Detector* instance;
//    static std::mutex mutex_;
//
//protected:
//    ~Detector();
//public:
//    Detector(Detector& other) = delete;
//    void operator=(const Detector&) = delete;
//    static Detector* GetInstance(std::string configPath);
//
//public:
//    bool process(int uuid, ImageFrame& inputframe, ResultFrame& resultframe);
//};


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