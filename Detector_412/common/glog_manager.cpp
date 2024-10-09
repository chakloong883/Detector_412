#include "glog_manager.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
void checkCreateFolder(std::string&& path) {
    fs::path folderPath = path;
    if (std::filesystem::exists(folderPath)) {
        std::cout << "Folder already exists." << std::endl;
    }
    else {
        // 创建文件夹
        if (std::filesystem::create_directories(folderPath)) {
            std::cout << "Folder created successfully." << std::endl;
        }
        else {
            std::cerr << "Failed to create folder." << std::endl;
        }
    }
}


std::map<std::string, std::shared_ptr<GlogManager>> GlogManager::instances_;

std::mutex GlogManager::mutex_;


GlogManager::GlogManager(const std::string& configPath) {
    checkCreateFolder("D://tmp");
    std::size_t position = configPath.find_last_of("/\\");

    //截取最后一个路径分隔符后面的字符串即为文件名
    std::string foldername = configPath.substr(position + 1, configPath.length() - position - 1);
    std::string loggerPath = "D://tmp//" + foldername + ".txt";
    auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(loggerPath, 2, 00);

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    spdlog::logger* logger = new spdlog::logger("multi_sink", { console_sink, file_sink });
    logger_ = std::shared_ptr<spdlog::logger>(logger);
}


std::shared_ptr<GlogManager> GlogManager::GetInstance(const std::string& configPath) {
    std::lock_guard<std::mutex> lock(mutex_); // 线程安全
    auto it = instances_.find(configPath);
    if (it == instances_.end()) {
        std::shared_ptr<GlogManager> instance(new GlogManager(configPath));
        instances_[configPath] = instance;
        return instance;
    }
    return it->second;
}

std::shared_ptr<spdlog::logger> GlogManager::getLogger() {
    std::lock_guard<std::mutex> lock(logMutex_); // 线程安全
    return logger_;
}