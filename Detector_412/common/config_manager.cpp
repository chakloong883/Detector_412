#include "config_manager.h"
#include <string>

//ConfigManager* ConfigManager::instance{nullptr};
//std::mutex ConfigManager::mutex_;

std::map<std::string, std::shared_ptr<ConfigManager>> ConfigManager::instances_;
std::mutex ConfigManager::mutex_;

ConfigManager::ConfigManager(const std::string& configPath) {
    std::string appPath = configPath  + "/app_config.yaml";
    YAML::Node node;
    try {
        node_ = YAML::LoadFile(appPath);
    } catch(YAML::BadFile &e) {
        throw "read app_config error!\n";
    }
}

std::shared_ptr<ConfigManager> ConfigManager::GetInstance(const std::string& configPath) {
    std::lock_guard<std::mutex> lock(mutex_); // 线程安全

    auto it = instances_.find(configPath);
    if (it == instances_.end()) {
        std::shared_ptr<ConfigManager> instance(new ConfigManager(configPath));
        instances_[configPath] = instance;
        return instance;
    }
    return it->second;
}

YAML::Node ConfigManager::getConfig() {
    return node_;
}