#pragma once
#include <mutex>
#include "yaml-cpp/yaml.h"


class ConfigManager
{
private:
    static std::map<std::string, std::shared_ptr<ConfigManager>> instances_;
    static std::mutex mutex_;
    std::mutex configMutex_;
    ConfigManager(const std::string& configPath);
    YAML::Node node_;

//protected:
//    ~ConfigManager(){}
public:
    ConfigManager(ConfigManager &other) = delete;
    void operator=(const ConfigManager &) = delete;
    static std::shared_ptr<ConfigManager> GetInstance(const std::string& configPath);
    YAML::Node getConfig();

};

//class ConfigManager
//{
//private:
//    YAML::Node node_;
//public:
//    ConfigManager(std::string& configPath);
//    ~ConfigManager() {}
//    YAML::Node getConfig();
//};

