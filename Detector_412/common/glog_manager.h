#pragma once
#include <mutex>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h" // 包含文件日志sink
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/daily_file_sink.h"
class GlogManager
{
private:
	static std::map<std::string, std::shared_ptr<GlogManager>> instances_;
	static std::mutex mutex_;
	std::mutex logMutex_;
	std::shared_ptr<spdlog::logger> logger_;
	GlogManager(const std::string& configPath);

public:
	GlogManager(GlogManager& other) = delete;
	void operator=(const GlogManager&) = delete;
	static std::shared_ptr<GlogManager> GetInstance(const std::string& configPath);
	std::shared_ptr<spdlog::logger> getLogger();

};