#include "time_logger.h"
#include <ctime>

TimeLogger::TimeLogger() {
    file_.open("/home/ctrl/capstone/src/perception/logging/timelog.csv", std::ios::out | std::ios::app);
    file_ << "tag,time_ms\n";
}

TimeLogger::~TimeLogger() {
    file_.close();
}

void TimeLogger::log(const std::string& tag, double ms) {
    std::lock_guard<std::mutex> lock(mtx_);
    file_ << tag << "," << ms << "\n";
}
