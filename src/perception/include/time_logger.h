#ifndef __TIME_LOGGER_H__
#define __TIME_LOGGER_H__

#include <fstream>
#include <mutex>
#include <string>

class TimeLogger {
public:
    static TimeLogger& instance() {
        static TimeLogger inst;
        return inst;
    }

    void log(const std::string& tag, double ms);

private:
    TimeLogger();
    ~TimeLogger();

    std::ofstream file_;
    std::mutex mtx_;
};

#endif
