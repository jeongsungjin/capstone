#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>
#include <string>
#include <iostream>

class Timer {
public:
    Timer(const std::string& name);
    ~Timer();

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

#endif
