#include "timer.h"

Timer::Timer(const std::string& name)
    : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

Timer::~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
    std::cout << "[TIMER] " << name_ << " took " << duration << " ms\n";
}
