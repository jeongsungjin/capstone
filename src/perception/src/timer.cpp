#include "timer.h"
#include <iomanip>

Timer::Timer(const std::string& name)
    : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

Timer::~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
    std::cout << "[TIMER] " << std::left << std::setw(20) << name_ 
              << "took " << std::right << std::setw(5) << duration
              << " ms\n";
}
